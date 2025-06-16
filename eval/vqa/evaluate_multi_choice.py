import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from PIL import Image
from tqdm import tqdm

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

ds_collections = {
    "iconqa_test": {
        "root": "data/iconqa/multi_text_choice_test.jsonl",
        "max_new_tokens": 10,
        "min_new_tokens": 1,
    },
    "pmcvqa_test_clean": {
        "root": "data/pmcvqa/test_clean.jsonl",
        "max_new_tokens": 10,
        "min_new_tokens": 1,
    },
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_["pixel_values"] for _ in batches], dim=0)
    questions = [_["question"] for _ in batches]
    answers = [_["answer"] for _ in batches]
    image_paths = [_["image_path"] for _ in batches]
    options = [_["option"] for _ in batches]
    num_patches_list = [_["num_patches_list"] for _ in batches]
    return pixel_values, questions, answers, image_paths, options, num_patches_list


class MultiChoiceQADataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        prompt,
        input_size=224,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        f = open(root, "r", encoding="utf-8")
        self.data = [json.loads(line) for line in f.readlines()]
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_path = data["image"]
        question = data["question"]

        choices = data["choices"]
        answer = data["answer"]
        choice_list = []

        options = {}
        multiple_choices = ["A", "B", "C", "D", "E"]
        for i, c in enumerate(choices):
            choice_list.append("{}. {}".format(multiple_choices[i], c))
            options[multiple_choices[i]] = c
        choice_txt = "\n".join(choice_list)

        image = Image.open(image_path).convert("RGB")
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        question += "\n" + choice_txt
        question += "\n" + self.prompt

        return {
            "question": question,
            "pixel_values": pixel_values,
            "answer": multiple_choices[answer],
            "image_path": image_path,
            "option": options,
            "num_patches_list": len(pixel_values),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    prompt = "Answer with the option's letter from the given choices directly."
    print("prompt:", prompt)

    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        dataset = MultiChoiceQADataset(
            root=ds_collections[ds_name]["root"],
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (
            pixel_values,
            questions,
            answers,
            image_paths,
            options,
            num_patches_list,
        ) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]["max_new_tokens"],
                min_new_tokens=ds_collections[ds_name]["min_new_tokens"],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            preds = model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config,
                verbose=True,
            )
            preds = [post_process(pred, options[0]) for pred in preds]

            for question, pred, answer, image_path in zip(
                questions, preds, answers, image_paths
            ):
                outputs.append(
                    {
                        "question": question,
                        "answer": pred,
                        "gt_answers": answer,
                        "image_path": image_path,
                    }
                )

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {ds_name} ...")
            time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
            results_file = f"{ds_name}_{time_prefix}.jsonl"
            output_path = os.path.join(args.out_dir, results_file)
            with open(output_path, "w") as f:
                for output in merged_outputs:
                    f.write(json.dumps(output) + "\n")
            print("Results saved to {}".format(output_path))
            cnt = 0
            for item in merged_outputs:
                if item["answer"] == item["gt_answers"]:
                    cnt += 1
            print(f"Acc@1: {cnt / len(merged_outputs)}")

            summaries.append([args.checkpoint, ds_name, cnt / len(merged_outputs)])

    if torch.distributed.get_rank() == 0:
        summary_file_name = "_".join(args.checkpoint.split("/")[-2:])
        os.makedirs(os.path.join("results", "summary"), exist_ok=True)
        writer = open(
            os.path.join("results", "summary", f"{summary_file_name}.txt"), "a"
        )
        print(
            f"write results to file {os.path.join('results', 'summary', f'{summary_file_name}.txt')}"
        )
        for summary in summaries:
            print(summary)
            writer.write(f"{summary}\n")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="iconqa_test,pmcvqa_test_clean")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, args.checkpoint.split("/")[-1])
    os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)
    # assert args.batch_size == 1, "Only batch size 1 is supported"

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    if args.auto:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    kwargs = {"device_map": "auto"} if args.auto else {}
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = args.datasets[0]

    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f"[test] total_params: {total_params}B, use num_beams: {args.num_beams}")
    else:
        print(f"[test] total_params: {total_params}B")
    print(f"[test] image_size: {image_size}")
    print(f"[test] template: {model.config.template}")
    print(f"[test] dynamic_image_size: {args.dynamic}")
    print(f"[test] use_thumbnail: {use_thumbnail}")

    evaluate_chat_model()
