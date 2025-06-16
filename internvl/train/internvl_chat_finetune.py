# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
import logging
import os
import sys
import warnings
from functools import partial

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from PIL import Image, ImageFile, PngImagePlugin
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

from internvl.dist_utils import init_dist, rank0_print
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.patch import (
    concat_pad_data_collator,
    replace_internlm2_attention_class,
    replace_llama_attention_class,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_phi3_attention_class,
    replace_qwen2_attention_class,
    replace_train_dataloader,
    replace_train_sampler,
)
from internvl.train.arguments import DataTrainingArguments, ModelArguments
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl.train.dataset import build_datasets
from internvl.train.dataset_packed import packed_collate_fn
from internvl.train.trainer import CustomTrainer, DMoLETrainer

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters: {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from liger_kernel.transformers import (
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_qwen2,
        )

        from internvl.patch import apply_liger_kernel_to_internvit

        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    logger.info("Loading InternVLChatModel...")
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
        logger.info("Using flash_attention_2 for InternLM")
    else:
        config.llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
        logger.info("Using flash_attention_2 for LLaMA")
    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    config.use_dmole = model_args.use_dmole
    config.dmole_arch_path = model_args.dmole_arch_path
    config.autoencoder_path = model_args.autoencoder_path
    config.task_id = model_args.task_id
    model = InternVLChatModel.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config
    )
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info("Finished")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(
        f"model.config.vision_config.image_size: {model.config.vision_config.image_size}"
    )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from "
            f"{model.config.vision_config.image_size} "
            f"to {data_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio**2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone and not isinstance(model.vision_model, PeftModel):
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm and not isinstance(model.language_model, PeftModel):
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    vision_dmole_arch = llm_dmole_arch = None
    if model_args.use_dmole:
        rank0_print("Using D-MoLE model...")
        dmole_arch = json.load(open(model_args.dmole_arch_path, "r"))
        vision_dmole_arch = {
            ".".join(k.split(".")[1:]): v
            for k, v in dmole_arch.items()
            if "vision" in k
        }
        llm_dmole_arch = {
            ".".join(k.split(".")[1:]): v
            for k, v in dmole_arch.items()
            if "language_model" in k
        }

    if not isinstance(model.vision_model, PeftModel):
        if model_args.use_backbone_lora:
            model.wrap_backbone_lora(
                r=model_args.use_backbone_lora,
                lora_alpha=2 * model_args.use_backbone_lora,
                dmole_arch=vision_dmole_arch,
            )
            model.config.use_backbone_lora = model_args.use_backbone_lora
    else:
        rank0_print("Vision model continues training from Lora checkpoint.")
        model.vision_model.print_trainable_parameters()

    if not isinstance(model.language_model, PeftModel):
        if model_args.use_llm_lora:
            model.wrap_llm_lora(
                r=model_args.use_llm_lora,
                lora_alpha=2 * model_args.use_llm_lora,
                dmole_arch=llm_dmole_arch,
            )
            model.config.use_llm_lora = model_args.use_llm_lora
    else:
        rank0_print("Language model continues training from Lora checkpoint.")
        model.language_model.print_trainable_parameters()

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

    if model_args.use_dmole:
        if model_args.use_llm_lora:
            model.language_model.set_expert_masks(model_args.task_id)
            model.language_model.freeze_old_experts(model_args.task_id)
        if model_args.use_backbone_lora:
            model.vision_model.set_expert_masks(model_args.task_id)
            model.vision_model.freeze_old_experts(model_args.task_id)

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    TrainerClass = DMoLETrainer if model_args.use_dmole else CustomTrainer

    if model_args.use_dmole:
        rank0_print("Using D-MoLE trainer...")
    else:
        rank0_print("Using Custom trainer...")

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # For sequential LoRA finetune, we need to merge the LoRA weights after training on each task
        if model_args.use_backbone_lora and not model_args.use_dmole:
            model.vision_model.merge_and_unload()
            model.vision_model = model.vision_model.model
            model.config.use_backbone_lora = 0
        if model_args.use_llm_lora and not model_args.use_dmole:
            model.language_model.merge_and_unload()
            model.language_model = model.language_model.model
            model.config.use_llm_lora = 0

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
