#!/usr/bin/env python3
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.patch import concat_pad_data_collator
from internvl.train.arguments import DataTrainingArguments
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


def _ensure_single_process_dist() -> None:
    if not dist.is_available() or not dist.is_initialized():
        dist.get_rank = lambda: 0
        dist.get_world_size = lambda: 1


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a strict first-batch signal probe for the internal 1-GPU D-MoLE lane."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--dmole-arch-path", required=True)
    parser.add_argument("--autoencoder-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--force-image-size", type=int, default=448)
    parser.add_argument("--down-sample-ratio", type=float, default=0.5)
    parser.add_argument("--conv-style", default="internlm2-chat")
    parser.add_argument("--min-dynamic-patch", type=int, default=1)
    parser.add_argument("--max-dynamic-patch", type=int, default=6)
    parser.add_argument("--use-llm-lora", type=int, default=8)
    parser.add_argument("--use-backbone-lora", type=int, default=8)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float32", "fp32", "bfloat16", "bf16"),
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
    )
    return parser


def _resolve_torch_dtype(dtype_name: str) -> tuple[str, torch.dtype]:
    normalized = dtype_name.strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return normalized, dtype_map[normalized]


def _configure_tokenizer(model_path: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = max_seq_length
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
    return tokenizer, num_new_tokens, img_context_token_id


def _configure_model(
    args: argparse.Namespace,
    tokenizer,
    num_new_tokens: int,
    img_context_token_id: int,
):
    normalized_dtype_name, torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    config = InternVLChatConfig.from_pretrained(args.model_name_or_path)
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = args.attn_implementation
    else:
        config.llm_config._attn_implementation = args.attn_implementation
    config.template = args.conv_style
    config.select_layer = -1
    config.dynamic_image_size = True
    config.use_thumbnail = True
    config.ps_version = "v2"
    config.min_dynamic_patch = args.min_dynamic_patch
    config.max_dynamic_patch = args.max_dynamic_patch
    config.use_dmole = True
    config.dmole_arch_path = args.dmole_arch_path
    config.autoencoder_path = args.autoencoder_path
    config.task_id = args.task_id

    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        config=config,
    )
    model.img_context_token_id = img_context_token_id

    patch_size = model.config.vision_config.patch_size
    model.config.force_image_size = args.force_image_size
    model.num_image_token = int(
        (args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio**2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))

    with open(args.dmole_arch_path, "r", encoding="utf-8") as handle:
        dmole_arch = json.load(handle)
    vision_dmole_arch = {
        key.split("vision_model.", 1)[1]: value
        for key, value in dmole_arch.items()
        if key.startswith("vision_model.")
    }
    llm_dmole_arch = {
        key.split("language_model.", 1)[1]: value
        for key, value in dmole_arch.items()
        if key.startswith("language_model.")
    }

    for parameter in model.language_model.parameters():
        parameter.requires_grad = False
    for parameter in model.vision_model.parameters():
        parameter.requires_grad = False
    for parameter in model.mlp1.parameters():
        parameter.requires_grad = False

    if not isinstance(model.vision_model, PeftModel):
        model.wrap_backbone_lora(
            r=args.use_backbone_lora,
            lora_alpha=2 * args.use_backbone_lora,
            dmole_arch=vision_dmole_arch,
        )
    if not isinstance(model.language_model, PeftModel):
        model.wrap_llm_lora(
            r=args.use_llm_lora,
            lora_alpha=2 * args.use_llm_lora,
            dmole_arch=llm_dmole_arch,
        )

    model.language_model.set_expert_masks(args.task_id)
    model.language_model.freeze_old_experts(args.task_id)
    model.vision_model.set_expert_masks(args.task_id)
    model.vision_model.freeze_old_experts(args.task_id)
    model.train()
    return model.cuda(), normalized_dtype_name, torch_dtype


def _build_batch(args: argparse.Namespace, tokenizer, model, batch_torch_dtype: torch.dtype):
    data_args = DataTrainingArguments(
        max_seq_length=args.max_seq_length,
        force_image_size=args.force_image_size,
        down_sample_ratio=args.down_sample_ratio,
        pad2square=False,
        conv_style=args.conv_style,
        meta_path=args.meta_path,
        use_data_resampling=False,
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch,
        normalize_type="imagenet",
        use_packed_ds=False,
    )
    dataset = build_datasets(
        data_args,
        tokenizer,
        model,
        group_by_length=False,
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch,
        normalize_type="imagenet",
    )
    batch = concat_pad_data_collator([dataset[0]])
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            if key == "pixel_values":
                batch[key] = value.to(device="cuda", dtype=batch_torch_dtype)
            else:
                batch[key] = value.to(device="cuda")
    return batch


def _compute_probe_report(args: argparse.Namespace) -> dict:
    _ensure_single_process_dist()
    tokenizer, num_new_tokens, img_context_token_id = _configure_tokenizer(
        args.model_name_or_path,
        args.max_seq_length,
    )
    model, normalized_dtype_name, batch_torch_dtype = _configure_model(
        args, tokenizer, num_new_tokens, img_context_token_id
    )
    batch = _build_batch(args, tokenizer, model, batch_torch_dtype)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=batch_torch_dtype)
        if batch_torch_dtype == torch.bfloat16
        else nullcontext()
    )
    with autocast_context:
        outputs = model(**batch)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    report = {
        "model_name_or_path": args.model_name_or_path,
        "meta_path": args.meta_path,
        "dmole_arch_path": args.dmole_arch_path,
        "autoencoder_path": args.autoencoder_path,
        "torch_dtype": normalized_dtype_name,
        "attn_implementation": args.attn_implementation,
        "input_ids_shape": list(batch["input_ids"].shape),
        "labels_shape": list(batch["labels"].shape),
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "image_flags_shape": list(batch["image_flags"].shape),
        "supervised_tokens": int((batch["labels"] != -100).sum().item()),
        "img_context_token_id": int(img_context_token_id),
        "num_img_context_tokens": int(
            (batch["input_ids"] == img_context_token_id).sum().item()
        ),
        "loss": None if loss is None else float(loss.detach().float().cpu()),
        "loss_isfinite": None
        if loss is None
        else bool(torch.isfinite(loss.detach()).cpu()),
        "loss_isnan": None
        if loss is None
        else bool(torch.isnan(loss.detach()).cpu()),
    }

    if loss is not None and torch.isfinite(loss.detach()):
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms = []
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad_norms.append(parameter.grad.detach().float().norm())
        if grad_norms:
            total_grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms))
            report["grad_norm"] = float(total_grad_norm.cpu())
            report["grad_norm_isfinite"] = bool(torch.isfinite(total_grad_norm).cpu())
        else:
            report["grad_norm"] = None
            report["grad_norm_isfinite"] = False
    else:
        report["grad_norm"] = None
        report["grad_norm_isfinite"] = False

    return report


def _write_report(report: dict, output_json: str) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    report = _compute_probe_report(args)
    _write_report(report, args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["supervised_tokens"] <= 0:
        raise SystemExit("FATAL: no supervised tokens were found in the initial D-MoLE batch.")
    if report["loss"] is None or not report["loss_isfinite"]:
        raise SystemExit("FATAL: initial D-MoLE loss is non-finite.")
    if report["loss"] <= 0.0:
        raise SystemExit("FATAL: initial D-MoLE loss is not strictly positive.")
    if not report["grad_norm_isfinite"]:
        raise SystemExit("FATAL: initial D-MoLE gradient norm is non-finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
