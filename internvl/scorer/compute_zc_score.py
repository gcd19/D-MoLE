# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import importlib.util
import json
import logging
import math
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from pandas import DataFrame
from peft import PeftModel
from PIL import Image, ImageFile, PngImagePlugin
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

from internvl.dist_utils import init_dist, rank0_print
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
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
from internvl.train.trainer import CustomTrainer
from internvl.scorer.score_policy import (
    FALLBACK_SCORE_PROVENANCE,
    STRICT_SCORE_PROVENANCE,
)

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
MODEL_DTYPE_ENV = "DMOLE_MODEL_DTYPE"
REQUIRE_LONESTAR_PHYSICS_ENV = "DMOLE_REQUIRE_LONESTAR_PHYSICS"
FAIL_ON_SANITIZED_SCORE_ENV = "DMOLE_FAIL_ON_SANITIZED_SCORE"
GEODESIC_TRUST_SCALE = 1.0
SCORE_EPSILON = 1e-12

LONESTAR_PHYSICS_EXTENSION_ENV = "LONESTAR_PHYSICS_EXTENSION"


def load_lonestar_physics():
    try:
        import lonestar_physics as installed_lonestar_physics

        return installed_lonestar_physics, "python-import"
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    candidate_paths: list[Path] = []
    explicit_path = os.environ.get(LONESTAR_PHYSICS_EXTENSION_ENV, "").strip()
    if explicit_path:
        candidate_paths.append(Path(explicit_path))
    candidate_paths.extend(
        [
            workspace_root / "lonestar-physics" / "target" / "maturin" / "liblonestar_physics.so",
            workspace_root / "lonestar-physics" / "target" / "release" / "liblonestar_physics.so",
            workspace_root / "lonestar-physics" / "target" / "debug" / "liblonestar_physics.so",
        ]
    )

    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "lonestar_physics", candidate_path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["lonestar_physics"] = module
            spec.loader.exec_module(module)
            return module, str(candidate_path)
        except Exception:
            sys.modules.pop("lonestar_physics", None)
            continue

    return None, None


_lp, _lp_source = load_lonestar_physics()


def resolve_model_dtype() -> torch.dtype:
    raw_dtype = os.environ.get(MODEL_DTYPE_ENV, "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if raw_dtype not in mapping:
        raise ValueError(
            f"Unsupported {MODEL_DTYPE_ENV}={raw_dtype!r}; "
            "expected one of: bfloat16, float32, float16."
        )
    return mapping[raw_dtype]


def cast_floating_tensors(value, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, dtype) for item in value)
    return value


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


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def require_lonestar_physics() -> None:
    if env_flag(REQUIRE_LONESTAR_PHYSICS_ENV) and _lp is None:
        raise RuntimeError(
            "LoneStarPhysics invariance scoring is required but lonestar_physics "
            "could not be imported or loaded from an authoritative extension artifact."
        )


def score_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return max(dist.get_world_size(), 1)
    return max(torch.cuda.device_count(), 1)


def layer_modality(layer_name: str) -> str:
    if "language_model" in layer_name:
        return "language_model"
    if "vision_model" in layer_name:
        return "vision_model"
    return "other"


def normalize_distribution(values: list[float]) -> list[float] | None:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        return None
    return [max(value, 0.0) / total for value in values]


def collect_batch_layer_scores(model, target_modules: list[str]) -> tuple[dict[str, float], set[str]]:
    batch_scores: dict[str, float] = {}
    sanitized_layers: set[str] = set()

    for name, module in model.named_modules():
        if not any(name.endswith(target) for target in target_modules):
            continue
        total_norm = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                sanitized_layers.add(name)
            total_norm += grad.norm().item()
        batch_scores[name] = total_norm * score_world_size()
    return batch_scores, sanitized_layers


def summarize_modality_distributions(
    modality_distributions: list[list[float]],
) -> tuple[float, float]:
    if not modality_distributions:
        return 0.0, 0.0
    if _lp is None:
        return 0.0, 1.0

    if len(modality_distributions) == 1:
        return 0.0, 1.0

    center = _lp.fisher_rao.frechet_mean(modality_distributions, 32)
    variance = _lp.fisher_rao.frechet_variance(modality_distributions, center)
    variance = 0.0 if variance is None else float(variance)
    trust = float(
        _lp.fisher_rao.distance_to_trust(
            math.sqrt(max(variance, 0.0)),
            GEODESIC_TRUST_SCALE,
        )
    )
    return variance, trust


def build_score_frame(
    batch_layer_scores: list[dict[str, float]],
    sanitized_layers: set[str],
) -> tuple[DataFrame, dict[str, object]]:
    if not batch_layer_scores:
        raise RuntimeError("zero-cost proxy scoring produced no batches")

    layer_names = sorted(
        {
            layer_name
            for batch_scores in batch_layer_scores
            for layer_name in batch_scores
            if layer_modality(layer_name) != "other"
        }
    )
    if not layer_names:
        raise RuntimeError("zero-cost proxy scoring produced no eligible layer scores")

    per_layer_history = {layer_name: [] for layer_name in layer_names}
    modality_layers = {
        "language_model": sorted(
            [layer_name for layer_name in layer_names if layer_modality(layer_name) == "language_model"]
        ),
        "vision_model": sorted(
            [layer_name for layer_name in layer_names if layer_modality(layer_name) == "vision_model"]
        ),
    }
    modality_distributions = {"language_model": [], "vision_model": []}

    for batch_scores in batch_layer_scores:
        for layer_name in layer_names:
            per_layer_history[layer_name].append(float(batch_scores.get(layer_name, 0.0)))

        for modality_name, modality_layer_names in modality_layers.items():
            if not modality_layer_names:
                continue
            raw_values = [
                float(batch_scores.get(layer_name, 0.0))
                for layer_name in modality_layer_names
            ]
            normalized = normalize_distribution(raw_values)
            if normalized is not None:
                modality_distributions[modality_name].append(normalized)

    modality_summary: dict[str, dict[str, float]] = {}
    for modality_name, distributions in modality_distributions.items():
        variance, trust = summarize_modality_distributions(distributions)
        modality_summary[modality_name] = {
            "frechet_variance": float(variance),
            "trust": float(trust),
            "distribution_count": float(len(distributions)),
        }

    score_provenance = (
        STRICT_SCORE_PROVENANCE if _lp is not None else FALLBACK_SCORE_PROVENANCE
    )
    records = []
    for layer_name in layer_names:
        modality_name = layer_modality(layer_name)
        history = per_layer_history[layer_name]
        raw_score = float(math.fsum(history))
        share_history = []
        modality_layer_names = modality_layers.get(modality_name, [])
        for batch_scores in batch_layer_scores:
            raw_values = [
                float(batch_scores.get(name, 0.0))
                for name in modality_layer_names
            ]
            normalized = normalize_distribution(raw_values)
            if normalized is None:
                share_history.append(0.0)
                continue
            share_history.append(
                normalized[modality_layer_names.index(layer_name)]
                if layer_name in modality_layer_names
                else 0.0
            )

        share_mean = float(math.fsum(share_history) / len(share_history))
        share_variance = float(
            math.fsum((value - share_mean) ** 2 for value in share_history)
            / len(share_history)
        )
        stability_penalty = share_variance / max(share_mean * share_mean, SCORE_EPSILON)
        modality_trust = modality_summary.get(modality_name, {}).get("trust", 0.0)
        modality_frechet_variance = modality_summary.get(modality_name, {}).get(
            "frechet_variance", 0.0
        )
        authoritative_score = raw_score * modality_trust / (1.0 + stability_penalty)
        if not math.isfinite(authoritative_score):
            raise RuntimeError(
                f"authoritative score became non-finite for layer {layer_name}"
            )

        records.append(
            {
                "layer": layer_name,
                "score": authoritative_score,
                "raw_score": raw_score,
                "share_mean": share_mean,
                "share_variance": share_variance,
                "stability_penalty": stability_penalty,
                "modality": modality_name,
                "modality_trust": modality_trust,
                "modality_frechet_variance": modality_frechet_variance,
                "sanitized_nonfinite": layer_name in sanitized_layers,
                "score_batches": len(history),
                "score_provenance": score_provenance,
            }
        )

    frame = DataFrame(records).sort_values(
        by=["score", "raw_score", "layer"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    manifest = {
        "score_provenance": score_provenance,
        "batch_count": len(batch_layer_scores),
        "sanitized_layers": sorted(sanitized_layers),
        "modality_summary": modality_summary,
        "lonestar_physics_available": _lp is not None,
        "lonestar_physics_source": _lp_source,
    }
    return frame, manifest


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
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    model_dtype = resolve_model_dtype()
    logger.info(f"Using model dtype: {model_dtype}")

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

    if model_args.model_name_or_path is not None:
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
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=model_dtype, config=config
        )
    else:
        logger.info("Loading ViT-6B...")
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=model_dtype, config=vision_config
        )
        logger.info("Loading LLaMA...")
        llm_config = AutoConfig.from_pretrained(
            model_args.llm_path, trust_remote_code=True
        )
        if llm_config.model_type == "internlm2":
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = "flash_attention_2"  # for InternLM
            logger.info("Using flash_attention_2 for InternLM")
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
            logger.info("Using flash_attention_2 for LLaMA")
        llm = model_type.from_pretrained(
            model_args.llm_path,
            torch_dtype=model_dtype,
            config=llm_config,
            trust_remote_code=True,
        )
        logger.info("Building InternVLChatConfig...")
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info("Building InternVLChatModel...")
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
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

    group_by_length = getattr(training_args, "group_by_length", False)

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        model,
        group_by_length=group_by_length,
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

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

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

    # print trainable parameters
    rank0_print("Model trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()

    logger.info("Computing zero-cost proxy score...")

    portion = model_args.zc_proxy_score_portion
    max_batch_num = int(len(train_dataloader) * portion + 0.5)
    if max_batch_num <= 0:
        raise RuntimeError(
            "zc_proxy_score_portion is too small for the available dataloader; "
            "no scoring batches would execute."
        )

    logger.info(
        f"Zero-cost proxy score portion: {portion}, max_batch_num: {max_batch_num}"
    )

    model = trainer._wrap_model(model).to(trainer.args.device)

    require_lonestar_physics()
    target_modules = ["attention.wqkv", "attention.wo", "attn.qkv", "attn.proj"]
    model.train()
    model.zero_grad()
    supervised_token_total = 0
    batch_layer_scores: list[dict[str, float]] = []
    sanitized_layers: set[str] = set()
    for i, inputs in tqdm(enumerate(train_dataloader), total=max_batch_num):
        if i > max_batch_num:
            break

        model.zero_grad(set_to_none=True)
        inputs = trainer._prepare_inputs(inputs)
        inputs = cast_floating_tensors(inputs, model_dtype)
        supervised_tokens = int((inputs["labels"] != IGNORE_INDEX).sum().item())
        if supervised_tokens <= 0:
            raise RuntimeError(
                "zero-cost proxy batch contains no supervised assistant tokens; "
                "label masking collapsed and the score would be meaningless."
            )
        supervised_token_total += supervised_tokens
        loss = trainer.compute_loss(model, inputs)

        loss.backward()
        if torch.distributed.get_rank() == 0:
            current_batch_scores, current_sanitized = collect_batch_layer_scores(
                model, target_modules
            )
            if current_batch_scores:
                batch_layer_scores.append(current_batch_scores)
            sanitized_layers.update(current_sanitized)

    logger.info("Finished computing zero-shot proxy score.")
    logger.info(f"Total supervised tokens used for proxy scoring: {supervised_token_total}")

    logger.info("Computing zero-shot proxy score for each layer...")

    if torch.distributed.get_rank() == 0:
        if sanitized_layers:
            if env_flag(FAIL_ON_SANITIZED_SCORE_ENV):
                raise RuntimeError(
                    "zero-cost proxy scoring encountered non-finite gradients in "
                    "strict mode for layers: "
                    + ", ".join(sorted(sanitized_layers)[:10])
                )
            logger.warning(
                "Sanitized non-finite gradients before proxy scoring for %d layers.",
                len(sanitized_layers),
            )

        df, manifest = build_score_frame(batch_layer_scores, sanitized_layers)
        total_proxy_score = float(df["score"].sum())
        if total_proxy_score <= 0:
            logger.warning(
                "authoritative score surface collapsed to zero; downstream strict "
                "selection will fail closed unless explicitly downgraded"
            )

        output_path = Path(model_args.zc_proxy_score_save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        output_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        logger.info("Finished computing zero-shot proxy score for each layer.")


if __name__ == "__main__":
    main()
