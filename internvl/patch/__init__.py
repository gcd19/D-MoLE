# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

def _optional_accelerator_stub(symbol_name, dependency_name):
    def _missing_dependency(*args, **kwargs):
        raise ImportError(
            f"{symbol_name} requires optional dependency {dependency_name!r}, "
            f"but it is not installed in the current environment."
        )

    return _missing_dependency


try:
    from .internlm2_packed_training_patch import replace_internlm2_attention_class
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_internlm2_attention_class = _optional_accelerator_stub(
        'replace_internlm2_attention_class', 'flash_attn'
    )

from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit

try:
    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_llama2_attn_with_flash_attn = _optional_accelerator_stub(
        'replace_llama2_attn_with_flash_attn', 'flash_attn'
    )

try:
    from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_llama_attn_with_flash_attn = _optional_accelerator_stub(
        'replace_llama_attn_with_flash_attn', 'flash_attn'
    )

try:
    from .llama_packed_training_patch import replace_llama_attention_class
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_llama_attention_class = _optional_accelerator_stub(
        'replace_llama_attention_class', 'flash_attn'
    )
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import (concat_pad_data_collator,
                                dpo_concat_pad_data_collator,
                                pad_data_collator)
try:
    from .phi3_packed_training_patch import replace_phi3_attention_class
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_phi3_attention_class = _optional_accelerator_stub(
        'replace_phi3_attention_class', 'flash_attn'
    )

try:
    from .qwen2_packed_training_patch import replace_qwen2_attention_class
except ModuleNotFoundError as exc:
    if exc.name != 'flash_attn':
        raise
    replace_qwen2_attention_class = _optional_accelerator_stub(
        'replace_qwen2_attention_class', 'flash_attn'
    )
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm',
           'replace_llama2_attn_with_flash_attn',
           'replace_train_sampler',
           'replace_train_dataloader',
           'replace_internlm2_attention_class',
           'replace_qwen2_attention_class',
           'replace_phi3_attention_class',
           'replace_llama_attention_class',
           'pad_data_collator',
           'dpo_concat_pad_data_collator',
           'concat_pad_data_collator',
           'apply_liger_kernel_to_internvit']
