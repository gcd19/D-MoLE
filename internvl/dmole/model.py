from itertools import chain

import regex as re
import torch
from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils import get_quantization_config

from internvl.dist_utils import rank0_print

from .layer import DMoLELinear


class MoELoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if lora_config.dmole_arch and not any(
            prefix in current_key for prefix in lora_config.dmole_arch
        ):
            rank0_print(
                f"Skipping {current_key} for D-MoLE because it is not in dmole_arch"
            )
            return

        if lora_config.dmole_arch and len(lora_config.dmole_arch[current_key]) == 0:
            rank0_print(
                f"Skipping {current_key} for D-MoLE because it is empty in dmole_arch"
            )
            return

        if current_key is None:
            raise ValueError(
                "Current Key shouldn't be `None` because it is the key of the target module"
            )

        rank0_print(f"Updating {current_key} for D-MoLE")

        expert_ids = num_experts = None
        if lora_config.dmole_arch:
            expert_ids = lora_config.dmole_arch.get(current_key)
            num_experts = len(expert_ids)

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(
            chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys())
        )
        target_name_key = next(
            filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
            current_key,
        )
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "expert_ids": expert_ids,
            "num_experts": num_experts,
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(
                self.model, method=quant_method
            )
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(
                lora_config, adapter_name, target, **kwargs
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def set_expert_masks(self, task_ids):
        if not isinstance(task_ids, list):
            task_ids = [task_ids]
        for _, layer in self.named_modules():
            if isinstance(layer, DMoLELinear):
                expert_masks = torch.tensor(
                    [
                        1 if expert_id in task_ids else 0
                        for expert_id in layer.expert_ids
                    ],
                    dtype=torch.bool,
                )
                layer.set_expert_masks(expert_masks)

    def freeze_old_experts(self, task_id):
        adapter_name = self.active_adapters[0]
        for _, layer in self.named_modules():
            if isinstance(layer, DMoLELinear):
                for i, expert_id in enumerate(layer.expert_ids):
                    if expert_id != task_id:
                        layer.lora_A[adapter_name][i].requires_grad_(False)
                        layer.lora_B[adapter_name][i].requires_grad_(False)
