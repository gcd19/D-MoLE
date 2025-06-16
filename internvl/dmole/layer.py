import math
from typing import Any, Union

import torch
import torch.nn as nn
from peft.tuners.lora import Linear, LoraLayer


def update_lora_module(lora_module, adapter_name, in_features, r, num_experts):
    if adapter_name not in lora_module.keys():
        lora_module.update(
            {
                adapter_name: nn.ModuleList(
                    [nn.Linear(in_features, r, bias=False) for _ in range(num_experts)]
                )
            }
        )


class LoraLinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.disable = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable:
            return self.base_layer(x, *args, **kwargs)
        return super().forward(x, *args, **kwargs)

    def disable_lora(self):
        self.disable = True

    def enable_lora(self):
        self.disable = False


class LoraBlock(nn.Module):
    def __init__(self, in_features, out_features, r, dropout, alpha):
        super().__init__()

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.scaling = alpha / r

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        x = self.lora_A(x)
        x = self.lora_B(x)
        x = self.dropout(x)
        x = x * self.scaling
        return x


class DMoLELinear(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.num_experts = kwargs.pop("num_experts", 8)
        self.expert_ids = kwargs.pop("expert_ids", None)
        self.expert_masks = torch.ones(self.num_experts, dtype=torch.bool)

        self.disable = False

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora,
    ):
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update({adapter_name: lora_dropout_layer})

        update_lora_module(
            self.lora_A, adapter_name, self.in_features, r, self.num_experts
        )
        
        update_lora_module(
            self.lora_B, adapter_name, r, self.out_features, self.num_experts
        )

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        if init_lora_weights:
            self.reset_lora_parameters_MoE(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def set_expert_masks(self, expert_masks):
        self.expert_masks = expert_masks

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        batch_size, sequence_length, hidden_dim = x.shape
        input_dtype = x.dtype

        self.expert_masks = self.expert_masks.to(x.device)

        for adapter_name in self.active_adapters:
            final_hidden_states = torch.zeros(
                (batch_size, sequence_length, self.out_features),
                device=x.device,
                dtype=input_dtype,
            )

            selected_experts = torch.where(self.expert_masks)[0]

            for expert_idx in selected_experts:
                expert_lora_A = self.lora_A[adapter_name][expert_idx]
                expert_lora_B = self.lora_B[adapter_name][expert_idx]

                current_hidden_states = expert_lora_B(
                    expert_lora_A(self.lora_dropout[adapter_name](x))
                )

                final_hidden_states += current_hidden_states

            result += final_hidden_states * self.scaling[adapter_name]

        return result

    def get_expert_usage_count(self):
        return self.expert_usage_count

    def reset_lora_parameters_MoE(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name][i].weight, a=math.sqrt(5)
                )
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)

    def disable_lora(self):
        self.disable = True

    def enable_lora(self):
        self.disable = False
