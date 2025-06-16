from dataclasses import dataclass, field
from typing import List

from peft.tuners.lora import LoraConfig


@dataclass
class MoELoraConfig(LoraConfig):
    dmole_arch: dict = field(default=None)
