import os

import torch
from peft import PeftModel
from transformers import Trainer
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available, logging

from internvl.dist_utils import rank0_print
from internvl.model.internvl_chat.modeling_internvl_chat import \
    InternVLChatModel
from internvl.train.dataset import process_inputs

logger = logging.get_logger(__name__)


def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


def save_lora_weights(
    model: InternVLChatModel,
    output_dir: str,
):
    """
    Save LoRA weights from the model.

    Args:
        model: The model to save LoRA weights from.
        output_dir: The directory to save the weights to.
    """
    os.makedirs(output_dir, exist_ok=True)

    lora_state_dict = {}
    config = (
        model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    )

    if config.get("use_backbone_lora", 0) > 0:
        backbone_lora_state_dict = {
            k: v
            for k, v in model.vision_model.state_dict().items()
            if "lora_" in k and all(dim > 0 for dim in v.shape)
        }
        lora_state_dict.update(
            {f"vision_model.{k}": v for k, v in backbone_lora_state_dict.items()}
        )
        logger.info(f"Saved {len(backbone_lora_state_dict)} backbone LoRA parameters")

    if config.get("use_llm_lora", 0) > 0:
        llm_lora_state_dict = {
            k: v
            for k, v in model.language_model.state_dict().items()
            if "lora_" in k and all(dim > 0 for dim in v.shape)
        }
        lora_state_dict.update(
            {f"language_model.{k}": v for k, v in llm_lora_state_dict.items()}
        )
        logger.info(f"Saved {len(llm_lora_state_dict)} LLM LoRA parameters")

    if not config.get("freeze_mlp", False):
        mlp_state_dict = {k: v for k, v in model.mlp1.state_dict().items()}
        lora_state_dict.update({f"mlp1.{k}": v for k, v in mlp_state_dict.items()})

    logger.info(f"Total LoRA parameters to save: {len(lora_state_dict)}")
    logger.info(f"LoRA state dict keys: {list(lora_state_dict.keys())[:10]}...")

    with open(os.path.join(output_dir, "lora_state_dict_keys.txt"), "w") as f:
        for key in lora_state_dict.keys():
            f.write(f"{key}\n")

    if lora_state_dict:
        torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.bin"))
        logger.info(f"LoRA weights saved to {output_dir}")
    else:
        logger.warning("No LoRA weights found to save!")

    return lora_state_dict


class CustomTrainer(Trainer):
    def _save(self, output_dir: str, state_dict=None):
        if (
            not self.model.config.use_llm_lora
            and not self.model.config.use_backbone_lora
        ):
            super()._save(output_dir)
            return

        lora_state_dict = save_lora_weights(self.model, output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(
            f"Model configuration, LoRA weights, tokenizer, and metadata saved to {output_dir}"
        )

        super()._save(output_dir, state_dict=lora_state_dict)


class DMoLETrainer(CustomTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # === start of DMoLE logic ===
        # get current task_id
        if hasattr(model, "module"):
            real_model = model.module
            current_task_id = model.module.config.task_id
        else:
            real_model = model
            current_task_id = model.config.task_id

        if current_task_id is not None:
            try:
                # Set no experts for sequence representation computation
                if hasattr(model, "module"):
                    model.module.set_expert_masks([])
                else:
                    model.set_expert_masks([])

                # create a copy of inputs for processing, avoid modifying the original data
                inputs_copy = {k: v for k, v in inputs.items()}
                if labels is not None:
                    inputs_copy["labels"] = labels  # restore labels for process_inputs

                new_inputs = process_inputs(inputs_copy, real_model, self.tokenizer)

                # get sequence representations
                if hasattr(model, "module"):
                    seq_reps = model.module.extract_sequence_feature(**new_inputs)
                    task_ids = model.module.get_task_ids(
                        seq_reps, mode="training", current_task_id=current_task_id
                    )
                    model.module.set_expert_masks(task_ids)
                else:
                    seq_reps = model.extract_sequence_feature(**new_inputs)
                    task_ids = model.get_task_ids(
                        seq_reps, mode="training", current_task_id=current_task_id
                    )
                    model.set_expert_masks(task_ids)

                logger.info(
                    f"Training: Selected task ids: {task_ids} (current task: {current_task_id})"
                )

            except Exception as e:
                rank0_print(
                    f"DMoLE processing failed: {e}, falling back to current task only"
                )
                # if error, use current task only
                if hasattr(model, "module"):
                    model.module.set_expert_masks([current_task_id])
                else:
                    model.set_expert_masks([current_task_id])
        # === end of DMoLE logic ===

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
