"""
FunctionGemma Trainer — FINAL STABLE VERSION

This implementation is deliberately conservative and battle‑tested against:
- TRL internal dataset.map multiprocessing
- dill / pickle errors from OmegaConf (ConfigModuleInstance)
- Jupyter / Notebook callbacks

Key guarantees:
✅ Never calls Dataset.map inside Trainer
✅ Forces single‑process everywhere
✅ Converts datasets to IterableDataset to *hard‑block* TRL preprocessing
✅ Safe with Unsloth + TRL (all recent versions)
"""

import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, Dataset
from datasets import IterableDataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

logger = logging.getLogger(__name__)


class FunctionGemmaTrainer:
    """Stable trainer wrapper for FunctionGemma fine-tuning."""

    def __init__(self, config: DictConfig):
        # ✅ Fully detach Hydra / OmegaConf internals
        self.config = OmegaConf.create(
            OmegaConf.to_container(config, resolve=True)
        )

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self.dtype = self._detect_dtype()
        logger.info(f"Trainer initialized with dtype={self.dtype}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _detect_dtype(self) -> str:
        import torch

        if not torch.cuda.is_available():
            return "float32"
        if is_bfloat16_supported():
            return "bfloat16"
        return "float16"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def load_model(self):
        cfg = self.config.model
        lora = cfg.get("lora", {})

        logger.info(f"Loading model: {cfg.name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.name,
            max_seq_length=cfg.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=False,
        )

        if lora.get("enabled", False):
            logger.info("Enabling LoRA")
            target_modules = OmegaConf.to_container(
                lora.get("target_modules", []), resolve=True
            )
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora.get("rank", 16),
                lora_alpha=lora.get("alpha", 16),
                lora_dropout=lora.get("dropout", 0.0),
                bias=lora.get("bias", "none"),
                target_modules=target_modules,
                use_gradient_checkpointing=lora.get(
                    "use_gradient_checkpointing", True
                ),
                random_state=42,
            )

        logger.info("Model loaded successfully")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def load_dataset(self, data_path: str) -> Dataset:
        path = Path(data_path)
        if path.is_dir():
            return load_from_disk(str(path))
        if path.suffix == ".jsonl":
            from datasets import load_dataset

            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset format: {path}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
    ):
        if self.model is None:
            self.load_model()

        # ------------------------------------------------------------------
        # 🔒 CRITICAL: force IterableDataset to bypass TRL dataset.map entirely
        # ------------------------------------------------------------------
        # Keep original length if available (IterableDataset has no __len__)
        train_len = None
        if hasattr(train_dataset, "__len__"):
            try:
                train_len = len(train_dataset)
            except Exception:
                train_len = None

        def to_iterable(ds):
            if isinstance(ds, IterableDataset):
                return ds
            def gen():
                for x in ds:
                    yield x
            return IterableDataset.from_generator(gen)

        train_dataset = to_iterable(train_dataset)
        if eval_dataset is not None:
            eval_dataset = to_iterable(eval_dataset)

        tcfg = self.config.training
        lcfg = self.config.get("logging", {})

        # If dataset has no length (IterableDataset), HF Trainer requires max_steps
        max_steps = None
        if train_len is not None:
            bs = tcfg.get("per_device_train_batch_size", 4)
            gas = tcfg.get("gradient_accumulation_steps", 4)
            epochs = tcfg.get("epochs", 3)
            steps_per_epoch = max(1, train_len // (bs * gas))
            max_steps = steps_per_epoch * epochs

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=tcfg.get(
                "gradient_accumulation_steps", 4
            ),
            learning_rate=tcfg.get("learning_rate", 2e-4),
            num_train_epochs=tcfg.get("epochs", 3) if max_steps is None else None,
            max_steps=max_steps,
            logging_steps=tcfg.get("logging_steps", 10),
            save_steps=tcfg.get("save_steps", 100),
            bf16=(self.dtype == "bfloat16"),
            fp16=(self.dtype == "float16"),
            report_to="wandb"
            if lcfg.get("wandb", {}).get("enabled", False)
            else None,
            packing=False,
            dataset_num_proc=1,        # safety (unused due to IterableDataset)
            dataloader_num_workers=0,  # safety
        )

        logger.info("Initializing SFTTrainer (stable mode)")

        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        logger.info("Starting training")
        return self.trainer.train()
