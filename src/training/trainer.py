"""
FunctionGemma 训练器
基于 Unsloth 的 SFTTrainer 封装
"""

import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
from datasets import load_from_disk, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


class FunctionGemmaTrainer:
    """
    FunctionGemma 训练器

    封装 Unsloth FastLanguageModel 和 TRL SFTTrainer
    提供简化的训练接口
    """

    def __init__(self, config: DictConfig):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # 自动检测数据类型
        self.dtype = self._auto_detect_dtype()

        logger.info(f"训练器初始化完成，数据类型：{self.dtype}")

    def _auto_detect_dtype(self) -> str:
        """
        自动检测支持的精度

        Returns:
            数据类型字符串
        """
        import torch

        if not torch.cuda.is_available():
            logger.warning("未检测到 GPU，使用 float32")
            return "float32"

        # 检查 bfloat16 支持
        from unsloth import is_bfloat16_supported

        if is_bfloat16_supported():
            logger.info("检测到 bfloat16 支持")
            return "bfloat16"
        else:
            logger.info("使用 float16")
            return "float16"

    def load_model(self):
        """
        加载 FunctionGemma 模型
        """
        logger.info(f"加载模型：{self.config.model.name}")

        model_config = self.config.model
        lora_config = self.config.model.lora

        # 使用 Unsloth 加载模型
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.name,
            max_seq_length=model_config.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=False,  # 全精度训练
        )

        # 配置 LoRA
        if lora_config.enabled:
            logger.info("配置 LoRA...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config.rank,
                target_modules=lora_config.target_modules,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                bias=lora_config.bias,
                use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
                random_state=42,
            )

        logger.info("模型加载完成")

    def load_dataset(self, data_path: str) -> Dataset:
        """
        加载训练数据集

        Args:
            data_path: 数据集路径

        Returns:
            数据集对象
        """
        logger.info(f"加载数据集：{data_path}")

        data_path = Path(data_path)

        if data_path.is_dir():
            dataset = load_from_disk(str(data_path))
        elif data_path.suffix == ".jsonl":
            from datasets import load_dataset

            dataset = load_dataset("json", data_files=str(data_path), split="train")
        else:
            raise ValueError(f"不支持的数据格式：{data_path.suffix}")

        logger.info(f"数据集大小：{len(dataset)}")
        return dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
    ):
        """
        开始训练

        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集 (可选)
            output_dir: 输出目录
            callbacks: 回调函数列表
        """
        if self.model is None:
            self.load_model()

        training_config = self.config.training

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir or self.config.logging.output_dir,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            lr_scheduler_type=training_config.lr_scheduler_type,
            warmup_ratio=training_config.warmup_ratio,
            weight_decay=training_config.weight_decay,
            optim=training_config.optimizer,
            num_train_epochs=training_config.epochs,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            fp16=self.dtype == "float16",
            bf16=self.dtype == "bfloat16",
            seed=training_config.seed,
            report_to="wandb" if self.config.logging.wandb.enabled else "none",
        )

        # 创建 Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            callbacks=callbacks or [],
        )

        # 开始训练
        logger.info("开始训练...")
        train_result = self.trainer.train()

        # 打印训练指标
        metrics = train_result.metrics
        logger.info(f"训练完成，指标：{metrics}")

        return train_result

    def save_model(self, output_dir: str):
        """
        保存模型

        Args:
            output_dir: 输出目录
        """
        if self.trainer is None:
            raise RuntimeError("未进行训练，无法保存")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存模型到：{output_path}")

        # 保存模型和 tokenizer
        self.trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        logger.info("模型保存完成")

    def evaluate(self, eval_dataset: Dataset) -> dict:
        """
        评估模型

        Args:
            eval_dataset: 验证数据集

        Returns:
            评估指标
        """
        if self.trainer is None:
            raise RuntimeError("未进行训练，无法评估")

        logger.info("开始评估...")
        metrics = self.trainer.evaluate(eval_dataset)

        logger.info(f"评估完成，指标：{metrics}")
        return metrics

    def inference(self, text: str, max_new_tokens: int = 128) -> str:
        """
        推理测试

        Args:
            text: 输入文本
            max_new_tokens: 最大生成 token 数

        Returns:
            生成结果
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")

        # 切换到推理模式
        FastLanguageModel.for_inference(self.model)

        # 构建输入
        messages = [
            {
                "role": "developer",
                "content": "You are a model that can do function calling with the following functions",
            },
            {"role": "user", "content": text},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # 生成
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=self.config.model.inference.top_k,
            top_p=self.config.model.inference.top_p,
            temperature=self.config.model.inference.temperature,
        )

        # 解码
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 切换回训练模式
        FastLanguageModel.for_training(self.model)

        return result
