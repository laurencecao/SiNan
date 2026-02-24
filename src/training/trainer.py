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
            
            # 确保所有 LoRA 参数都有默认值
            if not hasattr(lora_config, 'dropout') or lora_config.dropout is None:
                lora_config.dropout = 0.0
            if not hasattr(lora_config, 'bias') or lora_config.bias is None:
                lora_config.bias = "none"
            if not hasattr(lora_config, 'use_gradient_checkpointing') or lora_config.use_gradient_checkpointing is None:
                lora_config.use_gradient_checkpointing = True
            
            # 确保 target_modules 是 Python 原生列表或字符串
            from omegaconf import OmegaConf
            target_modules_raw = lora_config.target_modules
            logger.info(f"原始 target_modules 类型：{type(target_modules_raw)}")
            
            # 使用 to_container 转换为原生 Python 对象
            target_modules = OmegaConf.to_container(target_modules_raw, resolve=True)
            logger.info(f"to_container 后类型：{type(target_modules)}")
            
            # 如果是列表，确保是纯 Python list
            if isinstance(target_modules, (list, tuple)):
                target_modules = [str(x) for x in target_modules]
            
            logger.info(f"最终 target_modules: {target_modules}")
            logger.info(f"最终 target_modules 类型：{type(target_modules)}")
            
            # 验证类型
            assert type(target_modules) in (list, tuple, str), f"target_modules 类型错误：{type(target_modules)}"
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config.rank,
                target_modules=target_modules,
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
        
        # 为缺失的配置项添加默认值
        from omegaconf import OmegaConf
        
        # 获取配置值，如果缺失则使用默认值
        per_device_train_batch_size = OmegaConf.select(training_config, 'per_device_train_batch_size', default=4)
        per_device_eval_batch_size = OmegaConf.select(training_config, 'per_device_eval_batch_size', default=4)
        gradient_accumulation_steps = OmegaConf.select(training_config, 'gradient_accumulation_steps', default=4)
        learning_rate = OmegaConf.select(training_config, 'learning_rate', default=2e-4)
        lr_scheduler_type = OmegaConf.select(training_config, 'lr_scheduler_type', default='cosine')
        warmup_ratio = OmegaConf.select(training_config, 'warmup_ratio', default=0.1)
        weight_decay = OmegaConf.select(training_config, 'weight_decay', default=0.01)
        optimizer = OmegaConf.select(training_config, 'optimizer', default='adamw_torch')
        num_train_epochs = OmegaConf.select(training_config, 'epochs', default=3)
        logging_steps = OmegaConf.select(training_config, 'logging_steps', default=10)
        save_steps = OmegaConf.select(training_config, 'save_steps', default=100)
        eval_steps = OmegaConf.select(training_config, 'eval_steps', default=100)
        seed = OmegaConf.select(training_config, 'seed', default=42)

        # 配置训练参数
        training_args_dict = {
            'output_dir': str(output_dir) if output_dir else str(self.config.logging.output_dir),
            'per_device_train_batch_size': int(per_device_train_batch_size),
            'per_device_eval_batch_size': int(per_device_eval_batch_size),
            'gradient_accumulation_steps': int(gradient_accumulation_steps),
            'learning_rate': float(learning_rate),
            'lr_scheduler_type': str(lr_scheduler_type),
            'warmup_ratio': float(warmup_ratio),
            'weight_decay': float(weight_decay),
            'optim': str(optimizer),
            'num_train_epochs': int(num_train_epochs),
            'logging_steps': int(logging_steps),
            'save_steps': int(save_steps),
            'eval_steps': int(eval_steps),
            'save_total_limit': 3,
            'fp16': self.dtype == "float16",
            'bf16': self.dtype == "bfloat16",
            'seed': int(seed),
            'report_to': "wandb" if self.config.logging.wandb.enabled else "none",
        }
        
        # 如果有评估数据集，添加评估策略
        if eval_dataset:
            training_args_dict['eval_strategy'] = 'steps'
        
        # 尝试使用 SFTConfig (trl >= 0.12.0)
        try:
            from trl import SFTConfig
            
            # 解决多进程 pickling 错误：强制使用单进程处理数据集
            training_args_dict['dataset_num_proc'] = 1
            training_args_dict['dataset_text_field'] = 'text'
            # 确保 dataloader 也是单进程
            training_args_dict['dataloader_num_workers'] = 0
            
            training_args = SFTConfig(**training_args_dict)
        except ImportError:
            # 兼容老版本 trl
            training_args = TrainingArguments(**training_args_dict)

        # 创建 Trainer - 动态检测支持的参数
        import inspect
        from datasets import Dataset as HFDataset
        
        def clean_dataset_item(item):
            """清理数据集中的不可序列化对象，只保留基本类型"""
            if isinstance(item, dict):
                return {k: clean_dataset_item(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [clean_dataset_item(v) for v in item]
            elif isinstance(item, (str, int, float, bool, type(None))):
                return item
            else:
                # 对于其他类型（如配置对象），转换为字符串
                logger.warning(f"数据集中发现非基本类型 {type(item).__name__}，转换为字符串")
                return str(item)
        
        # 转换数据集格式（如果是 list，转换为 HuggingFace Dataset）
        if isinstance(train_dataset, list):
            logger.info("将 list 格式的训练数据集转换为 HuggingFace Dataset...")
            # 先清理数据，确保所有值都是可序列化的
            train_dataset = [clean_dataset_item(item) for item in train_dataset]
            train_dataset = HFDataset.from_list(train_dataset)
            logger.info(f"✅ 转换完成，数据集大小：{len(train_dataset)}")
        
        if eval_dataset is not None and isinstance(eval_dataset, list):
            logger.info("将 list 格式的评估数据集转换为 HuggingFace Dataset...")
            eval_dataset = [clean_dataset_item(item) for item in eval_dataset]
            eval_dataset = HFDataset.from_list(eval_dataset)
            logger.info(f"✅ 转换完成，数据集大小：{len(eval_dataset)}")
        
        # 获取 SFTTrainer 的签名
        sig = inspect.signature(SFTTrainer.__init__)
        valid_params = set(sig.parameters.keys())
        
        # 构建所有可能的参数（确保所有值都是纯 Python 类型）
        all_kwargs = {
            'model': self.model,
            'train_dataset': train_dataset,
            'args': training_args,
            'max_seq_length': int(self.config.model.max_seq_length),
            'callbacks': callbacks or [],
        }
        
        # 添加 tokenizer（如果有）
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            all_kwargs['tokenizer'] = self.tokenizer
        
        # 添加评估数据集（如果有）
        if eval_dataset:
            all_kwargs['eval_dataset'] = eval_dataset
        
        # 尝试不同的 dataset_text_field 参数名
        for field_name in ['dataset_text_field', 'dataset_field', 'text_field']:
            if field_name in valid_params:
                all_kwargs[field_name] = 'text'
                break
        
        # 添加数据处理参数（避免多进程序列化问题）
        if 'dataset_num_proc' in valid_params:
            all_kwargs['dataset_num_proc'] = 1
        if 'dataloader_num_workers' in valid_params:
            all_kwargs['dataloader_num_workers'] = 0
        
        # 只传递 SFTTrainer 支持的参数
        trainer_kwargs = {k: v for k, v in all_kwargs.items() if k in valid_params or k == 'kwargs'}
        
        logger.info(f"SFTTrainer 参数：{list(trainer_kwargs.keys())}")
        
        self.trainer = SFTTrainer(**trainer_kwargs)

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
