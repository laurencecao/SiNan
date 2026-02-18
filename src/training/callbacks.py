"""
训练回调函数
WandB 集成和自定义检查点
"""

import logging
from typing import Optional

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import has_length

logger = logging.getLogger(__name__)


class WandbCallback(TrainerCallback):
    """
    Weights & Biases 回调

    记录训练指标到 WandB
    """

    def __init__(
        self, project: str, entity: Optional[str] = None, name: Optional[str] = None
    ):
        """
        初始化 WandB 回调

        Args:
            project: WandB 项目名
            entity: WandB 用户名/团队名
            name: 运行名称
        """
        self.project = project
        self.entity = entity
        self.name = name
        self._initialized = False

    def setup(self, args: TrainingArguments, state: TrainerState):
        """
        初始化 WandB

        Args:
            args: 训练参数
            state: 训练状态
        """
        try:
            import wandb

            if not self._initialized:
                wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=self.name or state.log_history[0].get("train_runtime")
                    if state.log_history
                    else None,
                    config=args.to_dict(),
                )
                self._initialized = True
                logger.info("WandB 初始化成功")

        except ImportError:
            logger.warning("wandb 未安装，禁用 WandB 回调")
        except Exception as e:
            logger.warning(f"WandB 初始化失败：{e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        记录日志

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志字典
        """
        if not self._initialized:
            self.setup(args, state)

        if logs is not None:
            try:
                import wandb

                wandb.log(logs, step=state.global_step)
            except Exception as e:
                logger.warning(f"WandB 记录失败：{e}")

    def on_train_end(self, args, state, control, **kwargs):
        """
        训练结束处理

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
        """
        if self._initialized:
            try:
                import wandb

                wandb.finish()
                logger.info("WandB 运行已关闭")
            except Exception as e:
                logger.warning(f"WandB 关闭失败：{e}")


class SampleGenerationCallback(TrainerCallback):
    """
    样本生成回调

    在每个 epoch 结束时生成样本，评估模型质量
    """

    def __init__(
        self,
        tokenizer,
        test_prompts: list[str],
        generation_kwargs: Optional[dict] = None,
        log_to_wandb: bool = True,
    ):
        """
        初始化回调

        Args:
            tokenizer: Tokenizer
            test_prompts: 测试提示列表
            generation_kwargs: 生成参数
            log_to_wandb: 是否记录到 WandB
        """
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 128,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
        }
        self.log_to_wandb = log_to_wandb

    def on_epoch_end(self, args, state, control, model, **kwargs):
        """
        Epoch 结束时执行

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            model: 模型
        """
        from unsloth import FastLanguageModel

        logger.info(f"Epoch {state.epoch} 结束，生成测试样本...")

        # 切换到推理模式
        FastLanguageModel.for_inference(model)

        samples = []

        for prompt in self.test_prompts[:3]:  # 限制为 3 个样本
            try:
                messages = [
                    {
                        "role": "developer",
                        "content": "You are a model that can do function calling with the following functions",
                    },
                    {"role": "user", "content": prompt},
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)

                outputs = model.generate(inputs, **self.generation_kwargs)

                result = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

                samples.append(
                    {
                        "prompt": prompt,
                        "generation": result,
                    }
                )

                logger.info(f"测试样本:\n{result[:500]}...")

            except Exception as e:
                logger.warning(f"生成失败：{e}")

        # 切换回训练模式
        FastLanguageModel.for_training(model)

        # 记录到 WandB
        if self.log_to_wandb and samples:
            try:
                import wandb

                wandb.log(
                    {
                        "sample_generations": wandb.Html(
                            "\n".join(
                                [
                                    f"<h3>Prompt {i + 1}</h3><p>{s['prompt']}</p>"
                                    f"<h4>Generation</h4><pre>{s['generation']}</pre>"
                                    for i, s in enumerate(samples)
                                ]
                            )
                        ),
                        "epoch": state.epoch,
                    }
                )
            except Exception as e:
                logger.warning(f"WandB 记录失败：{e}")

        return control


class EarlyStoppingCallback(TrainerCallback):
    """
    早停回调

    当验证损失不再下降时提前停止训练
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        """
        初始化早停回调

        Args:
            patience: 容忍轮数
            min_delta: 最小改进阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        评估时检查

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            metrics: 评估指标
        """
        eval_loss = metrics.get("eval_loss")

        if eval_loss is None:
            return control

        if self.best_loss is None:
            self.best_loss = eval_loss
            self.counter = 0
        elif eval_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"早停计数：{self.counter}/{self.patience}")

            if self.counter >= self.patience:
                logger.info(f"触发早停，最佳损失：{self.best_loss}")
                control.should_training_stop = True
        else:
            self.best_loss = eval_loss
            self.counter = 0

        return control
