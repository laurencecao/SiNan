"""
SiNan - FunctionGemma 企业微调框架
主入口 CLI
"""

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.utils.config_loader import load_config, print_config
from src.data_engine.converter import DataConverter
from src.data_engine.formatter import FunctionGemmaFormatter
from src.training.trainer import FunctionGemmaTrainer
from src.training.callbacks import (
    WandbCallback,
    SampleGenerationCallback,
    EarlyStoppingCallback,
)
from src.utils.export import export_model

app = typer.Typer(help="FunctionGemma 企业微调框架")
console = Console()


def setup_logging(verbose: bool = False):
    """配置日志"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def convert(
    input_path: str = typer.Argument(..., help="输入文件或目录路径"),
    output_path: str = typer.Argument(..., help="输出文件路径"),
    user_col: str = typer.Option("User Prompt", "--user-col", help="用户提示列名"),
    tool_col: str = typer.Option("Tool Name", "--tool-col", help="工具名列名"),
    args_col: str = typer.Option("Tool Args", "--args-col", help="工具参数列名"),
    no_validate: bool = typer.Option(False, "--no-validate", help="禁用验证"),
    distractors: bool = typer.Option(False, "--distractors", help="添加负采样"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="详细输出"),
):
    """转换 Excel/CSV 为 JSONL 格式"""
    setup_logging(verbose)

    console.print("[bold blue]数据转换[/bold blue]")

    converter = DataConverter(
        user_prompt_col=user_col,
        tool_name_col=tool_col,
        tool_args_col=args_col,
        validate=not no_validate,
    )

    input_file = Path(input_path)

    if input_file.is_file():
        result = converter.convert(input_path, output_path, add_distractors=distractors)
        console.print(
            f"[green]✓ 转换完成：{result.valid_rows}/{result.total_rows} 行有效[/green]"
        )
    else:
        results = converter.convert_batch(
            input_path, output_path, add_distractors=distractors
        )
        total = sum(r.total_rows for r in results.values())
        valid = sum(r.valid_rows for r in results.values())
        console.print(f"[green]✓ 批量转换完成：{valid}/{total} 行有效[/green]")


@app.command()
def format(
    input_path: str = typer.Argument(..., help="输入 JSONL 文件路径"),
    output_path: str = typer.Argument(..., help="输出文件路径"),
    max_length: int = typer.Option(2048, "--max-length", help="最大序列长度"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="详细输出"),
):
    """格式化 JSONL 为 FunctionGemma 格式"""
    setup_logging(verbose)

    console.print("[bold blue]数据格式化[/bold blue]")

    formatter = FunctionGemmaFormatter(max_seq_length=max_length)
    stats = formatter.convert_jsonl(input_path, output_path)

    console.print(
        f"[green]✓ 格式化完成：{stats['success']}/{stats['total']} 成功[/green]"
    )


@app.command()
def train(
    config_name: str = typer.Option("base_config", "--config", help="配置文件名"),
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验配置名"),
    data_path: str = typer.Option(..., "--data", help="训练数据路径"),
    output_dir: str = typer.Option(None, "--output", "-o", help="输出目录"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="详细输出"),
):
    """训练 FunctionGemma 模型"""
    setup_logging(verbose)

    console.print("[bold blue]开始训练[/bold blue]")

    # 加载配置
    config = load_config(
        config_name=config_name,
        experiment=experiment,
        overrides=[f"data.processed_dir={data_path}"] if data_path else None,
    )

    if output_dir:
        config.logging.output_dir = output_dir

    print_config(config)

    # 加载数据
    converter = DataConverter()
    train_dataset = converter.load_dataset(data_path)

    # 创建训练器
    trainer = FunctionGemmaTrainer(config)

    # 配置回调
    callbacks = []

    if config.logging.wandb.enabled:
        callbacks.append(
            WandbCallback(
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
            )
        )

    # 添加样本生成回调
    test_prompts = [
        "查询北京天气",
        "把背景改成红色",
        "创建一个新的用户",
    ]
    callbacks.append(
        SampleGenerationCallback(
            tokenizer=trainer.tokenizer if trainer.tokenizer else None,
            test_prompts=test_prompts,
        )
    )

    # 早停
    if config.training.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(patience=config.training.early_stopping_patience)
        )

    # 开始训练
    trainer.load_model()
    trainer.train(
        train_dataset=train_dataset,
        output_dir=config.logging.output_dir,
        callbacks=callbacks,
    )

    # 保存模型
    trainer.save_model(config.logging.output_dir)

    console.print(
        f"[green]✓ 训练完成，模型已保存到：{config.logging.output_dir}[/green]"
    )


@app.command()
def export(
    model_path: str = typer.Argument(..., help="模型路径"),
    output_dir: str = typer.Argument(..., help="输出目录"),
    format: str = typer.Option(
        "gguf", "--format", "-f", help="导出格式 (pytorch/gguf)"
    ),
    quantization: str = typer.Option(
        "q8_0", "--quantization", "-q", help="GGUF 量化类型"
    ),
    merge_lora: bool = typer.Option(
        True, "--merge-lora/--no-merge-lora", help="是否合并 LoRA"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="详细输出"),
):
    """导出训练好的模型"""
    setup_logging(verbose)

    console.print("[bold blue]导出模型[/bold blue]")

    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    # 加载模型
    console.print(f"加载模型：{model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(model_path)

    # 导出
    output_path = export_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        export_format=format,
        merge_lora=merge_lora,
        quantization=quantization,
    )

    console.print(f"[green]✓ 模型已导出到：{output_path}[/green]")


@app.command()
def inference(
    model_path: str = typer.Argument(..., help="模型路径"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="输入提示"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="最大生成 token 数"),
):
    """推理测试"""
    from unsloth import FastLanguageModel

    console.print("[bold blue]推理测试[/bold blue]")

    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(model_path)

    # 推理
    messages = [
        {
            "role": "developer",
            "content": "You are a model that can do function calling with the following functions",
        },
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)

    console.print("\n[bold]输入:[/bold]")
    console.print(prompt)
    console.print("\n[bold]输出:[/bold]")
    console.print(result)


@app.command()
def config(
    config_name: str = typer.Option("base_config", "--config", help="配置文件名"),
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验配置名"),
):
    """查看配置"""
    config = load_config(config_name=config_name, experiment=experiment)
    print_config(config)


def main():
    """CLI 入口"""
    app()


if __name__ == "__main__":
    main()
