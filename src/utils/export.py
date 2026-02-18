"""
模型导出工具
支持 PyTorch、GGUF 等格式导出
"""

import logging
from pathlib import Path
from typing import Optional, Literal

logger = logging.getLogger(__name__)


def export_model(
    model,
    tokenizer,
    output_dir: str,
    export_format: Literal["pytorch", "gguf"] = "pytorch",
    merge_lora: bool = True,
    quantization: Optional[str] = None,
):
    """
    导出训练好的模型

    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        output_dir: 输出目录
        export_format: 导出格式 (pytorch/gguf)
        merge_lora: 是否合并 LoRA 适配器
        quantization: GGUF 量化类型 (q8_0, q4_k_m, q4_0 等)

    Returns:
        导出文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始导出模型：{output_path}")
    logger.info(f"导出格式：{export_format}")

    if export_format == "pytorch":
        return export_pytorch(model, tokenizer, output_path, merge_lora)

    elif export_format == "gguf":
        return export_gguf(model, tokenizer, output_path, merge_lora, quantization)

    else:
        raise ValueError(f"不支持的导出格式：{export_format}")


def export_pytorch(model, tokenizer, output_dir: Path, merge_lora: bool = True):
    """
    导出为 PyTorch 格式

    Args:
        model: 模型
        tokenizer: Tokenizer
        output_dir: 输出目录
        merge_lora: 是否合并 LoRA

    Returns:
        导出目录路径
    """
    from unsloth import FastLanguageModel

    logger.info("导出 PyTorch 格式...")

    # 保存模型
    model_path = output_dir / "pytorch"
    model_path.mkdir(parents=True, exist_ok=True)

    if merge_lora:
        logger.info("合并 LoRA 适配器...")
        model = FastLanguageModel.merge_and_unload(model)

    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    logger.info(f"PyTorch 模型已保存：{model_path}")
    return str(model_path)


def export_gguf(
    model,
    tokenizer,
    output_dir: Path,
    merge_lora: bool = True,
    quantization: str = "q8_0",
):
    """
    导出为 GGUF 格式 (用于 llama.cpp / Ollama)

    Args:
        model: 模型
        tokenizer: Tokenizer
        output_dir: 输出目录
        merge_lora: 是否合并 LoRA
        quantization: 量化类型

    Returns:
        GGUF 文件路径
    """
    from unsloth import FastLanguageModel

    logger.info(f"导出 GGUF 格式 (量化：{quantization})...")

    # 合并 LoRA (GGUF 导出需要合并)
    if merge_lora:
        logger.info("合并 LoRA 适配器...")
        model = FastLanguageModel.merge_and_unload(model)

    # 导出 GGUF
    gguf_path = output_dir / "gguf"
    gguf_path.mkdir(parents=True, exist_ok=True)

    # 使用 Unsloth 的 GGUF 导出功能
    model.save_pretrained_gguf(
        str(gguf_path), tokenizer, quantization_method=quantization
    )

    gguf_file = (
        list(gguf_path.glob("*.gguf"))[0] if list(gguf_path.glob("*.gguf")) else None
    )

    if gguf_file:
        logger.info(f"GGUF 模型已保存：{gguf_file}")
        return str(gguf_file)
    else:
        logger.error("GGUF 导出失败，未找到 .gguf 文件")
        return None


def export_to_ollama(
    gguf_path: str,
    model_name: str = "functiongemma-custom",
    template: Optional[str] = None,
) -> str:
    """
    导出到 Ollama 格式

    Args:
        gguf_path: GGUF 文件路径
        model_name: Ollama 模型名
        template: Modelfile 模板

    Returns:
        Ollama 模型名
    """
    import subprocess

    logger.info(f"导出到 Ollama: {model_name}")

    gguf_file = Path(gguf_path)

    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF 文件不存在：{gguf_path}")

    # 创建 Modelfile
    modelfile_content = (
        template
        or f"""
FROM {gguf_file.absolute()}

# FunctionGemma 模板
TEMPLATE \"\"\"
{{{{ if .Tools }}}}
<<start_of_turn>>developer
You are a model that can do function calling with the following functions
{{{{ range .Tools }}}}
<<start_function_declaration>>declaration:{{{{ .function.name }}}}{{{{ description:<escape>{{{{ .function.description }}}}<escape>,parameters:{{{{ .function.parameters }}}}}}}}>><end_function_declaration>>
{{{{ end }}}}
<<end_of_turn>>
{{{{ end }}}}
<<start_of_turn>>user
{{{{ .Prompt }}}}<<end_of_turn>>
<<start_of_turn>>model
\"\"\"

PARAMETER temperature 1.0
PARAMETER top_k 64
PARAMETER top_p 0.95
"""
    )

    modelfile_path = gguf_file.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content)

    # 使用 ollama create 导入
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Ollama 模型已创建：{model_name}")
        return model_name

    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama 导入失败：{e.stderr}")
        raise


def get_export_size(quantization: str) -> str:
    """
    获取不同量化格式的预估大小

    Args:
        quantization: 量化类型

    Returns:
        预估大小字符串
    """
    sizes = {
        "f16": "~540 MB",
        "q8_0": "~280 MB",
        "q4_k_m": "~160 MB",
        "q4_0": "~150 MB",
        "q3_k_m": "~130 MB",
        "q2_k": "~110 MB",
    }

    return sizes.get(quantization, "未知")
