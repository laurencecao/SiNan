"""
配置加载器
使用 OmegaConf 加载和管理 YAML 配置
"""

import logging
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def load_config(
    config_name: str = "base_config",
    config_dir: str = "configs",
    experiment: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """
    加载配置文件

    Args:
        config_name: 基础配置文件名 (不含 .yaml)
        config_dir: 配置文件目录
        experiment: 实验配置名 (可选，会合并到基础配置)
        overrides: OmegaConf 覆盖参数 (命令行格式)

    Returns:
        合并后的配置对象
    """
    config_path = Path(config_dir)

    # 加载基础配置
    base_config_file = config_path / f"{config_name}.yaml"

    if not base_config_file.exists():
        raise FileNotFoundError(f"配置文件不存在：{base_config_file}")

    logger.info(f"加载基础配置：{base_config_file}")
    config = OmegaConf.load(base_config_file)

    # 加载实验配置 (如果指定)
    if experiment:
        experiment_file = config_path / "experiments" / f"{experiment}.yaml"

        if experiment_file.exists():
            logger.info(f"加载实验配置：{experiment_file}")
            experiment_config = OmegaConf.load(experiment_file)
            config = OmegaConf.merge(config, experiment_config)
        else:
            logger.warning(f"实验配置文件不存在：{experiment_file}")

    # 应用覆盖参数
    if overrides:
        logger.info(f"应用覆盖参数：{overrides}")
        override_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_config)

    # 解析路径 (相对路径转绝对路径)
    config = _resolve_paths(config, config_path.parent)

    return config


def _resolve_paths(config: Any, base_dir: Path) -> Any:
    """
    递归解析配置中的相对路径

    Args:
        config: 配置对象
        base_dir: 基础目录

    Returns:
        解析后的配置
    """
    if isinstance(config, DictConfig):
        resolved = OmegaConf.create()

        for key, value in config.items():
            resolved[key] = _resolve_paths(value, base_dir)

        return resolved

    elif isinstance(config, list):
        return [_resolve_paths(item, base_dir) for item in config]

    elif isinstance(config, str):
        # 检测是否为路径字段
        if key_is_path_field(config):
            path = Path(config)
            if not path.is_absolute():
                return str(base_dir / path)

        return config

    else:
        return config


def key_is_path_field(value: str) -> bool:
    """
    判断值是否可能是路径

    Args:
        value: 配置值

    Returns:
        是否可能是路径
    """
    # 简单启发式判断
    path_indicators = ["data/", "output/", "logs/", "models/", "config/"]
    return any(indicator in value for indicator in path_indicators)


def save_config(config: DictConfig, output_path: str, resolve: bool = True) -> None:
    """
    保存配置到文件

    Args:
        config: 配置对象
        output_path: 输出文件路径
        resolve: 是否解析插值
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if resolve:
        config = OmegaConf.to_container(config, resolve=True)

    OmegaConf.save(config, output_file)
    logger.info(f"配置已保存：{output_file}")


def print_config(config: DictConfig) -> None:
    """
    打印格式化后的配置

    Args:
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60 + "\n")
