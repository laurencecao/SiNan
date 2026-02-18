"""
数据转换器模块
将 Excel/CSV 业务数据转换为 JSONL 训练格式
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """转换结果统计"""

    total_rows: int
    valid_rows: int
    invalid_rows: int
    warnings: list[str]


class DataConverter:
    """
    数据转换器

    支持从 Excel/CSV 文件读取数据，验证并转换为 JSONL 格式
    用于 FunctionGemma 微调训练
    """

    def __init__(
        self,
        user_prompt_col: str = "User Prompt",
        tool_name_col: str = "Tool Name",
        tool_args_col: str = "Tool Args",
        validate: bool = True,
    ):
        """
        初始化数据转换器

        Args:
            user_prompt_col: 用户提示列名
            tool_name_col: 工具名列名
            tool_args_col: 工具参数列名
            validate: 是否进行数据验证
        """
        self.user_prompt_col = user_prompt_col
        self.tool_name_col = tool_name_col
        self.tool_args_col = tool_args_col
        self.validate = validate

    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        读取 Excel 或 CSV 文件

        Args:
            file_path: 文件路径

        Returns:
            DataFrame 对象
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        suffix = path.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            logger.info(f"读取 Excel 文件：{file_path}")
            return pd.read_excel(file_path)
        elif suffix == ".csv":
            logger.info(f"读取 CSV 文件：{file_path}")
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{suffix}，支持 .xlsx, .xls, .csv")

    def validate_row(self, row: pd.Series) -> tuple[bool, Optional[str]]:
        """
        验证单行数据

        Args:
            row: DataFrame 行

        Returns:
            (是否有效，错误信息)
        """
        # 检查工具名是否为空
        if (
            pd.isna(row.get(self.tool_name_col))
            or str(row.get(self.tool_name_col, "")).strip() == ""
        ):
            return False, f"工具名为空"

        # 检查用户提示是否为空
        if (
            pd.isna(row.get(self.user_prompt_col))
            or str(row.get(self.user_prompt_col, "")).strip() == ""
        ):
            return False, f"用户提示为空"

        # 检查工具参数是否为有效 JSON
        tool_args = row.get(self.tool_args_col, "{}")
        if not pd.isna(tool_args) and str(tool_args).strip():
            try:
                json.loads(str(tool_args))
            except json.JSONDecodeError as e:
                return False, f"工具参数 JSON 无效：{e}"

        return True, None

    def convert(
        self,
        input_path: str,
        output_path: str,
        add_distractors: bool = False,
        distractor_ratio: float = 0.1,
    ) -> ConversionResult:
        """
        转换数据文件为 JSONL 格式

        Args:
            input_path: 输入文件路径
            output_path: 输出 JSONL 文件路径
            add_distractors: 是否添加负采样干扰项
            distractor_ratio: 干扰项比例 (0-1)

        Returns:
            转换结果统计
        """
        logger.info(f"开始转换数据：{input_path} -> {output_path}")

        # 读取数据
        df = self.read_file(input_path)
        total_rows = len(df)

        logger.info(f"读取到 {total_rows} 行数据")

        # 验证和转换
        valid_rows = 0
        invalid_rows = 0
        warnings = []

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, row in tqdm(df.iterrows(), total=total_rows, desc="转换中"):
                # 验证
                if self.validate:
                    is_valid, error_msg = self.validate_row(row)
                    if not is_valid:
                        invalid_rows += 1
                        warning = f"行 {idx + 1} 无效：{error_msg}"
                        warnings.append(warning)
                        logger.warning(warning)
                        continue

                # 提取数据
                try:
                    user_content = str(row[self.user_prompt_col]).strip()
                    tool_name = str(row[self.tool_name_col]).strip()
                    tool_args = row.get(self.tool_args_col, "{}")

                    if pd.isna(tool_args):
                        tool_args = "{}"

                    # 确保 tool_args 是有效 JSON 字符串
                    if isinstance(tool_args, dict):
                        tool_args_str = json.dumps(tool_args, ensure_ascii=False)
                    else:
                        tool_args_str = str(tool_args).strip()
                        # 验证 JSON
                        try:
                            json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args_str = "{}"

                    # 创建 JSONL 记录
                    record = {
                        "user_content": user_content,
                        "tool_name": tool_name,
                        "tool_arguments": tool_args_str,
                    }

                    # 添加负采样干扰项 (可选)
                    if (
                        add_distractors
                        and idx > 0
                        and idx % int(1 / distractor_ratio) == 0
                    ):
                        # 随机选择一个之前的工具作为干扰
                        prev_idx = idx - 1
                        prev_row = df.iloc[prev_idx]
                        distractor_record = {
                            "user_content": user_content,
                            "tool_name": str(prev_row[self.tool_name_col]).strip(),
                            "tool_arguments": "{}",
                            "_is_distractor": True,
                        }
                        f.write(
                            json.dumps(distractor_record, ensure_ascii=False) + "\n"
                        )

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    valid_rows += 1

                except Exception as e:
                    invalid_rows += 1
                    warning = f"行 {idx + 1} 转换失败：{e}"
                    warnings.append(warning)
                    logger.warning(warning)

        result = ConversionResult(
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            warnings=warnings,
        )

        logger.info(f"转换完成：{valid_rows}/{total_rows} 行有效")

        return result

    def convert_batch(
        self, input_dir: str, output_dir: str, add_distractors: bool = False
    ) -> dict[str, ConversionResult]:
        """
        批量转换目录中的所有文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            add_distractors: 是否添加负采样

        Returns:
            每个文件的转换结果
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 查找所有支持的文件
        files = (
            list(input_path.glob("*.xlsx"))
            + list(input_path.glob("*.xls"))
            + list(input_path.glob("*.csv"))
        )

        if not files:
            logger.warning(f"在 {input_dir} 中未找到任何 Excel/CSV 文件")
            return results

        logger.info(f"找到 {len(files)} 个文件待转换")

        for file in tqdm(files, desc="批量转换"):
            output_file = output_path / f"{file.stem}.jsonl"
            result = self.convert(
                str(file), str(output_file), add_distractors=add_distractors
            )
            results[file.name] = result

        return results


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="转换 Excel/CSV 为 JSONL 格式")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("output", help="输出文件或目录路径")
    parser.add_argument("--user-col", default="User Prompt", help="用户提示列名")
    parser.add_argument("--tool-col", default="Tool Name", help="工具名列名")
    parser.add_argument("--args-col", default="Tool Args", help="工具参数列名")
    parser.add_argument("--no-validate", action="store_true", help="禁用验证")
    parser.add_argument("--distractors", action="store_true", help="添加负采样干扰项")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    converter = DataConverter(
        user_prompt_col=args.user_col,
        tool_name_col=args.tool_col,
        tool_args_col=args.args_col,
        validate=not args.no_validate,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        result = converter.convert(
            args.input, args.output, add_distractors=args.distractors
        )
        print(f"\n转换完成：{result.valid_rows}/{result.total_rows} 行有效")
        if result.warnings:
            print(f"警告：{len(result.warnings)} 条")
    else:
        results = converter.convert_batch(
            args.input, args.output, add_distractors=args.distractors
        )
        total_valid = sum(r.valid_rows for r in results.values())
        total_rows = sum(r.total_rows for r in results.values())
        print(f"\n批量转换完成：{total_valid}/{total_rows} 行有效")


if __name__ == "__main__":
    main()
