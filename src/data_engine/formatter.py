"""
FunctionGemma 格式化器
将 JSONL 数据转换为 FunctionGemma 特殊 Token 格式
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FunctionSchema:
    """函数定义"""

    name: str
    description: str
    parameters: dict


class FunctionGemmaFormatter:
    """
    FunctionGemma 格式化器

    将标准 JSONL 训练数据转换为 FunctionGemma 的特殊 Token 格式
    参考：https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
    """

    # FunctionGemma 特殊 Token
    TOKEN_START_TURN = "<<start_of_turn>>"
    TOKEN_END_TURN = "<<end_of_turn>>"
    TOKEN_START_FUNCTION_DECLARATION = "<<start_function_declaration>>"
    TOKEN_END_FUNCTION_DECLARATION = "<<end_function_declaration>>"
    TOKEN_START_FUNCTION_CALL = "<<start_function_call>>"
    TOKEN_END_FUNCTION_CALL = "<<end_function_call>>"
    TOKEN_START_FUNCTION_RESPONSE = "<<start_function_response>>"
    TOKEN_END_FUNCTION_RESPONSE = "<<end_function_response>>"
    TOKEN_ESCAPE = "<escape>"

    def __init__(self, max_seq_length: int = 2048, add_generation_prompt: bool = True):
        """
        初始化格式化器

        Args:
            max_seq_length: 最大序列长度
            add_generation_prompt: 是否添加生成提示
        """
        self.max_seq_length = max_seq_length
        self.add_generation_prompt = add_generation_prompt

    @staticmethod
    def escape_string(value: str) -> str:
        """
        使用 <escape> 包裹字符串值

        Args:
            value: 原始字符串

        Returns:
            包裹后的字符串
        """
        return f"{FunctionGemmaFormatter.TOKEN_ESCAPE}{value}{FunctionGemmaFormatter.TOKEN_ESCAPE}"

    def format_function_declaration(
        self, name: str, description: str, parameters: Optional[dict] = None
    ) -> str:
        """
        格式化函数声明

        Args:
            name: 函数名
            description: 函数描述
            parameters: 参数定义 (JSON Schema 格式)

        Returns:
            格式化后的函数声明字符串
        """
        params_str = ""

        if parameters:
            props = parameters.get("properties", {})
            required = parameters.get("required", [])

            if props:
                params_parts = []
                for param_name, param_info in props.items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "string")
                    is_required = param_name in required

                    param_str = (
                        f"{param_name}:{{"
                        f"description:{self.escape_string(param_desc)},"
                        f"type:{param_type},"
                        f"required:{str(is_required).lower()}"
                        f"}}"
                    )
                    params_parts.append(param_str)

                params_str = (
                    f",parameters:{{properties:{{ {', '.join(params_parts)} }}}}"
                )

        declaration = (
            f"declaration:{name}{{"
            f"description:{self.escape_string(description)}"
            f"{params_str}"
            f"}}"
        )

        return (
            f"{self.TOKEN_START_FUNCTION_DECLARATION}"
            f"{declaration}"
            f"{self.TOKEN_END_FUNCTION_DECLARATION}"
        )

    def format_function_call(self, name: str, arguments: dict) -> str:
        """
        格式化函数调用

        Args:
            name: 函数名
            arguments: 函数参数

        Returns:
            格式化后的函数调用字符串
        """
        args_parts = []

        for key, value in arguments.items():
            if isinstance(value, str):
                args_parts.append(f"{key}:{self.escape_string(value)}")
            elif isinstance(value, bool):
                args_parts.append(f"{key}:{str(value).lower()}")
            elif value is None:
                args_parts.append(f"{key}:null")
            else:
                args_parts.append(f"{key}:{value}")

        args_str = ", ".join(args_parts)

        return (
            f"{self.TOKEN_START_FUNCTION_CALL}"
            f"call:{name}{{{args_str}}}"
            f"{self.TOKEN_END_FUNCTION_CALL}"
        )

    def format_function_response(self, name: str, result: dict) -> str:
        """
        格式化函数响应

        Args:
            name: 函数名
            result: 函数返回结果

        Returns:
            格式化后的函数响应字符串
        """
        result_parts = []

        for key, value in result.items():
            if isinstance(value, str):
                result_parts.append(f"{key}:{self.escape_string(value)}")
            elif isinstance(value, bool):
                result_parts.append(f"{key}:{str(value).lower()}")
            elif value is None:
                result_parts.append(f"{key}:null")
            else:
                result_parts.append(f"{key}:{value}")

        result_str = ", ".join(result_parts)

        return (
            f"{self.TOKEN_START_FUNCTION_RESPONSE}"
            f"response:{name}{{{result_str}}}"
            f"{self.TOKEN_END_FUNCTION_RESPONSE}"
        )

    def format_training_sample(
        self,
        user_content: str,
        tool_name: str,
        tool_arguments: dict,
        function_declarations: Optional[list[FunctionSchema]] = None,
    ) -> str:
        """
        格式化完整的训练样本

        Args:
            user_content: 用户输入
            tool_name: 工具名
            tool_arguments: 工具参数
            function_declarations: 函数声明列表

        Returns:
            完整的训练样本字符串
        """
        parts = []

        # Step 1: Developer Turn (函数声明)
        developer_content = (
            "You are a model that can do function calling with the following functions"
        )

        if function_declarations:
            for func in function_declarations:
                developer_content += "\n" + self.format_function_declaration(
                    func.name, func.description, func.parameters
                )

        parts.append(
            f"{self.TOKEN_START_TURN}developer\n"
            f"{developer_content}\n"
            f"{self.TOKEN_END_TURN}"
        )

        # Step 2: User Turn
        parts.append(
            f"{self.TOKEN_START_TURN}user\n{user_content}\n{self.TOKEN_END_TURN}"
        )

        # Step 3: Model Turn (函数调用)
        function_call = self.format_function_call(tool_name, tool_arguments)
        parts.append(
            f"{self.TOKEN_START_TURN}model\n{function_call}\n{self.TOKEN_END_TURN}"
        )

        return "\n".join(parts)

    def format_chat_template(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> str:
        """
        格式化聊天模板 (兼容 HuggingFace transformers)

        Args:
            messages: 消息列表，每个消息包含 role 和 content
            tools: 工具定义列表

        Returns:
            格式化后的完整提示
        """
        parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "developer" or role == "system":
                developer_content = content

                if tools:
                    for tool in tools:
                        func_info = tool.get("function", {})
                        developer_content += "\n" + self.format_function_declaration(
                            func_info.get("name", ""),
                            func_info.get("description", ""),
                            func_info.get("parameters", {}),
                        )

                parts.append(
                    f"{self.TOKEN_START_TURN}developer\n"
                    f"{developer_content}\n"
                    f"{self.TOKEN_END_TURN}"
                )

            elif role == "user":
                # 检查是否包含函数响应
                if "tool_responses" in message:
                    for response in message["tool_responses"]:
                        response_str = self.format_function_response(
                            response["name"], response["result"]
                        )
                        parts.append(
                            f"{self.TOKEN_START_TURN}user\n"
                            f"{response_str}\n"
                            f"{self.TOKEN_END_TURN}"
                        )
                else:
                    parts.append(
                        f"{self.TOKEN_START_TURN}user\n{content}\n{self.TOKEN_END_TURN}"
                    )

            elif role == "assistant" or role == "model":
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        func_info = tool_call.get("function", {})
                        call_str = self.format_function_call(
                            func_info.get("name", ""),
                            json.loads(func_info.get("arguments", "{}")),
                        )
                        parts.append(
                            f"{self.TOKEN_START_TURN}model\n"
                            f"{call_str}\n"
                            f"{self.TOKEN_END_TURN}"
                        )
                elif content:
                    parts.append(
                        f"{self.TOKEN_START_TURN}model\n"
                        f"{content}\n"
                        f"{self.TOKEN_END_TURN}"
                    )

        return "\n".join(parts)

    def convert_jsonl(
        self,
        input_path: str,
        output_path: str,
        function_declarations: Optional[list[FunctionSchema]] = None,
    ) -> dict:
        """
        转换 JSONL 文件为 FunctionGemma 格式

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出文件路径
            function_declarations: 函数声明列表

        Returns:
            转换统计信息
        """
        logger.info(f"开始格式化：{input_path} -> {output_path}")

        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        stats = {"total": 0, "success": 0, "failed": 0}

        with (
            open(input_file, "r", encoding="utf-8") as f_in,
            open(output_file, "w", encoding="utf-8") as f_out,
        ):
            for line in tqdm(f_in, desc="格式化中"):
                stats["total"] += 1

                try:
                    record = json.loads(line.strip())

                    user_content = record.get("user_content", "")
                    tool_name = record.get("tool_name", "")
                    tool_arguments = record.get("tool_arguments", {})

                    if isinstance(tool_arguments, str):
                        tool_arguments = json.loads(tool_arguments)

                    formatted = self.format_training_sample(
                        user_content=user_content,
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                        function_declarations=function_declarations,
                    )

                    # 写入训练格式 (text 字段)
                    train_record = {"text": formatted}
                    f_out.write(json.dumps(train_record, ensure_ascii=False) + "\n")

                    stats["success"] += 1

                except Exception as e:
                    logger.warning(f"行 {stats['total']} 格式化失败：{e}")
                    stats["failed"] += 1

        logger.info(f"格式化完成：{stats['success']}/{stats['total']} 成功")
        return stats


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="格式化数据为 FunctionGemma 格式")
    parser.add_argument("input", help="输入 JSONL 文件路径")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--max-length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    formatter = FunctionGemmaFormatter(max_seq_length=args.max_length)
    stats = formatter.convert_jsonl(args.input, args.output)

    print(f"\n格式化完成：{stats['success']}/{stats['total']} 成功")


if __name__ == "__main__":
    main()
