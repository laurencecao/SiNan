# 数据格式

本文档详细介绍 SiNan 支持的数据格式和转换流程。

## 数据格式概览

```
原始数据 → 中间格式 → 训练格式
(Excel/CSV) → (JSONL) → (FunctionGemma Token)
```

## 1. 原始数据格式 (Excel/CSV)

### 标准列名

| 列名 | 必填 | 说明 | 示例 |
|------|------|------|------|
| User Prompt | ✅ | 用户输入文本 | "查询北京天气" |
| Tool Name | ✅ | 函数名称 | "get_weather" |
| Tool Args | ✅ | 函数参数 (JSON) | `{"location": "Beijing"}` |

### Excel 示例

创建 `data/raw/hr_functions.xlsx`:

| User Prompt | Tool Name | Tool Args |
|-------------|-----------|-----------|
| 查询北京天气 | get_weather | `{"location": "Beijing"}` |
| 查询上海天气 | get_weather | `{"location": "Shanghai"}` |
| 把背景改成红色 | change_color | `{"color": "red"}` |
| 创建用户张三 | create_user | `{"name": "张三", "age": 25}` |

### CSV 示例

```csv
User Prompt,Tool Name,Tool Args
查询北京天气，get_weather,"{""location"": ""Beijing""}"
查询上海天气，get_weather,"{""location"": ""Shanghai""}"
把背景改成红色，change_color,"{""color"": ""red""}"
```

### 自定义列名

在 `configs/base_config.yaml` 中配置:

```yaml
data:
  columns:
    user_prompt: "员工问题"    # 你的 Excel 列名
    tool_name: "HR 功能"       # 你的 Excel 列名
    tool_args: "参数"          # 你的 Excel 列名
```

## 2. JSONL 格式 (中间格式)

### 标准格式

每行一个 JSON 对象:

```json
{"user_content": "查询北京天气", "tool_name": "get_weather", "tool_arguments": {"location": "Beijing"}}
{"user_content": "查询上海天气", "tool_name": "get_weather", "tool_arguments": {"location": "Shanghai"}}
{"user_content": "把背景改成红色", "tool_name": "change_color", "tool_arguments": {"color": "red"}}
```

### 使用转换器生成

```bash
python main.py convert data/raw/hr_functions.xlsx data/processed/hr_functions.jsonl
```

### 数据验证规则

转换器会自动验证:

1. **Tool Name 非空** - 函数名不能为空
2. **User Prompt 非空** - 用户输入不能为空
3. **Tool Args 有效 JSON** - 参数必须是合法 JSON

验证失败的行会被记录到日志:

```
行 5 无效：工具名为空
行 12 无效：工具参数 JSON 无效：Expecting property name...
```

## 3. FunctionGemma Token 格式 (训练格式)

### Token 类型

FunctionGemma 使用特殊的控制 Token:

| Token | 用途 | 示例 |
|-------|------|------|
| `<<start_of_turn>>role` | 对话轮次开始 | `<<start_of_turn>>user` |
| `<<end_of_turn>>` | 对话轮次结束 | `<<end_of_turn>>` |
| `<<start_function_declaration>>` | 函数声明开始 | |
| `<<end_function_declaration>>` | 函数声明结束 | |
| `<<start_function_call>>` | 函数调用开始 | |
| `<<end_function_call>>` | 函数调用结束 | |
| `<<start_function_response>>` | 函数响应开始 | |
| `<<end_function_response>>` | 函数响应结束 | |
| `<escape>` | 字符串值包裹 | `<escape>value<escape>` |

### 完整训练样本格式

```
<<start_of_turn>>developer
You are a model that can do function calling with the following functions
<<start_function_declaration>>declaration:get_weather{description:<escape>>获取天气<escape>,parameters:{properties:{location:{description:<escape>>城市名<escape>,type:string,required:true}}}}<<end_function_declaration>>
<<end_of_turn>>
<<start_of_turn>>user
查询北京天气<<end_of_turn>>
<<start_of_turn>>model
<<start_function_call>>call:get_weather{location:<escape>>Beijing<escape>>}<<end_function_call>>
<<end_of_turn>>
```

### 格式化流程

```bash
python main.py format data/processed/hr.jsonl data/formatted/hr_formatted.jsonl
```

## 4. 训练数据集格式

### HuggingFace Dataset

最终训练数据需要转换为 HuggingFace Dataset 格式:

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='data/formatted/hr_formatted.jsonl', split='train')
```

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 完整的 FunctionGemma 格式化文本 |

## 5. 数据量建议

### 最小可行数据集

| 函数数 | 每函数样本 | 总样本 | 预期效果 |
|--------|-----------|--------|---------|
| 3-5 | 50-100 | 150-500 | 基础可用 (~70%) |

### 生产推荐

| 函数数 | 每函数样本 | 总样本 | 预期效果 |
|--------|-----------|--------|---------|
| 10-20 | 200-500 | 2000-10000 | 生产质量 (~85%+) |

### 高质量数据集

| 函数数 | 每函数样本 | 总样本 | 预期效果 |
|--------|-----------|--------|---------|
| 20+ | 500+ | 10000+ | 优秀 (~90%+) |

## 6. 数据质量检查清单

### ✅ 必查项

- [ ] 所有函数名拼写一致
- [ ] 参数 JSON 格式正确
- [ ] 用户提示多样化 (不只一种问法)
- [ ] 覆盖常见边界情况

### ⚠️ 注意事项

- **避免重复**: 相似样本不超过 10%
- **平衡分布**: 每个函数样本数尽量均衡
- **覆盖场景**: 包含简单和复杂查询
- **参数多样性**: 同一函数的参数值要多样化

### ❌ 常见错误

```
# 错误 1: 参数不是有效 JSON
{"user_content": "查询天气", "tool_name": "get_weather", "tool_arguments": "location=Beijing"}
# 应该是：{"location": "Beijing"}

# 错误 2: 函数名不一致
{"tool_name": "get_weather"}
{"tool_name": "getWeather"}  # 应该统一

# 错误 3: 缺少必填参数
{"tool_name": "create_user", "tool_arguments": {}}  # 缺少 name 等必填字段
```

## 7. 数据增强技巧

### 同义改写

对同一个函数，创建多种问法:

```
get_weather:
- "查询北京天气"
- "北京今天天气怎么样"
- "告诉我北京的天气"
- "北京天气如何"
```

### 参数变化

对同一个函数，使用不同参数:

```
create_user:
- {"name": "张三", "age": 25}
- {"name": "李四", "age": 30}
- {"name": "王五", "age": 35}
```

### 负采样增强 (可选)

添加干扰项提高鲁棒性:

```bash
python main.py convert data/raw/ data/processed/ --distractors
```

## 8. 示例数据集

### HR 系统示例

```json
{"user_content": "查询张三的年假", "tool_name": "get_leave_balance", "tool_arguments": {"employee_id": "001", "leave_type": "annual"}}
{"user_content": "李四请病假", "tool_name": "request_leave", "tool_arguments": {"employee_id": "002", "leave_type": "sick", "days": 3}}
{"user_content": "创建新员工王五", "tool_name": "create_employee", "tool_arguments": {"name": "王五", "department": "技术部"}}
```

### CRM 系统示例

```json
{"user_content": "查询客户张三的信息", "tool_name": "get_customer", "tool_arguments": {"customer_id": "C001"}}
{"user_content": "更新李四的联系方式", "tool_name": "update_customer", "tool_arguments": {"customer_id": "C002", "phone": "13800138000"}}
{"user_content": "创建新的销售机会", "tool_name": "create_opportunity", "tool_arguments": {"customer_id": "C003", "amount": 100000}}
```

## 9. 数据转换工具

### 批量转换

```bash
python main.py convert data/raw/ data/processed/
```

### 查看转换统计

```
找到 3 个文件待转换
转换完成：284/300 行有效
警告：16 条
```

### 调试模式

```bash
python main.py convert data/raw/demo.xlsx data/processed/demo.jsonl -v
```

详细输出每个样本的转换过程。
