# HR 系统路由示例

本示例演示如何为 HR 系统训练一个智能路由模型。

## 场景描述

HR 系统需要处理员工的各类请求:
- 查询年假余额
- 请假申请
- 查询工资
- 创建新员工
- 更新员工信息

## 1. 准备数据

### 创建 Excel 文件

`data/raw/hr_functions.xlsx`:

| User Prompt | Tool Name | Tool Args |
|-------------|-----------|-----------|
| 查询张三的年假 | get_leave_balance | `{"employee_id": "001", "leave_type": "annual"}` |
| 李四还有多少病假 | get_leave_balance | `{"employee_id": "002", "leave_type": "sick"}` |
| 我要请 3 天年假 | request_leave | `{"employee_id": "001", "leave_type": "annual", "days": 3}` |
| 王五请病假一周 | request_leave | `{"employee_id": "003", "leave_type": "sick", "days": 7}` |
| 查询张三的工资 | get_salary | `{"employee_id": "001", "month": "2024-01"}` |
| 创建新员工赵六 | create_employee | `{"name": "赵六", "department": "技术部", "position": "工程师"}` |
| 更新李四的部门 | update_employee | `{"employee_id": "002", "department": "市场部"}` |

### 数据量建议

| 函数 | 最小样本 | 推荐样本 |
|------|----------|----------|
| get_leave_balance | 50 | 200 |
| request_leave | 50 | 200 |
| get_salary | 50 | 150 |
| create_employee | 50 | 150 |
| update_employee | 50 | 150 |
| **总计** | **250** | **850** |

## 2. 转换数据

```bash
python main.py convert \
  data/raw/hr_functions.xlsx \
  data/processed/hr_functions.jsonl
```

输出:
```
✓ 转换完成：850/850 行有效
```

## 3. 配置训练

### 创建实验配置

`configs/experiments/exp_hr_routing.yaml`:

```yaml
training:
  epochs: 5
  learning_rate: 1.0e-4
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2

model:
  lora:
    rank: 32
    alpha: 32

logging:
  wandb:
    project: "functiongemma-hr-routing"
    name: "hr-routing-v1"
```

## 4. 开始训练

```bash
python main.py train \
  --data data/processed/hr_functions.jsonl \
  --experiment exp_hr_routing \
  --output outputs/models/hr_v1
```

训练输出:
```
加载模型：google/functiongemma-270m-it
配置 LoRA...
数据集大小：850
开始训练...
[1/5] Loss: 2.345
[2/5] Loss: 1.234
[3/5] Loss: 0.678
[4/5] Loss: 0.456
[5/5] Loss: 0.345
✓ 训练完成
```

## 5. 测试推理

```bash
python main.py inference \
  outputs/models/hr_v1 \
  --prompt "查询张三的年假余额"
```

输出:
```
<<start_of_turn>>model
<<start_function_call>>call:get_leave_balance{employee_id:<<escape>>001<<escape>>,leave_type:<<escape>>annual<<escape>>}<<end_function_call>>
<<end_of_turn>>
```

## 6. 导出部署

```bash
# 导出 GGUF
python main.py export \
  outputs/models/hr_v1 \
  outputs/models/hr_gguf \
  --format gguf \
  --quantization q8_0

# 导入 Ollama
ollama create functiongemma-hr -f outputs/models/hr_gguf/Modelfile

# 运行
ollama run functiongemma-hr "李四请病假 3 天"
```

## 7. 集成到 HR 系统

### Python 集成

```python
import requests

def call_hr_function(user_input: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "functiongemma-hr",
            "prompt": user_input,
            "stream": False
        }
    )
    
    result = response.json()["response"]
    
    # 解析函数调用
    if "call:get_leave_balance" in result:
        # 提取参数并调用实际函数
        employee_id = extract_param(result, "employee_id")
        leave_type = extract_param(result, "leave_type")
        return actual_get_leave_balance(employee_id, leave_type)
    
    # ... 处理其他函数
```

### API 集成

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/hr/function")
def hr_function_call(user_input: str):
    # 调用模型
    result = model.inference(user_input)
    
    # 解析并执行函数
    function_name, params = parse_function_call(result)
    
    # 调用实际 HR 系统 API
    hr_result = call_hr_api(function_name, params)
    
    return {"function": function_name, "result": hr_result}
```

## 8. 性能优化

### 缓存常用查询

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_inference(user_input: str):
    return model.inference(user_input)
```

### 批处理

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(model.inference, user_inputs))
```

## 9. 监控和日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_function_call(user_input: str, function_name: str, params: dict):
    logger.info(f"HR Function Call: {user_input} -> {function_name}({params})")
```

## 10. 效果评估

### 测试集

创建 `data/test/hr_test.jsonl`:

```json
{"user_content": "张三还有几天年假", "expected_function": "get_leave_balance"}
{"user_content": "我要请假", "expected_function": "request_leave"}
{"user_content": "查工资", "expected_function": "get_salary"}
```

### 评估脚本

```python
def evaluate(test_file: str) -> float:
    correct = 0
    total = 0
    
    with open(test_file) as f:
        for line in f:
            test_case = json.loads(line)
            result = model.inference(test_case["user_content"])
            
            if test_case["expected_function"] in result:
                correct += 1
            total += 1
    
    return correct / total

accuracy = evaluate("data/test/hr_test.jsonl")
print(f"准确率：{accuracy:.2%}")
```

### 预期效果

| 数据量 | 准确率 | 说明 |
|--------|--------|------|
| 250 样本 | ~75% | 基本可用 |
| 850 样本 | ~85% | 生产质量 |
| 2000+ 样本 | ~90%+ | 优秀 |

## 下一步

- [CRM Agent 示例](./crm-agent.md)
- [部署指南](../deployment.md)
- [性能优化](../best-practices.md)
