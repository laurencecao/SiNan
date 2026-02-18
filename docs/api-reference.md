# API 参考

本文档提供 SiNan 核心模块的 API 参考。

## 数据引擎

### DataConverter

**模块**: `src.data_engine.converter`

```python
class DataConverter:
    """数据转换器 - Excel/CSV 转 JSONL"""
    
    def __init__(
        self,
        user_prompt_col: str = "User Prompt",
        tool_name_col: str = "Tool Name",
        tool_args_col: str = "Tool Args",
        validate: bool = True
    )
    
    def read_file(self, file_path: str) -> pd.DataFrame
    def validate_row(self, row: pd.Series) -> tuple[bool, str]
    def convert(
        self,
        input_path: str,
        output_path: str,
        add_distractors: bool = False
    ) -> ConversionResult
    def convert_batch(
        self,
        input_dir: str,
        output_dir: str
    ) -> dict[str, ConversionResult]
```

**使用示例**:

```python
from src.data_engine import DataConverter

converter = DataConverter(
    user_prompt_col="员工问题",
    tool_name_col="HR 功能",
    validate=True
)

result = converter.convert(
    "data/raw/hr.xlsx",
    "data/processed/hr.jsonl"
)

print(f"转换完成：{result.valid_rows}/{result.total_rows}")
```

### FunctionGemmaFormatter

**模块**: `src.data_engine.formatter`

```python
class FunctionGemmaFormatter:
    """FunctionGemma 格式化器"""
    
    def __init__(
        self,
        max_seq_length: int = 2048,
        add_generation_prompt: bool = True
    )
    
    @staticmethod
    def escape_string(value: str) -> str
    
    def format_function_declaration(
        self,
        name: str,
        description: str,
        parameters: dict = None
    ) -> str
    
    def format_function_call(
        self,
        name: str,
        arguments: dict
    ) -> str
    
    def format_function_response(
        self,
        name: str,
        result: dict
    ) -> str
    
    def format_training_sample(
        self,
        user_content: str,
        tool_name: str,
        tool_arguments: dict
    ) -> str
```

**使用示例**:

```python
from src.data_engine import FunctionGemmaFormatter

formatter = FunctionGemmaFormatter()

# 格式化函数声明
declaration = formatter.format_function_declaration(
    name="get_weather",
    description="获取天气",
    parameters={
        "properties": {
            "location": {
                "description": "城市名",
                "type": "string",
                "required": True
            }
        }
    }
)

# 格式化函数调用
call = formatter.format_function_call(
    name="get_weather",
    arguments={"location": "Beijing"}
)
```

## 训练模块

### FunctionGemmaTrainer

**模块**: `src.training.trainer`

```python
class FunctionGemmaTrainer:
    """FunctionGemma 训练器"""
    
    def __init__(self, config: DictConfig)
    
    def load_model(self)
    def load_dataset(self, data_path: str) -> Dataset
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        callbacks: list = None
    ) -> TrainResult
    def save_model(self, output_dir: str)
    def evaluate(self, eval_dataset: Dataset) -> dict
    def inference(self, text: str, max_new_tokens: int = 128) -> str
```

**使用示例**:

```python
from src.training import FunctionGemmaTrainer
from src.utils.config_loader import load_config
from datasets import load_dataset

# 加载配置
config = load_config()

# 创建训练器
trainer = FunctionGemmaTrainer(config)

# 加载数据
dataset = load_dataset('json', data_files='data/processed/hr.jsonl')

# 训练
trainer.load_model()
trainer.train(
    train_dataset=dataset['train'],
    output_dir='outputs/models/hr_v1'
)

# 保存
trainer.save_model('outputs/models/hr_v1')

# 推理
result = trainer.inference("查询张三的年假")
print(result)
```

### Callbacks

**模块**: `src.training.callbacks`

```python
class WandbCallback(TrainerCallback):
    """WandB 监控回调"""
    
    def __init__(
        self,
        project: str,
        entity: str = None,
        name: str = None
    )

class SampleGenerationCallback(TrainerCallback):
    """样本生成回调"""
    
    def __init__(
        self,
        tokenizer,
        test_prompts: list[str],
        generation_kwargs: dict = None
    )

class EarlyStoppingCallback(TrainerCallback):
    """早停回调"""
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4
    )
```

**使用示例**:

```python
from src.training.callbacks import (
    WandbCallback,
    SampleGenerationCallback,
    EarlyStoppingCallback
)

callbacks = [
    WandbCallback(project="hr-routing"),
    SampleGenerationCallback(
        tokenizer=trainer.tokenizer,
        test_prompts=["查询天气", "创建用户"]
    ),
    EarlyStoppingCallback(patience=3)
]

trainer.train(dataset, callbacks=callbacks)
```

## 工具模块

### Config Loader

**模块**: `src.utils.config_loader`

```python
def load_config(
    config_name: str = "base_config",
    config_dir: str = "configs",
    experiment: str = None,
    overrides: list[str] = None
) -> DictConfig

def save_config(
    config: DictConfig,
    output_path: str
)

def print_config(config: DictConfig)
```

**使用示例**:

```python
from src.utils.config_loader import load_config, print_config

# 加载配置
config = load_config(
    experiment="exp_hr_routing",
    overrides=["training.epochs=10"]
)

# 查看配置
print_config(config)

# 保存配置
save_config(config, "outputs/config_used.yaml")
```

### Export

**模块**: `src.utils.export`

```python
def export_model(
    model,
    tokenizer,
    output_dir: str,
    export_format: str = "pytorch",
    merge_lora: bool = True,
    quantization: str = "q8_0"
) -> str

def export_gguf(
    model,
    tokenizer,
    output_dir: str,
    quantization: str = "q8_0"
) -> str

def export_to_ollama(
    gguf_path: str,
    model_name: str
) -> str
```

**使用示例**:

```python
from src.utils.export import export_model, export_to_ollama

# 导出 GGUF
gguf_path = export_model(
    model=trained_model,
    tokenizer=tokenizer,
    output_dir="outputs/models/hr_gguf",
    export_format="gguf",
    quantization="q8_0"
)

# 导入 Ollama
ollama_name = export_to_ollama(
    gguf_path=gguf_path,
    model_name="functiongemma-hr"
)
```

## CLI 命令

### 转换数据

```bash
python main.py convert \
  input.xlsx \
  output.jsonl \
  --user-col "User Prompt" \
  --tool-col "Tool Name" \
  --args-col "Tool Args" \
  --no-validate \
  --distractors \
  -v
```

### 训练

```bash
python main.py train \
  --data data/processed/hr.jsonl \
  --experiment exp_hr_routing \
  --output outputs/models/hr_v1 \
  -v
```

### 导出

```bash
python main.py export \
  outputs/models/hr_v1 \
  outputs/models/hr_gguf \
  --format gguf \
  --quantization q8_0 \
  --merge-lora
```

### 推理

```bash
python main.py inference \
  outputs/models/hr_v1 \
  --prompt "查询张三的年假" \
  --max-tokens 256
```

### 配置

```bash
# 查看配置
python main.py config \
  --experiment exp_hr_routing

# 使用自定义配置
python main.py config \
  --config custom_config
```

## 配置选项

### Model

```yaml
model:
  name: "google/functiongemma-270m-it"
  max_seq_length: 2048
  dtype: "bfloat16"
  lora:
    enabled: true
    rank: 16
    alpha: 16
    dropout: 0.0
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
```

### Training

```yaml
training:
  epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping: false
  early_stopping_patience: 3
```

### Logging

```yaml
logging:
  wandb:
    enabled: true
    project: "functiongemma-hr-routing"
    entity: "your-team"
  output_dir: "outputs"
  log_dir: "outputs/logs"
```

### Export

```yaml
export:
  formats:
    - "pytorch"
    - "gguf"
  gguf:
    quantization: "q8_0"
  merge_lora: true
  export_dir: "outputs/models"
```

## 错误处理

### DataConversionError

```python
from src.data_engine.converter import DataConversionError

try:
    converter.convert("input.xlsx", "output.jsonl")
except DataConversionError as e:
    print(f"转换失败：{e}")
```

### TrainingError

```python
from src.training.trainer import TrainingError

try:
    trainer.train(dataset)
except TrainingError as e:
    print(f"训练失败：{e}")
```

### ExportError

```python
from src.utils.export import ExportError

try:
    export_model(model, tokenizer, "output/")
except ExportError as e:
    print(f"导出失败：{e}")
```
