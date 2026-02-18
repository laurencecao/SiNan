# 架构设计

本文档详细介绍 SiNan 的系统架构、模块设计和数据流。

## 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         SiNan 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  数据层     │    │   配置层     │    │    核心引擎     │   │
│  │  Data Layer │───▶│ Config Layer │───▶│  Core Engine    │   │
│  │             │    │              │    │                 │   │
│  │ • Excel/CSV │    │ • OmegaConf  │    │ • Unsloth       │   │
│  │ • JSONL     │    │ • YAML       │    │ • LoRA/QLoRA    │   │
│  │ • Dataset   │    │ • 实验配置   │    │ • SFTTrainer    │   │
│  └─────────────┘    └──────────────┘    └────────┬────────┘   │
│                                                   │             │
│                                                   ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              监控与产物 (Monitoring & Artifacts)         │   │
│  │                                                         │   │
│  │  • WandB Dashboard  • LoRA Adapters  • GGUF/Ollama     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 模块设计

### 1. 数据引擎模块 (`src/data_engine/`)

负责将企业原始业务数据转换为 FunctionGemma 训练格式。

```
┌────────────────────────────────────────────┐
│         Data Engine                        │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  Converter   │─────▶│  Formatter   │   │
│  │              │      │              │   │
│  │ • Excel 读取  │      │ • Token 注入  │   │
│  │ • CSV 解析    │      │ • 格式验证   │   │
│  │ • 数据校验   │      │ • 模板生成   │   │
│  └──────────────┘      └──────────────┘   │
│                                            │
└────────────────────────────────────────────┘
```

#### Converter (转换器)

**文件**: `src/data_engine/converter.py`

**职责**:
- 读取 Excel/CSV 文件
- 验证数据完整性
- 转换为标准 JSONL 格式
- 支持批量转换

**核心方法**:
```python
class DataConverter:
    def read_file(self, file_path: str) -> pd.DataFrame
    def validate_row(self, row: pd.Series) -> tuple[bool, str]
    def convert(self, input_path: str, output_path: str) -> ConversionResult
    def convert_batch(self, input_dir: str, output_dir: str) -> dict
```

#### Formatter (格式化器)

**文件**: `src/data_engine/formatter.py`

**职责**:
- 将 JSONL 转换为 FunctionGemma 特殊 Token 格式
- 处理 `<escape>` 字符串包裹
- 生成函数声明、调用、响应格式

**核心方法**:
```python
class FunctionGemmaFormatter:
    def format_function_declaration(...) -> str
    def format_function_call(...) -> str
    def format_function_response(...) -> str
    def format_training_sample(...) -> str
```

### 2. 配置管理模块 (`src/utils/config_loader.py`)

**职责**:
- 使用 OmegaConf 加载 YAML 配置
- 支持配置继承和覆盖
- 解析相对路径

**配置层级**:
```
base_config.yaml (基础配置)
       ↓
experiments/*.yaml (实验覆盖)
       ↓
CLI overrides (命令行覆盖)
```

**使用示例**:
```python
from src.utils.config_loader import load_config

config = load_config(
    config_name="base_config",
    experiment="exp_hr_routing",
    overrides=["training.epochs=10"]
)
```

### 3. 训练引擎模块 (`src/training/`)

```
┌────────────────────────────────────────────┐
│         Training Engine                    │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────┐      ┌──────────────┐   │
│  │   Trainer    │◀────▶│  Callbacks   │   │
│  │              │      │              │   │
│  │ • 模型加载   │      │ • WandB      │   │
│  │ • LoRA 配置   │      │ • 样本生成   │   │
│  │ • 训练循环   │      │ • 早停       │   │
│  └──────────────┘      └──────────────┘   │
│                                            │
└────────────────────────────────────────────┘
```

#### Trainer (训练器)

**文件**: `src/training/trainer.py`

**职责**:
- 加载 FunctionGemma 模型
- 配置 LoRA 适配器
- 执行训练循环
- 保存训练结果

**核心流程**:
```python
class FunctionGemmaTrainer:
    def load_model(self)           # 加载模型和配置 LoRA
    def load_dataset(self, path)   # 加载数据集
    def train(self, dataset)       # 执行训练
    def save_model(self, path)     # 保存模型
    def inference(self, text)      # 推理测试
```

#### Callbacks (回调函数)

**文件**: `src/training/callbacks.py`

**内置回调**:
- `WandbCallback` - 记录训练指标到 WandB
- `SampleGenerationCallback` - Epoch 结束生成测试样本
- `EarlyStoppingCallback` - 验证损失不下降时早停

### 4. 工具模块 (`src/utils/`)

#### Export (导出工具)

**文件**: `src/utils/export.py`

**支持格式**:
- PyTorch (原始格式)
- GGUF (llama.cpp / Ollama)
- 量化格式 (q8_0, q4_k_m, q4_0 等)

**导出流程**:
```
训练完成的模型
       ↓
合并 LoRA 适配器
       ↓
选择导出格式
       ↓
量化 (可选)
       ↓
GGUF/PyTorch 文件
```

## 数据流

### 完整训练流程

```
1. 原始数据 (Excel/CSV)
       │
       ▼
2. DataConverter 转换
       │
       ▼
3. JSONL 格式
       │
       ▼
4. FunctionGemmaFormatter 格式化
       │
       ▼
5. FunctionGemma Token 格式
       │
       ▼
6. FunctionGemmaTrainer 训练
       │
       ▼
7. LoRA Adapters
       │
       ▼
8. Export 导出
       │
       ├──────▶ PyTorch 格式
       │
       └──────▶ GGUF 格式 (量化)
```

### 推理流程

```
用户输入
   │
   ▼
加载模型 (FastLanguageModel)
   │
   ▼
构建消息 (apply_chat_template)
   │
   ▼
模型生成 (generate)
   │
   ▼
解析输出 (decode)
   │
   ▼
提取函数调用
   │
   ▼
执行函数
   │
   ▼
返回结果
```

## 技术选型

| 组件 | 技术 | 理由 |
|------|------|------|
| **基座模型** | FunctionGemma 270M | 轻量级，专为函数调用优化 |
| **训练框架** | Unsloth | 训练速度 2x，显存 -60% |
| **微调方法** | LoRA | 高效参数微调 |
| **配置管理** | OmegaConf | YAML 支持，类型安全 |
| **数据处理** | Pandas | 强大的表格处理 |
| **监控** | WandB | 实时可视化 |
| **部署格式** | GGUF | 兼容 llama.cpp/Ollama |

## 扩展点

### 自定义数据源

继承 `DataConverter` 实现自定义读取逻辑:

```python
class CustomConverter(DataConverter):
    def read_file(self, file_path: str) -> pd.DataFrame:
        # 实现自定义读取逻辑
        pass
```

### 自定义回调

继承 `TrainerCallback` 实现自定义逻辑:

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 自定义步骤结束逻辑
        pass
```

### 自定义导出格式

在 `export.py` 中添加新的导出函数:

```python
def export_custom_format(model, tokenizer, output_dir):
    # 实现自定义导出逻辑
    pass
```

## 性能优化

### 显存优化

```yaml
# configs/base_config.yaml
model:
  dtype: "bfloat16"  # 使用 bfloat16 减少显存
  lora:
    use_gradient_checkpointing: true  # 激活检查点节省显存

training:
  per_device_train_batch_size: 4  # 根据显存调整
  gradient_accumulation_steps: 4  # 累积梯度保持有效 batch size
```

### 训练速度优化

```yaml
training:
  optimizer: "adamw_torch"  # 优化的 AdamW
  lr_scheduler_type: "cosine"  # 余弦退火
  warmup_ratio: 0.1  # 10% warmup
```

## 安全考虑

- **数据验证**: 所有输入数据经过严格验证
- **路径解析**: 防止路径遍历攻击
- **模型导出**: 仅导出 LoRA 适配器，不包含敏感数据

## 未来规划

- [ ] 支持多 GPU 训练
- [ ] 添加 DPO/RLHF 支持
- [ ] 集成更多量化工具 (AWQ, GPTQ)
- [ ] 支持流式推理
- [ ] 添加模型评估基准
