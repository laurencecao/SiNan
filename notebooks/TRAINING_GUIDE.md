# 📓 Jupyter Notebook 使用指南

本文档详细介绍如何使用 `training.ipynb` 进行交互式模型微调。

## 📋 目录

- [启动 Notebook](#启动-notebook)
- [使用流程](#使用流程)
- [功能详解](#功能详解)
- [使用技巧](#使用技巧)
- [常见问题](#常见问题)
- [高级用法](#高级用法)

## 启动 Notebook

### 1. 安装依赖

确保已安装 Jupyter 和相关依赖：

```bash
# 安装 Jupyter
pip install jupyter notebook ipywidgets matplotlib pandas

# 或使用项目依赖
pip install -r requirements.txt
```

### 2. 启动 Jupyter Server

```bash
cd /workspace/repos/SiNan

# 启动 Jupyter Notebook
jupyter notebook notebooks/

# 或启动 Jupyter Lab（推荐）
jupyter lab notebooks/
```

### 3. 访问 Notebook

- 浏览器会自动打开 `http://localhost:8888`
- 点击 `training.ipynb` 打开

## 使用流程

### Step 1: 环境初始化 (Cell 1-2)

运行前两个 Cell 检查环境：

```python
# Cell 1 会显示：
# 🔥 PyTorch 版本: 2.x.x
# 🎮 CUDA 可用: True
# 📺 GPU: NVIDIA RTX 4090
# 💾 GPU 显存: 24.00 GB
```

**⚠️ 注意**：如果没有 GPU，训练会非常慢！

### Step 2: 配置参数 (Cell 3)

使用交互式控件配置训练参数：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| 模型名称 | HuggingFace 模型 ID | `google/functiongemma-270m-it` |
| 最大序列长度 | 输入文本最大长度 | 2048 |
| LoRA Rank | 低秩适应维度 | 16 |
| LoRA Alpha | LoRA 缩放因子 | 16 |
| 训练轮数 | Epoch 数量 | 3-5 |
| Batch Size | 每设备批次大小 | 4-8 |
| 学习率 | 优化器学习率 | 2e-4 |
| 梯度累积 | 累积步数 | 4 |

**💡 技巧**：
- 小数据集 (<1000): `rank=8`, `epochs=5-10`, `lr=1e-4`
- 中等数据集 (1000-10000): `rank=16`, `epochs=3-5`, `lr=2e-4`
- 大数据集 (>10000): `rank=32`, `epochs=2-3`, `lr=5e-5`

### Step 3: 加载数据 (Cell 4)

#### 使用现有数据

确保数据路径正确：
```python
# 默认路径
data/processed/train.jsonl
```

#### 创建示例数据

如果没有数据，Cell 会自动创建示例数据。

#### 数据格式

训练数据应为 JSONL 格式，每行一个样本：

```json
{"text": "<<start_of_turn>>user\n查询北京天气<<end_of_turn>>\n<<start_of_turn>>model\n<<start_function_call>>call:get_weather{...}<<end_function_call>><<end_of_turn>>"}
```

### Step 4: 数据分析 (Cell 5)

可视化数据分布：
- 文本长度直方图
- 工具类别分布
- 样本预览

**💡 技巧**：观察文本长度分布，确保大部分样本在 `max_seq_length` 范围内。

### Step 5: 开始训练 (Cell 6-7)

#### 自定义回调

Notebook 包含 `JupyterVisualizationCallback` 类，提供：
- 实时 Loss 曲线
- 学习率变化
- 梯度范数
- Epoch 进度

#### 训练过程

运行 Cell 7 开始训练，你会看到：

```
🚀 训练开始！
📥 加载模型...
✅ 模型加载完成
🎯 开始训练...
```

训练过程中会实时显示图表！

#### 监控指标

图表每 5 步更新一次，显示：
1. **Training Loss** - 训练损失曲线
2. **Learning Rate** - 学习率调度
3. **Gradient Norm** - 梯度范数
4. **Training Progress** - Epoch 进度

### Step 6: 保存模型 (Cell 8)

训练完成后自动保存：
- 模型权重
- Tokenizer
- 配置文件
- 训练指标

### Step 7: 推理测试 (Cell 9-10)

#### 交互式推理

使用交互式界面测试模型：
- 输入提示文本
- 点击"运行推理"
- 查看生成结果

#### 批量测试

批量测试预定义提示：
```python
test_prompts = [
    "查询北京天气",
    "把背景改成蓝色",
    "创建一个名字叫张三的用户",
]
```

### Step 8: 导出模型 (Cell 11)

支持两种导出格式：

#### PyTorch 格式
- 完整的 PyTorch 模型
- 适合进一步微调
- 文件较大

#### GGUF 格式
- 量化模型
- 适合部署到 Ollama/llama.cpp
- 文件较小

**💡 技巧**：生产环境推荐使用 `q8_0` 量化，平衡精度和速度。

## 功能详解

### 1. 实时可视化回调

```python
class JupyterVisualizationCallback(TrainerCallback):
    def __init__(self, update_steps: int = 10):
        # update_steps: 每 N 步更新一次图表
```

**自定义更新频率**：
```python
# 每 1 步更新（更流畅但更耗资源）
callbacks = [JupyterVisualizationCallback(update_steps=1)]

# 每 20 步更新（更节省资源）
callbacks = [JupyterVisualizationCallback(update_steps=20)]
```

### 2. 交互式控件

使用 ipywidgets 创建：

```python
# 滑块
widgets.IntSlider(value=16, min=4, max=64, description='LoRA Rank:')

# 对数滑块（适合学习率）
widgets.FloatLogSlider(value=2e-4, min=-5, max=-3, description='LR:')

# 文本输入
widgets.Text(value='model_name', description='模型:')

# 下拉选择
widgets.Dropdown(options=['pytorch', 'gguf'], description='格式:')
```

### 3. 数据可视化

```python
# 文本长度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
text_lengths = df['text'].str.len()
axes[0].hist(text_lengths, bins=30)

# 工具分布
tool_counts = df['tool_name'].value_counts()
tool_counts.plot(kind='bar', ax=axes[1])
```

## 使用技巧

### 💡 技巧 1: 断点续训

如果想从上次中断的地方继续：

```python
# 修改输出目录为上次的路径
config_widgets['output_dir'].value = 'outputs/models/experiment_20240115_120000'

# 加载已有模型继续训练
trainer.load_model()  # 会自动加载 output_dir 中的模型
```

### 💡 技巧 2: 多组实验对比

创建多个 Notebook 实例，使用不同参数：

| 实验 | LoRA Rank | Learning Rate | Batch Size |
|------|-----------|---------------|------------|
| Exp 1 | 8 | 1e-4 | 8 |
| Exp 2 | 16 | 2e-4 | 4 |
| Exp 3 | 32 | 5e-5 | 2 |

### 💡 技巧 3: 显存优化

如果显存不足：

```python
# 减小 batch size
config_widgets['batch_size'].value = 2

# 增加梯度累积（保持等效 batch size）
config_widgets['gradient_accumulation'].value = 8

# 减小序列长度
config_widgets['max_seq_length'].value = 1024

# 降低 LoRA rank
config_widgets['lora_rank'].value = 8
```

### 💡 技巧 4: 快速验证

在完整训练前先快速验证：

```python
# 使用小数据集快速测试
config_widgets['epochs'].value = 1
config_widgets['data_path'].value = 'data/processed/tiny_sample.jsonl'

# 快速验证数据格式和代码
```

### 💡 技巧 5: 自定义回调

添加自定义回调函数：

```python
class MyCustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step}: 自定义处理")

# 添加到回调列表
callbacks = [
    JupyterVisualizationCallback(),
    MyCustomCallback()
]
```

### 💡 技巧 6: 训练时保存最佳模型

```python
from transformers import EarlyStoppingCallback

# 添加早停回调
callbacks.append(
    EarlyStoppingCallback(early_stopping_patience=3)
)
```

### 💡 技巧 7: 训练后分析

```python
# 加载训练指标
import json
with open(f"{output_dir}/training_metrics.json") as f:
    metrics = json.load(f)

# 绘制完整曲线
plt.plot(metrics['loss_history'])
plt.title("Training Loss")
plt.show()
```

## 常见问题

### Q1: 图表不更新怎么办？

**A**: 
1. 确保已启用 matplotlib 交互模式：`plt.ion()`
2. 检查是否在 Jupyter Lab 而非普通 Notebook
3. 尝试重启 Kernel

### Q2: 交互式控件不显示？

**A**:
```bash
# 安装并启用 ipywidgets
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

# 或在 Jupyter Lab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Q3: 训练时 Kernel 崩溃？

**A**:
- 检查 GPU 显存是否耗尽
- 减小 batch size
- 减小 max_seq_length
- 使用梯度检查点：`use_gradient_checkpointing: true`

### Q4: 如何查看训练历史？

**A**: 训练历史会自动保存到 `metrics_history` 字典：

```python
# 在最后一个 Cell 中查看
from collections import defaultdict
print(callback.metrics_history.keys())
# dict_keys(['loss', 'learning_rate', 'grad_norm', 'epoch'])
```

### Q5: 如何保存高质量图表？

**A**: 修改保存参数：

```python
# 在 JupyterVisualizationCallback 中
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
```

## 高级用法

### 1. 多 GPU 训练

```python
# 自动检测多 GPU
import torch
print(f"GPU 数量: {torch.cuda.device_count()}")

# 使用 DataParallel（在配置中）
training:
  per_device_train_batch_size: 4  # 每个 GPU 的 batch size
```

### 2. 混合精度训练

```python
# 在配置中选择精度
config_dict = {
    'model': {
        'dtype': 'bfloat16',  # 或 'float16', 'float32'
    }
}
```

**推荐**：
- Ampere 架构 GPU (RTX 30xx/40xx, A100): 使用 `bfloat16`
- 旧架构 GPU: 使用 `float16`

### 3. 自定义学习率调度

```python
# 可选调度器类型
lr_scheduler_type_options = [
    'linear',      # 线性衰减
    'cosine',      # 余弦退火（推荐）
    'cosine_with_restarts',  # 带重启的余弦
    'polynomial',  # 多项式衰减
    'constant',    # 常数
    'constant_with_warmup',  # 常数+预热
]
```

### 4. 冻结特定层

```python
# 在加载模型后，冻结底层
for name, param in trainer.model.named_parameters():
    if 'embed' in name or 'lm_head' in name:
        param.requires_grad = False
```

### 5. 使用 Weights & Biases

```python
# 启用 WandB
config_dict['logging']['wandb']['enabled'] = True
config_dict['logging']['wandb']['project'] = 'my-project'
config_dict['logging']['wandb']['name'] = 'experiment-1'
```

### 6. 集成 TensorBoard

```python
# 在 TrainingArguments 中添加
from transformers import TrainingArguments

training_args = TrainingArguments(
    # ... 其他参数
    report_to=["tensorboard"],
    logging_dir="./logs",
)
```

然后在终端运行：
```bash
tensorboard --logdir=./logs
```

### 7. 自动化超参搜索

```python
# 使用简单循环测试多组参数
learning_rates = [1e-4, 2e-4, 5e-4]
ranks = [8, 16, 32]

for lr in learning_rates:
    for rank in ranks:
        config_widgets['learning_rate'].value = lr
        config_widgets['lora_rank'].value = rank
        # ... 运行训练
```

## 最佳实践

### ✅ 应该做的

1. **始终检查 GPU 状态** - 确保 CUDA 可用且显存充足
2. **从小参数开始** - 先快速验证，再放大训练
3. **监控显存使用** - 使用 `nvidia-smi` 监控
4. **保存训练日志** - 便于后续分析和复现
5. **使用验证集** - 防止过拟合
6. **版本控制配置** - 记录每次实验的参数

### ❌ 不应该做的

1. **不要一开始就用大 batch size** - 容易导致 OOM
2. **不要忽视数据质量** - 垃圾进垃圾出
3. **不要训练太久** - 注意观察验证集指标，防止过拟合
4. **不要频繁修改多个参数** - 一次只改一个，便于定位问题

## 参考资源

- [Jupyter Notebook 官方文档](https://jupyter-notebook.readthedocs.io/)
- [ipywidgets 文档](https://ipywidgets.readthedocs.io/)
- [matplotlib 教程](https://matplotlib.org/tutorials/index.html)
- [HuggingFace Trainer 文档](https://huggingface.co/docs/transformers/main_classes/trainer)

---

**提示**: 如果有其他问题，欢迎在 GitHub Issues 中提问！
