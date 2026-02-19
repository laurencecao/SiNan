# SiNan - FunctionGemma 企业微调框架

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SiNan** 是一个基于 Google FunctionGemma 270M 和 Unsloth 框架的企业级 AI 路由微调系统。通过简单的文本输入，自动调用预定义的企业方法，实现 AI 化的智能路由。

## ✨ 特性

- 🚀 **极速训练** - 基于 Unsloth，训练速度提升 2 倍，显存减少 60%
- 📓 **交互式 Notebook** - Jupyter Notebook 可视化训练，实时查看 Loss 曲线和指标
- 📊 **Excel/CSV 支持** - 直接将企业业务数据转换为训练格式
- 🎯 **FunctionGemma 优化** - 专为函数调用优化的轻量级模型 (270M)
- 🔧 **配置化** - OmegaConf 实现"配置即代码"
- 📈 **实时监控** - WandB 集成 + Notebook 实时可视化，多维度查看训练指标
- 📦 **一键部署** - 支持 GGUF 量化，可部署到 CPU/GPU/边缘设备

## 📦 安装

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-org/SiNan.git
cd SiNan

# 创建虚拟环境
uv venv .venv
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 云端安装 (推荐)

```bash
# 运行环境初始化脚本
bash scripts/setup_env.sh

# 激活环境
conda activate function_gemma_env
```

## 🚀 快速开始

我们提供两种训练方式：**交互式 Jupyter Notebook**（推荐初学者）和 **命令行 CLI**（适合生产环境）。

### 📝 方式一：交互式 Notebook（推荐）

使用 Jupyter Notebook 进行可视化训练：

```bash
# 启动 Jupyter
jupyter notebook notebooks/

# 打开 training.ipynb，按顺序运行 Cell
```

**Notebook 优势：**
- 📊 **实时可视化** - Loss 曲线、学习率动态绘制
- 🎛️ **交互式配置** - 拖拽滑块调整参数
- 🔍 **即时反馈** - 每步训练结果立即可见
- 🧪 **快速实验** - 无需写代码即可对比多组参数

详细使用指南：[notebooks/TRAINING_GUIDE.md](notebooks/TRAINING_GUIDE.md)

### 🖥️ 方式二：命令行 CLI

适合自动化部署和批处理：

### 1. 准备数据

创建 Excel 文件 (如 `data/raw/hr_functions.xlsx`)，包含以下列：

| User Prompt | Tool Name | Tool Args |
|-------------|-----------|-----------|
| 查询北京天气 | get_weather | `{"location": "Beijing"}` |
| 把背景改成红色 | change_background | `{"color": "red"}` |
| 创建新用户 | create_user | `{"name": "张三", "age": 25}` |

### 2. 转换数据

```bash
# 转换 Excel 为 JSONL
python main.py convert data/raw/hr_functions.xlsx data/processed/hr_functions.jsonl

# 批量转换整个目录
python main.py convert data/raw/ data/processed/
```

### 3. 配置训练

编辑 `configs/experiments/exp_hr_routing.yaml` (可选):

```yaml
training:
  epochs: 5
  learning_rate: 1.0e-4
  per_device_train_batch_size: 8
```

### 4. 开始训练

```bash
# 使用基础配置
python main.py train --data data/processed/hr_functions.jsonl --output outputs/models/hr_v1

# 使用实验配置
python main.py train --data data/processed/ --experiment exp_hr_routing
```

### 5. 导出模型

```bash
# 导出为 GGUF 格式 (用于 Ollama/llama.cpp)
python main.py export outputs/models/hr_v1 outputs/models/hr_v1_gguf --format gguf --quantization q8_0

# 导出为 PyTorch 格式
python main.py export outputs/models/hr_v1 outputs/models/hr_v1_pt --format pytorch
```

### 6. 推理测试

```bash
python main.py inference outputs/models/hr_v1 --prompt "查询北京天气"
```

## 📖 命令行接口

```bash
# 查看所有命令
python main.py --help

# 数据转换
python main.py convert --help

# 训练
python main.py train --help

# 导出
python main.py export --help

# 推理
python main.py inference --help
```

## 📁 项目结构

```
SiNan/
├── configs/                    # 配置文件
│   ├── base_config.yaml        # 基础配置
│   ├── experiments/            # 实验配置
│   └── templates/              # 模板配置
├── data/                       # 数据目录
│   ├── raw/                    # 原始 Excel/CSV
│   └── processed/              # 处理后的 JSONL
├── notebooks/                  # Jupyter Notebooks
│   ├── training.ipynb          # 交互式训练 Notebook
│   ├── TRAINING_GUIDE.md       # Notebook 详细使用指南
│   └── README.md               # Notebooks 说明
├── outputs/                    # 输出目录
│   ├── logs/                   # 训练日志
│   └── models/                 # 训练好的模型
├── scripts/                    # Shell 脚本
│   ├── setup_env.sh            # 环境初始化
│   └── run_cloud_train.sh      # 云端训练
├── src/                        # 源代码
│   ├── data_engine/            # 数据引擎
│   │   ├── converter.py        # 数据转换
│   │   └── formatter.py        # 格式化器
│   ├── training/               # 训练模块
│   │   ├── trainer.py          # 训练器
│   │   └── callbacks.py        # 回调函数
│   └── utils/                  # 工具类
│       ├── config_loader.py    # 配置加载
│       └── export.py           # 模型导出
├── main.py                     # CLI 主入口
├── requirements.txt            # 依赖
└── README.md                   # 项目文档
```

## 🔧 配置说明

### 基础配置 (`configs/base_config.yaml`)

```yaml
model:
  name: "google/functiongemma-270m-it"
  max_seq_length: 2048
  dtype: "bfloat16"
  lora:
    rank: 16
    alpha: 16
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"

training:
  epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
```

### 实验配置 (`configs/experiments/exp_hr_routing.yaml`)

```yaml
training:
  epochs: 5
  learning_rate: 1.0e-4

logging:
  wandb:
    project: "hr-routing-experiment"
```

## 📊 训练数据格式

### JSONL 格式

每行一个训练样本：

```json
{"user_content": "查询北京天气", "tool_name": "get_weather", "tool_arguments": {"location": "Beijing"}}
{"user_content": "把背景改成红色", "tool_name": "change_background", "tool_arguments": {"color": "red"}}
```

### FunctionGemma Token 格式

```
<<start_of_turn>>developer
You are a model that can do function calling with the following functions
<<start_function_declaration>>declaration:get_weather{description:<<escape>>获取天气<<escape>>,parameters:{...}}<<end_function_declaration>>
<<end_of_turn>>
<<start_of_turn>>user
查询北京天气<<end_of_turn>>
<<start_of_turn>>model
<<start_function_call>>call:get_weather{location:<<escape>>Beijing<<escape>>}<<end_function_call>>
<<end_of_turn>>
```

## 📈 监控与可视化

### 🎯 Notebook 实时可视化

使用 `training.ipynb` 可获得最佳可视化体验：

- **实时指标曲线** - Loss、学习率、梯度范数动态绘制
- **交互式参数调整** - 拖拽滑块实时修改训练参数
- **数据分布可视化** - 文本长度、工具类别统计图表
- **即时推理测试** - 训练完成后立即测试模型效果

启动 Notebook：
```bash
jupyter notebook notebooks/training.ipynb
```

### WandB 集成

训练自动记录到 Weights & Biases:

```bash
# 登录 WandB
wandb login

# 训练时自动记录
python main.py train --data data/processed/ --experiment exp_hr_routing
```

访问 https://wandb.ai 查看训练指标。

## 🚀 云端部署

### AWS EC2

```bash
# 启动实例 (推荐 g4dn.xlarge - T4 GPU)
aws ec2 run-instances --image-id ami-0c55b159cbfafe1f0 --instance-type g4dn.xlarge

# SSH 连接
ssh -i key.pem ec2-user@<instance-ip>

# 运行初始化脚本
bash scripts/setup_env.sh

# 训练
bash scripts/run_cloud_train.sh exp_hr_routing
```

### AutoDL / RunPod

```bash
# 选择 GPU 实例 (T4/L4/A10)
# 上传项目代码
# 运行初始化脚本
bash scripts/setup_env.sh
```

## 📝 最佳实践

### Notebook 使用技巧

**1. 快速原型验证**
```python
# 先使用小参数快速验证
epochs = 1
batch_size = 2
data_subset = 100  # 只用 100 条样本
```

**2. 多组实验对比**
- 打开多个 Notebook 窗口
- 使用不同参数并行实验
- 对比不同配置的训练曲线

**3. 显存不足时的调整**
```python
# 减小 batch size + 增加梯度累积
batch_size = 2
gradient_accumulation_steps = 8  # 等效 batch size = 16

# 减小序列长度
max_seq_length = 1024  # 默认 2048

# 降低 LoRA rank
lora_rank = 8  # 默认 16
```

**4. 断点续训**
```python
# 修改输出目录为已有路径
output_dir = 'outputs/models/experiment_20240115_120000'
# 会自动加载已有模型继续训练
```

**5. 保存最佳模型**
```python
# 在训练配置中启用早停
early_stopping = true
early_stopping_patience = 3
```

详细技巧参考：[notebooks/TRAINING_GUIDE.md](notebooks/TRAINING_GUIDE.md#使用技巧)

### 数据量建议

| 场景 | 函数数 | 每函数样本 | 总样本 | 预期准确率 |
|------|--------|-----------|--------|----------|
| 最小可行 | 3-5 | 50-100 | 150-500 | ~70% |
| 生产推荐 | 10-20 | 200-500 | 2000-10000 | ~85%+ |
| 高质量 | 20+ | 500+ | 10000+ | ~90%+ |

### 超参数调优

```yaml
# 小数据集 (<1000 样本)
training:
  epochs: 5-10
  learning_rate: 1.0e-4
  lora.rank: 8

# 中等数据集 (1000-10000)
training:
  epochs: 3-5
  learning_rate: 2.0e-4
  lora.rank: 16

# 大数据集 (>10000)
training:
  epochs: 2-3
  learning_rate: 5.0e-5
  lora.rank: 32
```

## 🛠️ 故障排除

### CUDA Out of Memory

```yaml
# 减少 batch size
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

### 训练 Loss 不下降

- 检查数据格式是否正确
- 增加学习率
- 增加训练轮数
- 检查 Token 格式是否符合 FunctionGemma 规范

### 推理结果不正确

- 确保训练数据质量
- 增加每函数样本数
- 调整推理参数 (temperature, top_k, top_p)

### Notebook 图表不显示

- 确保已安装 `ipywidgets`: `pip install ipywidgets`
- 启用扩展: `jupyter nbextension enable --py widgetsnbextension`
- 重启 Jupyter Kernel

### Notebook Kernel 崩溃

- 检查 GPU 显存是否耗尽: `nvidia-smi`
- 减小 batch size 和序列长度
- 关闭其他 Notebook 释放资源
- 详细解决方案: [TRAINING_GUIDE.md#常见问题](notebooks/TRAINING_GUIDE.md#常见问题)

## 📚 参考资料

### 官方文档
- [FunctionGemma 官方文档](https://ai.google.dev/gemma/docs/functiongemma)
- [Unsloth 文档](https://unsloth.ai/docs)
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [OmegaConf](https://omegaconf.readthedocs.io/)

### 项目文档
- [📓 Notebook 使用指南](notebooks/TRAINING_GUIDE.md) - 详细的交互式训练教程
- [🔧 配置说明](#配置说明) - 配置文件详解
- [📊 训练数据格式](#训练数据格式) - 数据格式规范
- [📝 最佳实践](#最佳实践) - 推荐的使用方法

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

## 📧 联系方式

- Email: your-email@example.com
- GitHub Issues: [提交问题](https://github.com/your-org/SiNan/issues)
