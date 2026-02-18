# SiNan - FunctionGemma 企业微调框架

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SiNan** 是一个基于 Google FunctionGemma 270M 和 Unsloth 框架的企业级 AI 路由微调系统。通过简单的文本输入，自动调用预定义的企业方法，实现 AI 化的智能路由。

## ✨ 特性

- 🚀 **极速训练** - 基于 Unsloth，训练速度提升 2 倍，显存减少 60%
- 📊 **Excel/CSV 支持** - 直接将企业业务数据转换为训练格式
- 🎯 **FunctionGemma 优化** - 专为函数调用优化的轻量级模型 (270M)
- 🔧 **配置化** - OmegaConf 实现"配置即代码"
- 📈 **实时监控** - WandB 集成，实时查看训练指标
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

## 📚 参考资料

- [FunctionGemma 官方文档](https://ai.google.dev/gemma/docs/functiongemma)
- [Unsloth 文档](https://unsloth.ai/docs)
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [OmegaConf](https://omegaconf.readthedocs.io/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

## 📧 联系方式

- Email: your-email@example.com
- GitHub Issues: [提交问题](https://github.com/your-org/SiNan/issues)
