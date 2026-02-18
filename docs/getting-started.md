# 快速开始

本指南帮助你在 5 分钟内上手 SiNan，完成从数据准备到模型推理的完整流程。

## 前置要求

- Python 3.11+
- 8GB+ RAM (GPU 推荐 16GB+)
- 10GB+ 可用磁盘空间
- (可选) NVIDIA GPU (4GB+ VRAM)

## 第一步：安装 (2 分钟)

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/SiNan.git
cd SiNan
```

### 2. 创建虚拟环境

```bash
# 使用 uv (推荐，更快)
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 或使用 venv
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 4. 验证安装

```bash
python main.py --help
```

看到帮助信息说明安装成功！

## 第二步：准备数据 (1 分钟)

### 创建示例数据

创建 Excel 文件 `data/raw/demo.xlsx`，包含以下列：

| User Prompt | Tool Name | Tool Args |
|-------------|-----------|-----------|
| 查询北京天气 | get_weather | `{"location": "Beijing"}` |
| 查询上海天气 | get_weather | `{"location": "Shanghai"}` |
| 把背景改成红色 | change_color | `{"color": "red"}` |
| 把背景改成蓝色 | change_color | `{"color": "blue"}` |
| 创建用户张三 | create_user | `{"name": "张三", "age": 25}` |
| 创建用户李四 | create_user | `{"name": "李四", "age": 30}` |

**提示**: 至少准备 50-100 条样本以获得较好效果。

## 第三步：转换数据 (30 秒)

```bash
python main.py convert data/raw/demo.xlsx data/processed/demo.jsonl
```

输出示例:
```
✓ 转换完成：6/6 行有效
```

## 第四步：训练模型 (2-5 分钟)

### 快速训练 (测试用)

```bash
python main.py train \
  --data data/processed/demo.jsonl \
  --output outputs/models/demo_v1
```

### 正式训练 (推荐配置)

```bash
python main.py train \
  --data data/processed/demo.jsonl \
  --experiment exp_hr_routing \
  --output outputs/models/demo_v1
```

训练过程:
```
加载模型：google/functiongemma-270m-it
配置 LoRA...
数据集大小：6
开始训练...
[1/3] Loss: 2.345
[2/3] Loss: 1.234
[3/3] Loss: 0.567
✓ 训练完成，模型已保存到：outputs/models/demo_v1
```

## 第五步：推理测试 (30 秒)

```bash
python main.py inference \
  outputs/models/demo_v1 \
  --prompt "查询北京天气"
```

输出示例:
```
输入:
查询北京天气

输出:
<<start_of_turn>>model
<<start_function_call>>call:get_weather{location:<<escape>>Beijing<<escape>>}<<end_function_call>>
<<end_of_turn>>
```

模型成功识别出应该调用 `get_weather` 函数！

## 下一步

### 📚 深入学习

- [数据格式详解](./data-format.md) - 了解训练数据格式
- [训练指南](./training-guide.md) - 超参数调优和最佳实践
- [部署指南](./deployment.md) - 模型量化和部署

### 🔧 进阶使用

```bash
# 导出为 GGUF 格式 (用于 Ollama)
python main.py export outputs/models/demo_v1 outputs/models/demo_gguf \
  --format gguf \
  --quantization q8_0

# 使用 Ollama 运行
ollama create functiongemma-demo -f outputs/models/demo_gguf/Modelfile
ollama run functiongemma-demo "查询北京天气"
```

### 📊 监控训练

```bash
# 登录 WandB
wandb login

# 训练时自动记录
python main.py train --data data/processed/ --experiment exp_hr_routing
```

访问 https://wandb.ai 查看实时训练指标。

## 常见问题

### Q: 训练很慢怎么办？

A: 使用 GPU 可以显著提升速度。如果没有 GPU，可以减少训练样本数进行测试。

### Q: 显存不足怎么办？

A: 在 `configs/base_config.yaml` 中减少 `per_device_train_batch_size`:

```yaml
training:
  per_device_train_batch_size: 2  # 从 4 减少到 2
  gradient_accumulation_steps: 8  # 增加到 8 保持有效 batch size
```

### Q: 准确率不高怎么办？

A: 
1. 增加训练样本数 (每函数 200-500 样本)
2. 增加训练轮数 (`epochs: 5-10`)
3. 检查数据质量 (确保函数调用格式正确)

## 获取帮助

- [完整文档](./README.md)
- [常见问题](./faq.md)
- [GitHub Issues](https://github.com/your-org/SiNan/issues)
