# 训练指南

本文档详细介绍如何训练 FunctionGemma 模型，包括超参数调优、监控和最佳实践。

## 训练流程概览

```
1. 准备数据 → 2. 配置参数 → 3. 启动训练 → 4. 监控指标 → 5. 评估导出
```

## 1. 训练前准备

### 检查数据

```bash
# 查看数据量
wc -l data/processed/hr_functions.jsonl

# 预览数据
head -n 5 data/processed/hr_functions.jsonl
```

### 验证配置

```bash
# 查看当前配置
python main.py config --experiment exp_hr_routing
```

### 硬件检查

```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 2. 启动训练

### 基础训练

```bash
python main.py train \
  --data data/processed/hr_functions.jsonl \
  --output outputs/models/hr_v1
```

### 使用实验配置

```bash
python main.py train \
  --data data/processed/hr_functions.jsonl \
  --experiment exp_hr_routing \
  --output outputs/models/hr_v1
```

### 云端训练

```bash
# 后台运行，防止 SSH 断开
bash scripts/run_cloud_train.sh exp_hr_routing

# 查看日志
tail -f outputs/logs/exp_hr_routing_*.log
```

## 3. 超参数配置

### 基础配置 (`configs/base_config.yaml`)

```yaml
model:
  name: "google/functiongemma-270m-it"
  max_seq_length: 2048
  dtype: "bfloat16"  # 或 "float16"
  
  lora:
    rank: 16         # LoRA 秩
    alpha: 16        # LoRA alpha
    target_modules:  # 目标模块
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

training:
  epochs: 3                      # 训练轮数
  learning_rate: 2.0e-4          # 学习率
  per_device_train_batch_size: 4 # 批次大小
  gradient_accumulation_steps: 4 # 梯度累积
  lr_scheduler_type: "cosine"    # 学习率调度器
  warmup_ratio: 0.1              # 预热比例
  weight_decay: 0.01             # 权重衰减
```

### 超参数调优指南

#### 根据数据量调整

**小数据集 (<1000 样本)**:
```yaml
training:
  epochs: 5-10
  learning_rate: 1.0e-4
  per_device_train_batch_size: 4
model:
  lora:
    rank: 8
    alpha: 8
```

**中等数据集 (1000-10000)**:
```yaml
training:
  epochs: 3-5
  learning_rate: 2.0e-4
  per_device_train_batch_size: 8
model:
  lora:
    rank: 16
    alpha: 16
```

**大数据集 (>10000)**:
```yaml
training:
  epochs: 2-3
  learning_rate: 5.0e-5
  per_device_train_batch_size: 16
model:
  lora:
    rank: 32
    alpha: 32
```

#### 根据显存调整

**显存不足时**:
```yaml
training:
  per_device_train_batch_size: 2      # 减小批次
  gradient_accumulation_steps: 8      # 增加累积
model:
  lora:
    use_gradient_checkpointing: true  # 激活检查点
```

**显存充足时**:
```yaml
training:
  per_device_train_batch_size: 16     # 增大批次
  gradient_accumulation_steps: 2      # 减少累积
```

## 4. 训练监控

### WandB 集成

#### 配置 WandB

```yaml
# configs/base_config.yaml
logging:
  wandb:
    enabled: true
    project: "functiongemma-hr-routing"
    entity: "your-team"  # 可选
```

#### 登录 WandB

```bash
wandb login
```

#### 查看训练指标

访问 https://wandb.ai 查看:
- Training Loss
- Learning Rate
- Epoch 进度
- 生成的测试样本

### 本地日志

```bash
# 实时查看日志
tail -f outputs/logs/exp_hr_routing_*.log

# 查看训练指标
cat outputs/models/hr_v1/trainer_state.json | jq '.log_history'
```

## 5. 训练指标解读

### 正常训练曲线

```
Epoch 1/3: Loss 2.5 → 1.8  (快速下降)
Epoch 2/3: Loss 1.8 → 0.9  (稳定下降)
Epoch 3/3: Loss 0.9 → 0.6  (收敛)
```

### 异常情况及处理

#### Loss 不下降

**可能原因**:
- 学习率太低
- 数据格式错误
- 模型未正确加载

**解决方案**:
```yaml
training:
  learning_rate: 5.0e-4  # 提高学习率
```

#### Loss 震荡

**可能原因**:
- 学习率太高
- 批次太小

**解决方案**:
```yaml
training:
  learning_rate: 1.0e-4  # 降低学习率
  per_device_train_batch_size: 8  # 增大批次
```

#### 过拟合

**可能原因**:
- 训练轮数太多
- 数据量太少

**解决方案**:
```yaml
training:
  epochs: 2  # 减少轮数
  early_stopping: true  # 启用早停
  early_stopping_patience: 3
```

## 6. 模型评估

### 推理测试

```bash
python main.py inference \
  outputs/models/hr_v1 \
  --prompt "查询张三的年假"
```

### 批量测试

创建测试集 `data/test.jsonl`:

```json
{"user_content": "查询张三的年假", "expected_tool": "get_leave_balance"}
{"user_content": "李四请病假", "expected_tool": "request_leave"}
```

编写评估脚本:

```python
from src.training.trainer import FunctionGemmaTrainer
from src.utils.config_loader import load_config

config = load_config()
trainer = FunctionGemmaTrainer(config)
trainer.load_model()

# 测试
test_cases = [...]
correct = 0

for case in test_cases:
    result = trainer.inference(case["user_content"])
    if case["expected_tool"] in result:
        correct += 1

accuracy = correct / len(test_cases)
print(f"准确率：{accuracy:.2%}")
```

## 7. 保存和导出

### 保存检查点

训练会自动保存:
```
outputs/models/hr_v1/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── trainer_state.json
```

### 导出为 GGUF

```bash
python main.py export \
  outputs/models/hr_v1 \
  outputs/models/hr_gguf \
  --format gguf \
  --quantization q8_0
```

### 导出格式选择

| 格式 | 大小 | 速度 | 用途 |
|------|------|------|------|
| PyTorch | ~540MB | 最快 | 开发测试 |
| GGUF q8_0 | ~280MB | 快 | 生产部署 |
| GGUF q4_k_m | ~160MB | 中等 | 边缘设备 |
| GGUF q4_0 | ~150MB | 中等 | 资源受限 |

## 8. 最佳实践

### ✅ 推荐做法

1. **从小数据集开始** - 先用 50-100 样本验证流程
2. **监控训练指标** - 使用 WandB 实时查看
3. **保存多个版本** - 每次实验保存不同目录
4. **记录超参数** - 使用实验配置管理
5. **定期评估** - 每个 epoch 测试推理效果

### ❌ 避免的错误

1. **数据量太少** - 少于 50 样本/函数
2. **学习率过高** - 导致训练不稳定
3. **不验证数据** - 格式错误导致训练失败
4. **忽略显存** - OOM 导致训练中断
5. **不过滤异常值** - 错误数据影响模型质量

## 9. 故障排除

### CUDA Out of Memory

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
model:
  lora:
    use_gradient_checkpointing: true
```

### 训练很慢

- 使用 GPU (推荐 T4/L4/A10)
- 减少 `max_seq_length`
- 使用 `bfloat16` 代替 `float32`

### 准确率低

1. 增加训练数据 (200-500 样本/函数)
2. 增加训练轮数 (5-10 epochs)
3. 调整学习率 (1.0e-4 ~ 5.0e-4)
4. 检查数据质量

## 10. 实验管理

### 创建实验配置

`configs/experiments/exp_hr_v2.yaml`:

```yaml
training:
  epochs: 5
  learning_rate: 1.5e-4

model:
  lora:
    rank: 32
    alpha: 32

logging:
  wandb:
    name: "hr-routing-v2"
```

### 运行实验

```bash
python main.py train \
  --data data/processed/hr.jsonl \
  --experiment exp_hr_v2 \
  --output outputs/models/hr_v2
```

### 比较实验

在 WandB 中对比不同实验的 Loss 曲线和准确率。
