# 常见问题 (FAQ)

## 安装和环境

### Q: 安装时遇到依赖冲突怎么办？

**A**: 使用干净的虚拟环境:

```bash
# 删除旧环境
rm -rf .venv

# 创建新环境
uv venv .venv
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 重新安装
pip install -r requirements.txt
```

### Q: 没有 GPU 可以运行吗？

**A**: 可以，但训练会很慢。推荐:
- 使用 CPU 进行小规模测试
- 云端 GPU 进行正式训练 (AutoDL/T4 约 ¥0.5/小时)

### Q: CUDA 版本不兼容怎么办？

**A**: 根据 CUDA 版本安装对应的 PyTorch:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

### Q: Excel 文件读取失败？

**A**: 检查:
1. 文件路径是否正确
2. 列名是否匹配 (User Prompt, Tool Name, Tool Args)
3. 文件是否被其他程序占用

```bash
# 使用详细输出查看错误
python main.py convert data/raw/demo.xlsx data/processed/demo.jsonl -v
```

### Q: 多少数据量合适？

**A**: 
- 最小可行：50-100 样本/函数
- 生产推荐：200-500 样本/函数
- 高质量：500+ 样本/函数

### Q: 数据格式错误如何处理？

**A**: 转换器会输出详细错误信息:

```
行 5 无效：工具名为空
行 12 无效：工具参数 JSON 无效：Expecting property name...
```

根据错误信息修正 Excel 文件。

## 训练

### Q: CUDA Out of Memory 怎么办？

**A**: 减少显存占用:

```yaml
# configs/base_config.yaml
training:
  per_device_train_batch_size: 1  # 减小批次
  gradient_accumulation_steps: 16  # 增加累积
model:
  lora:
    use_gradient_checkpointing: true  # 激活检查点
```

### Q: 训练 Loss 不下降？

**A**: 可能原因:
1. 学习率太低 → 提高到 5.0e-4
2. 数据格式错误 → 检查 Token 格式
3. 训练轮数太少 → 增加到 5-10 epochs

### Q: 训练很慢怎么办？

**A**: 优化建议:
1. 使用 GPU (T4/L4/A10)
2. 使用 `bfloat16` 精度
3. 减少 `max_seq_length` 到 1024
4. 使用 Unsloth (已集成)

### Q: 如何监控训练进度？

**A**: 使用 WandB:

```bash
# 登录
wandb login

# 训练 (自动记录)
python main.py train --data ... --experiment ...

# 访问 https://wandb.ai 查看
```

## 推理和部署

### Q: 推理结果不正确？

**A**: 检查:
1. 训练数据质量
2. 推理参数 (temperature, top_k, top_p)
3. 输入提示格式

```bash
# 调整推理参数
python main.py inference model_path \
  --prompt "测试" \
  --max-tokens 256
```

### Q: GGUF 导出失败？

**A**: 确保:
1. 模型训练完成
2. 磁盘空间充足
3. 重新合并 LoRA:

```bash
python main.py export model_path output_gguf \
  --format gguf \
  --merge-lora
```

### Q: Ollama 导入失败？

**A**: 检查 Modelfile 路径:

```bash
# 查看 GGUF 文件
ls outputs/models/hr_gguf/*.gguf

# 修改 Modelfile 中的绝对路径
# 重新导入
ollama create functiongemma-hr -f Modelfile
```

### Q: 部署后响应慢？

**A**: 优化建议:
1. 使用 GPU 推理
2. 减少量化等级 (q8_0 → f16)
3. 使用批处理
4. 增加并发

## 性能和成本

### Q: 训练需要多长时间？

**A**: 参考时间 (FunctionGemma 270M):

| 数据量 | GPU | 时间 |
|--------|-----|------|
| 500 样本 | T4 | ~10 分钟 |
| 2000 样本 | T4 | ~30 分钟 |
| 10000 样本 | A10 | ~1 小时 |

### Q: 云端训练成本多少？

**A**: 参考价格:

| 云服务 | 实例 | 价格 | 1 小时成本 |
|--------|------|------|-----------|
| AutoDL | T4 | ¥0.5/小时 | ¥0.5 |
| AWS | g4dn.xlarge | $0.526/小时 | ¥3.8 |
| Google Cloud | n1-standard-4+T4 | $0.35/小时 | ¥2.5 |

### Q: 推理延迟多少？

**A**: 参考延迟 (FunctionGemma 270M):

| 部署方式 | 硬件 | 延迟 |
|----------|------|------|
| Ollama | CPU | ~500ms |
| Ollama | GPU | ~100ms |
| llama.cpp | CPU | ~300ms |
| llama.cpp | GPU | ~50ms |

## 最佳实践

### Q: 如何提高准确率？

**A**: 
1. **增加数据量** - 200-500 样本/函数
2. **数据多样性** - 同一函数的多种问法
3. **调整超参数** - epochs: 5-10, lr: 1.0e-4
4. **数据质量** - 确保格式正确

### Q: 如何管理多个实验？

**A**: 使用实验配置:

```bash
# 创建不同实验配置
configs/experiments/exp_v1.yaml
configs/experiments/exp_v2.yaml

# 分别训练
python main.py train --experiment exp_v1 --output models/v1
python main.py train --experiment exp_v2 --output models/v2

# WandB 对比
```

### Q: 如何版本控制模型？

**A**: 
1. 使用 Git 管理代码和配置
2. 使用 WandB 管理训练记录
3. 使用 DVC 管理大文件 (可选)
4. 模型文件标注版本和日期

## 故障排除

### Q: 找不到模块错误？

```bash
ModuleNotFoundError: No module named 'xxx'
```

**A**: 
```bash
# 确保激活虚拟环境
source .venv/bin/activate

# 重新安装
pip install -r requirements.txt
```

### Q: 导入错误？

```bash
ImportError: libcuda.so.1: cannot open shared object file
```

**A**: 
```bash
# 安装 CUDA 驱动
# 或无 GPU 环境使用 CPU 模式
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: 训练中断后如何继续？

**A**: 
```bash
# 从最后一个检查点继续
python main.py train \
  --data ... \
  --output outputs/models/hr_v1 \
  # 会自动加载检查点
```

## 获取帮助

### 有用的链接

- [完整文档](./README.md)
- [快速开始](./getting-started.md)
- [训练指南](./training-guide.md)
- [部署指南](./deployment.md)

### 社区支持

- [GitHub Issues](https://github.com/your-org/SiNan/issues)
- [讨论区](https://github.com/your-org/SiNan/discussions)

### 联系方式

- Email: support@example.com
- 微信：SiNan-Support
