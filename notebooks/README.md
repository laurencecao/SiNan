# 📓 Jupyter Notebooks

本目录包含用于交互式训练和可视化的 Jupyter Notebooks。

## 🚀 training.ipynb

FunctionGemma 模型的交互式训练 Notebook，提供可视化的微调环境。

### 功能特性

- **📊 实时训练可视化**
  - Loss 曲线动态绘制
  - 学习率变化监控
  - 梯度范数追踪
  - Epoch 进度显示

- **🎛️ 交互式参数配置**
  - 通过 ipywidgets 滑动条和输入框配置训练参数
  - 实时调整 LoRA 参数 (rank, alpha)
  - 动态设置 batch size、学习率等

- **📈 数据可视化**
  - 文本长度分布直方图
  - 工具类别分布统计
  - 样本预览与质量检查

- **🤖 训练功能**
  - 集成 FunctionGemmaTrainer
  - 支持自定义回调
  - 实时保存训练指标

- **🧪 推理测试**
  - 交互式单条推理
  - 批量推理测试
  - 结果可视化展示

- **💾 模型导出**
  - 支持 PyTorch 格式导出
  - 支持 GGUF 量化导出
  - 交互式导出界面

### 使用方法

1. **启动 Jupyter**
   ```bash
   cd /workspace/repos/SiNan
   jupyter notebook notebooks/
   ```

2. **打开 training.ipynb**

3. **按顺序运行 Cell**
   - 环境初始化
   - 导入依赖
   - 配置参数（使用交互式控件）
   - 加载数据
   - 开始训练
   - 推理测试

### 注意事项

- 确保已安装所有依赖：`pip install -r requirements.txt`
- 训练需要 GPU，如果没有 GPU 训练会非常慢
- 建议在训练前检查 GPU 显存是否充足
- 默认使用 `google/functiongemma-270m-it` 模型

### 数据准备

如果没有现有数据，Notebook 会自动创建示例数据。你也可以：

1. 使用 Excel/CSV 准备数据
2. 使用 `python main.py convert` 转换为 JSONL
3. 在 Notebook 中指定数据路径

### 可视化示例

训练过程中会实时显示：
- 训练 Loss 曲线
- 学习率变化
- 梯度范数
- 当前 Epoch

训练结束后保存为 `training_metrics.png`。
