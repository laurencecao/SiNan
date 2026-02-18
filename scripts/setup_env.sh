#!/bin/bash
# FunctionGemma-Unsloth-Enterprise-Tuner 环境初始化脚本
# 适用于云主机 (AWS EC2, Google Cloud, RunPod, AutoDL)

set -e

echo "=========================================="
echo "FunctionGemma 环境初始化"
echo "=========================================="

# 检查 CUDA 版本
echo "检查 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)
    echo "检测到 CUDA 版本：$CUDA_VERSION"
else
    echo "警告：未检测到 CUDA，将使用 CPU 模式"
    CUDA_VERSION=""
fi

# 创建 Conda 环境
ENV_NAME="function_gemma_env"
echo "创建 Conda 环境：$ENV_NAME"

if command -v conda &> /dev/null; then
    # 检查环境是否已存在
    if conda env list | grep -q "$ENV_NAME"; then
        echo "环境已存在，跳过创建"
    else
        conda create -n "$ENV_NAME" python=3.11 -y
    fi
else
    echo "错误：未检测到 Conda，请先安装 Miniconda/Anaconda"
    exit 1
fi

# 激活环境
echo "激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 安装 PyTorch (根据 CUDA 版本)
echo "安装 PyTorch..."
if [[ -n "$CUDA_VERSION" ]]; then
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "未知的 CUDA 版本，使用默认安装"
        pip install torch torchvision torchaudio
    fi
else
    echo "CPU 模式安装 PyTorch"
    pip install torch torchvision torchaudio
fi

# 安装 Unsloth
echo "安装 Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 安装其他依赖
echo "安装其他依赖..."
pip install transformers trl accelerate peft bitsandbytes
pip install pandas openpyxl polars
pip install omegaconf hydra-core
pip install wandb
pip install gguf sentencepiece
pip install tqdm rich typer python-dotenv
pip install jupyter notebook ipywidgets

# 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"

# 完成
echo ""
echo "=========================================="
echo "环境初始化完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo "  python main.py --help"
echo ""
echo "训练示例:"
echo "  python main.py train --data data/processed --experiment exp_hr_routing"
echo ""
