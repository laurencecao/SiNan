#!/bin/bash
# FunctionGemma 云端一键训练脚本
# 后台运行训练任务，防止 SSH 断开导致中断

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "FunctionGemma 云端训练"
echo "=========================================="

# 获取实验名称
EXP_NAME="${1:-default_run}"
shift || true

echo "实验名称：$EXP_NAME"

# 激活 Conda 环境
ENV_NAME="function_gemma_env"
echo "激活环境：$ENV_NAME"

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "错误：未检测到 Conda"
    exit 1
fi

# 创建日志目录
LOG_DIR="$PROJECT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

echo "日志文件：$LOG_FILE"
echo ""

# 构建训练命令
TRAIN_CMD="python $PROJECT_DIR/main.py train --experiment $EXP_NAME"

# 添加额外参数
if [[ $# -gt 0 ]]; then
    TRAIN_CMD="$TRAIN_CMD $@"
fi

echo "训练命令：$TRAIN_CMD"
echo ""

# 后台运行
echo "开始训练 (后台运行)..."
nohup bash -c "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
PID=$!

echo "训练进程已启动 (PID: $PID)"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看进程:"
echo "  ps aux | grep $PID"
echo ""
echo "停止训练:"
echo "  kill $PID"
echo ""

# 等待几秒检查是否正常运行
sleep 5
if ps -p $PID > /dev/null; then
    echo "✓ 训练正常运行中"
else
    echo "✗ 训练进程已退出，请检查日志"
    echo "日志内容:"
    tail -n 50 "$LOG_FILE"
    exit 1
fi
