# 部署指南

本文档介绍如何将训练好的 FunctionGemma 模型部署到生产环境。

## 部署选项概览

| 部署方式 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **Ollama** | 本地开发/测试 | 简单易用 | 性能一般 |
| **llama.cpp** | CPU 部署 | 轻量快速 | 需要编译 |
| **vLLM** | GPU 服务 | 高性能 | 需要 GPU |
| **TFLite** | 移动设备 | 离线运行 | 需要转换 |

## 1. GGUF 导出

### 导出为 GGUF 格式

```bash
python main.py export \
  outputs/models/hr_v1 \
  outputs/models/hr_gguf \
  --format gguf \
  --quantization q8_0
```

### 量化选项

| 量化 | 大小 | 质量损失 | 推荐场景 |
|------|------|----------|----------|
| `f16` | ~540MB | 无 | 高精度需求 |
| `q8_0` | ~280MB | <1% | 生产推荐 ⭐ |
| `q4_k_m` | ~160MB | ~2% | 边缘设备 |
| `q4_0` | ~150MB | ~3% | 资源受限 |
| `q3_k_m` | ~130MB | ~5% | 极限压缩 |
| `q2_k` | ~110MB | ~8% | 不推荐 |

### 选择量化等级

```bash
# 生产环境 (推荐)
--quantization q8_0

# 边缘设备 (手机/嵌入式)
--quantization q4_k_m

# 极限压缩
--quantization q4_0
```

## 2. Ollama 部署

### 安装 Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 下载 https://ollama.com/download/OllamaSetup.exe
```

### 创建 Modelfile

在 GGUF 导出目录中会自动生成 `Modelfile`:

```
FROM /path/to/hr_v1_q8_0.gguf

# FunctionGemma 模板
TEMPLATE """
{{ if .Tools }}
<<start_of_turn>>developer
You are a model that can do function calling with the following functions
{{ range .Tools }}
<<start_function_declaration>>declaration:{{ .function.name }}{{ description:<escape>{{ .function.description }}<escape>,parameters:{{ .function.parameters }}}}<<end_function_declaration>>
{{ end }}
<<end_of_turn>>
{{ end }}
<<start_of_turn>>user
{{ .Prompt }}<<end_of_turn>>
<<start_of_turn>>model
"""

PARAMETER temperature 1.0
PARAMETER top_k 64
PARAMETER top_p 0.95
```

### 导入模型

```bash
ollama create functiongemma-hr -f outputs/models/hr_gguf/Modelfile
```

### 运行模型

```bash
# 交互式运行
ollama run functiongemma-hr "查询张三的年假"

# API 调用
curl http://localhost:11434/api/generate -d '{
  "model": "functiongemma-hr",
  "prompt": "查询张三的年假",
  "stream": false
}'
```

### 部署为服务

```bash
# systemd 服务 (Linux)
sudo systemctl enable ollama
sudo systemctl start ollama

# 验证服务
curl http://localhost:11434/api/tags
```

## 3. llama.cpp 部署

### 安装 llama.cpp

```bash
# 克隆仓库
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 编译 (CPU)
make

# 编译 (GPU - CUDA)
LLAMA_CUBLAS=1 make

# 编译 (GPU - Metal Mac)
LLAMA_METAL=1 make
```

### 运行模型

```bash
# CPU 推理
./main -m outputs/models/hr_gguf/hr_v1_q8_0.gguf \
  -p "查询张三的年假" \
  -n 128

# GPU 推理 (CUDA)
./main -m outputs/models/hr_gguf/hr_v1_q8_0.gguf \
  -p "查询张三的年假" \
  -n 128 \
  -ngl 99  # 所有层卸载到 GPU
```

### 启动服务器

```bash
# 启动 API 服务器
./server -m outputs/models/hr_gguf/hr_v1_q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048

# API 调用
curl http://localhost:8080/completion -d '{
  "prompt": "查询张三的年假",
  "n_predict": 128
}'
```

## 4. Docker 部署

### 创建 Dockerfile

```dockerfile
FROM ollama/ollama:latest

COPY outputs/models/hr_gguf/ /models/
COPY outputs/models/hr_gguf/Modelfile /models/Modelfile

RUN ollama create functiongemma-hr -f /models/Modelfile

EXPOSE 11434

CMD ["ollama", "serve"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t functiongemma-hr:latest .

# 运行容器
docker run -d \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  --name functiongemma-hr \
  functiongemma-hr:latest

# 验证
curl http://localhost:11434/api/tags
```

### Docker Compose

```yaml
version: '3.8'

services:
  functiongemma:
    image: functiongemma-hr:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

## 5. 云端部署

### AWS EC2

#### 启动实例

```bash
# 推荐实例类型
# - GPU: g4dn.xlarge (T4 16GB) - $0.526/小时
# - CPU: c6i.2xlarge - $0.34/小时

aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx
```

#### 部署脚本

```bash
#!/bin/bash
# deploy_aws.sh

# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 上传模型
scp outputs/models/hr_gguf/ ec2-user@<ip>:/home/ec2-user/

# 导入模型
ssh ec2-user@<ip> "ollama create functiongemma-hr -f /home/ec2-user/Modelfile"

# 启动服务
ssh ec2-user@<ip> "sudo systemctl start ollama"
```

### Google Cloud

```bash
# 创建 VM
gcloud compute instances create functiongemma-vm \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# SSH 连接
gcloud compute ssh functiongemma-vm

# 部署 (同上)
```

## 6. 性能优化

### 批处理

```python
# 批量推理提高吞吐量
prompts = ["查询张三的年假", "李四请病假", ...]
results = []

for prompt in prompts:
    result = model.generate(prompt)
    results.append(result)
```

### 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_inference(prompt: str) -> str:
    return model.generate(prompt)
```

### 并发

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(model.generate, prompts))
```

## 7. 监控和日志

### 健康检查

```bash
# Ollama 健康检查
curl http://localhost:11434/api/tags

# llama.cpp 健康检查
curl http://localhost:8080/health
```

### 性能监控

```python
import time
import statistics

latencies = []

for _ in range(100):
    start = time.time()
    model.generate("测试提示")
    latencies.append(time.time() - start)

print(f"P50: {statistics.median(latencies):.2f}s")
print(f"P99: {sorted(latencies)[99]:.2f}s")
```

### 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_inference(prompt: str, result: str):
    logger.info(f"Inference: {prompt[:50]}... -> {result[:50]}...")
```

## 8. 安全考虑

### API 认证

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

@app.post("/generate")
def generate(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # 处理请求
```

### 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
def generate(request: Request):
    # 处理请求
```

### 输入验证

```python
def validate_input(prompt: str) -> bool:
    # 长度检查
    if len(prompt) > 1000:
        return False
    
    # 敏感词检查
    sensitive_words = ["password", "secret", "key"]
    for word in sensitive_words:
        if word in prompt.lower():
            return False
    
    return True
```

## 9. 部署检查清单

### 部署前

- [ ] 模型已导出为 GGUF 格式
- [ ] 量化等级已选择
- [ ] Modelfile 已创建
- [ ] 测试推理效果

### 部署中

- [ ] 安装运行时 (Ollama/llama.cpp)
- [ ] 导入模型
- [ ] 启动服务
- [ ] 验证健康检查

### 部署后

- [ ] 性能测试 (延迟/吞吐量)
- [ ] 监控告警配置
- [ ] 日志收集配置
- [ ] 备份策略

## 10. 故障排除

### 模型加载失败

```bash
# 检查文件完整性
ls -lh outputs/models/hr_gguf/*.gguf

# 重新导出
python main.py export ...
```

### 推理结果异常

```bash
# 检查量化等级 (尝试更高精度)
--quantization f16

# 检查输入格式
python main.py inference ... --prompt "测试"
```

### 服务崩溃

```bash
# 查看日志
journalctl -u ollama -f

# 重启服务
sudo systemctl restart ollama
```
