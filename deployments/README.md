# XConnector + Dynamo 完整部署指南

## 📋 概述

本指南详细说明如何在服务器上部署集成了 XConnector 的 Dynamo 推理服务，包括模型准备、环境配置、服务启动等完整流程。

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   XConnector    │◄──►│     Dynamo      │◄──►│     vLLM        │
│   (KV Cache)    │    │   (Routing)     │    │  (Inference)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      etcd       │    │      NATS       │    │    GPU Memory   │
│ (Service Disc.) │    │  (Messaging)    │    │   Management    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 部署流程

### 1. 环境准备

#### 1.1 系统要求

```bash
# 硬件要求
- GPU: NVIDIA A100/H100 或同等性能 GPU
- 内存: 至少 128GB 系统内存
- 存储: 至少 500GB 可用空间
- 网络: 高速网络连接

# 软件要求
- Ubuntu 20.04/22.04
- Docker 和 Docker Compose
- NVIDIA Container Toolkit
- Python 3.8+
```

#### 1.2 依赖安装

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 2. 模型准备

#### 2.1 模型选择和下载

```bash
# 创建模型目录
sudo mkdir -p /data/model
sudo chown -R $USER:$USER /data/model

# 选择合适大小的模型（推荐）
# 对于 80GB GPU，建议使用：
# - 7B-13B 模型：完整功能
# - 30B 模型：基础功能  
# - 70B 模型：需要内存优化

# 示例：下载 DeepSeek 模型
cd /data/model
# 使用 huggingface-cli, git lfs, 或其他方式下载模型
# 确保模型完整且可访问
```

#### 2.2 模型验证

```bash
# 检查模型文件完整性
ls -la /data/model/your-model-name/
# 应该包含：
# - config.json
# - tokenizer.json
# - *.safetensors 或 *.bin 文件
# - tokenizer_config.json

# 检查磁盘空间
df -h /data/model
```

### 3. XConnector 代码部署

#### 3.1 下载 XConnector 源码

```bash
# 创建工作目录
mkdir -p /home/lfc
cd /home/lfc

# 从 GitHub 克隆 XConnector 代码
git clone https://github.com/xiaoyaolfc/xconnector
# 或者上传预打包的代码

# 设置权限
chmod -R 755 /home/lfc/xconnector
```

#### 3.2 部署初始化脚本

```bash
cd /home/lfc/xconnector

# 上传 init-xconnector.sh 脚本
# 路径：/deployments/init-xconnector.sh -> /home/lfc/xconnector/
cp deployments/init-xconnector.sh ./init-xconnector.sh
chmod +x init-xconnector.sh

# 验证脚本
./init-xconnector.sh --help  # 检查脚本是否可执行
```

#### 3.3 创建配置目录和文件

```bash
# 创建配置目录
mkdir -p /home/lfc/xconnector-configs

# 上传预配置的配置文件
# 从 integrations/dynamo/configs/ 复制所需配置
cp -r /home/lfc/xconnector/integrations/dynamo/configs/* /home/lfc/xconnector-configs/

# 创建主配置文件
cat > /home/lfc/xconnector-configs/dynamo-xconnector.yaml << 'EOF'
xconnector:
  enabled: true
  mode: "embedded"
  offline_mode: true
  
  # 服务发现配置（如果需要）
  etcd:
    enabled: true
    endpoints: ["http://127.0.0.1:2379"]
    timeout: 5
  
  nats:
    enabled: true  
    url: "nats://127.0.0.1:4222"
    timeout: 5
  
  # 适配器配置
  adapters:
    - name: "lmcache"
      type: "cache"
      class_path: "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter"
      config:
        storage_backend: "memory"
        max_cache_size: 2048
        enable_compression: false
      enabled: true
  
  # 日志配置
  logging:
    level: "INFO"
    console: true
    file: "/workspace/logs/xconnector.log"
    
  # 故障处理
  fault_tolerance:
    graceful_degradation: true
    fail_on_error: false
    offline_fallback: true
    retry_attempts: 3
    retry_delay: 1.0
    
  # 性能配置
  performance:
    async_workers: 2
    queue_size: 1000
    timeout: 30
EOF
```

### 4. 基础服务启动

#### 4.1 启动 etcd 和 NATS

```bash
cd /home/lfc/xconnector

# 启动基础服务（etcd + NATS）
docker compose -f deploy/metrics/docker-compose.yml up -d

# 等待服务启动
sleep 10

# 验证服务状态
docker compose -f deploy/metrics/docker-compose.yml ps
```

#### 4.2 配置 etcd 认证（可选）

```bash
# 进入 etcd 容器
docker exec -it $(docker compose -f deploy/metrics/docker-compose.yml ps -q etcd-server) bash

# 在容器内配置认证
export ETCDCTL_ENDPOINTS=http://192.168.2.58:2379

# 创建用户和角色
etcdctl user add root --new-user-password='7#k9$ZPmqB2@xY*'
etcdctl role add root
etcdctl user grant-role root root
etcdctl auth enable

# 验证认证
etcdctl --user=root:'7#k9$ZPmqB2@xY*' endpoint health

# 退出容器
exit
```

#### 4.3 验证基础服务

```bash
# 测试 etcd 连接
docker exec $(docker compose -f deploy/metrics/docker-compose.yml ps -q etcd-server) \
    etcdctl endpoint health

# 测试 NATS 连接  
docker exec $(docker compose -f deploy/metrics/docker-compose.yml ps -q nats-server) \
    nats server info
```

### 5. Dynamo 容器启动

#### 5.1 创建 Dynamo 配置

```bash
# 根据模型大小选择合适的配置模板
cd /home/lfc/xconnector-configs

# 对于小模型（7B-13B）- 推荐配置
cat > agg_with_xconnector.yaml << 'EOF'
Common:
  model: /data/model/your-model-name
  block-size: 64
  max-model-len: 16384

Frontend:
  served_model_name: your-model-name
  endpoint: dynamo.Processor.chat/completions
  port: 8000

Processor:
  router: round-robin
  router-num-threads: 4
  common-configs: [model, block-size, max-model-len]

VllmWorker:
  enforce-eager: true
  max-num-batched-tokens: 16384
  enable-prefix-caching: true
  gpu-memory-utilization: 0.90
  ServiceArgs:
    workers: 1
    resources:
      gpu: '8'
  common-configs: [model, block-size, max-model-len]

Planner:
  environment: local
  no-operation: true
EOF

# 对于大模型（30B+）- 内存优化配置
cat > agg_with_xconnector_large.yaml << 'EOF'
Common:
  model: /data/model/your-large-model-name
  block-size: 32
  max-model-len: 8192

Frontend:
  served_model_name: your-large-model-name
  endpoint: dynamo.Processor.chat/completions
  port: 8000

Processor:
  router: round-robin
  router-num-threads: 2
  common-configs: [model, block-size, max-model-len]

VllmWorker:
  enforce-eager: true
  max-num-batched-tokens: 8192
  enable-prefix-caching: false
  gpu-memory-utilization: 0.80
  max-num-seqs: 4
  ServiceArgs:
    workers: 1
    resources:
      gpu: '8'
  common-configs: [model, block-size, max-model-len]

Planner:
  environment: local
  no-operation: true
EOF
```

#### 5.2 启动 Dynamo 容器

```bash
cd /home/lfc/xconnector

# 启动带 XConnector 集成的 Dynamo 容器
./container/run.sh --framework vllm --mount-workspace --xconnector-enabled --xconnector-path /home/lfc/xconnector -it

# 容器启动后进入交互模式
```

### 6. 容器内环境配置

#### 6.1 设置环境变量

```bash
# 在容器内执行
export XCONNECTOR_CONFIG_FILE=/workspace/configs/dynamo-xconnector.yaml
export XCONNECTOR_ENABLED=true
export PYTHONPATH="/workspace/xconnector:/workspace:$PYTHONPATH"

# 验证环境变量
echo "XConnector 配置文件: $XCONNECTOR_CONFIG_FILE"
echo "Python 路径: $PYTHONPATH"
```

#### 6.2 初始化 XConnector

```bash
# 运行 XConnector 初始化脚本
source /workspace/xconnector/init-xconnector.sh

# 检查初始化结果
# 应该看到：
# ✅ 必需依赖检查通过
# ✅ XConnector autopatch 导入成功  
# ✅ 集成状态: sdk_available=True, sdk_ready=True
# ✅ XConnector 离线初始化完成！
```

#### 6.3 验证集成状态

```bash
# 测试 XConnector 配置检测
python3 -c "
import sys
sys.path.insert(0, '/workspace/xconnector')
from integrations.dynamo.config_detector import detect_xconnector_config
config = detect_xconnector_config()
print('✅ 配置检测成功' if config else '❌ 配置检测失败')
"

# 测试集成状态
python3 -c "
import sys
sys.path.insert(0, '/workspace/xconnector')
from integrations.dynamo.autopatch import get_integration_status
status = get_integration_status()
print('集成状态:', status)
"
```

### 7. 启动推理服务

#### 7.1 启动 Dynamo 服务

```bash
# 进入 Dynamo 示例目录
cd $DYNAMO_HOME/examples/llm

# 根据模型大小选择合适的配置文件启动
# 小模型
dynamo serve graphs.agg:Frontend -f /workspace/configs/agg_with_xconnector.yaml

# 大模型
dynamo serve graphs.agg:Frontend -f /workspace/configs/agg_with_xconnector_large.yaml
```

#### 7.2 验证服务启动

```bash
# 在另一个终端检查服务状态
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# 检查日志
tail -f /workspace/logs/xconnector.log
```

### 8. 服务测试

#### 8.1 基础功能测试

```bash
# 测试推理接口
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "your-model-name",
       "messages": [
         {"role": "user", "content": "Hello, how are you?"}
       ],
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

#### 8.2 XConnector 功能测试

```bash
# 测试缓存功能（发送相同请求）
for i in {1..3}; do
  echo "请求 $i:"
  curl -X POST "http://localhost:8000/v1/chat/completions" \
       -H "Content-Type: application/json" \
       -d '{
         "model": "your-model-name", 
         "messages": [{"role": "user", "content": "What is AI?"}],
         "max_tokens": 50
       }' | jq '.choices[0].message.content'
  echo "---"
done
```

## 🔧 故障排查

### 常见问题及解决方案

#### 1. 内存不足错误

```bash
# 错误：No available memory for the cache blocks
# 解决：调整配置参数

# 查看 GPU 内存使用
nvidia-smi

# 调整配置文件中的参数：
# - gpu-memory-utilization: 0.80 -> 0.70
# - max-model-len: 16384 -> 8192  
# - max-num-batched-tokens: 16384 -> 8192
# - enable-prefix-caching: true -> false
```

#### 2. XConnector 初始化失败

```bash
# 错误：sdk_available=False
# 检查配置文件路径
ls -la /workspace/configs/dynamo-xconnector.yaml

# 检查环境变量
echo $XCONNECTOR_CONFIG_FILE

# 重新设置环境变量
export XCONNECTOR_CONFIG_FILE=/workspace/configs/dynamo-xconnector.yaml
source /workspace/xconnector/init-xconnector.sh
```

#### 3. 服务连接失败

```bash
# 检查 etcd 和 NATS 服务状态
docker compose -f deploy/metrics/docker-compose.yml ps

# 重启服务
docker compose -f deploy/metrics/docker-compose.yml restart

# 检查端口占用
netstat -tulpn | grep -E "2379|4222|8000"
```

### 日志查看

```bash
# XConnector 日志
tail -f /workspace/logs/xconnector.log

# Dynamo 日志
# 查看容器内的标准输出

# 基础服务日志
docker compose -f deploy/metrics/docker-compose.yml logs etcd-server
docker compose -f deploy/metrics/docker-compose.yml logs nats-server
```

## 📈 性能优化

### 1. 内存优化

```yaml
# 针对不同模型大小的优化建议
# 7B-13B 模型：
gpu-memory-utilization: 0.90
max-model-len: 16384
enable-prefix-caching: true

# 30B-40B 模型：  
gpu-memory-utilization: 0.85
max-model-len: 8192
enable-prefix-caching: true

# 70B+ 模型：
gpu-memory-utilization: 0.75
max-model-len: 4096
enable-prefix-caching: false
```

### 2. 缓存优化

```yaml
# XConnector 缓存配置优化
adapters:
  - name: "lmcache"
    config:
      storage_backend: "memory"
      max_cache_size: 4096  # 根据可用内存调整
      enable_compression: true  # 启用压缩节省内存
      ttl_seconds: 3600    # 设置缓存过期时间
```

### 3. 并发优化

```yaml
# Dynamo 并发配置
VllmWorker:
  max-num-seqs: 8        # 根据 GPU 内存调整
  max-num-batched-tokens: 16384  # 批处理大小
  
Processor:
  router-num-threads: 4   # 路由线程数
```

## 🚀 生产环境部署建议

### 1. 监控和告警

```bash
# 部署监控服务
docker compose -f deploy/metrics/docker-compose.yml up -d prometheus grafana

# 配置 GPU 监控
# 配置内存使用监控  
# 配置服务健康检查
```

### 2. 高可用配置

```bash
# 多实例部署
# 负载均衡配置
# 故障转移机制
```

### 3. 安全配置

```bash
# API 认证配置
# 网络安全配置
# 访问控制配置
```

## 📝 备注

- 根据实际硬件配置调整内存和并发参数
- 建议在测试环境充分验证后再部署到生产环境
- 定期备份配置文件和模型数据
- 监控服务性能指标，及时优化配置

## 🔗 相关链接

- [XConnector GitHub 仓库](https://github.com/your-org/xconnector)
- [Dynamo 官方文档](https://dynamo.docs)
- [vLLM 配置指南](https://vllm.readthedocs.io)
- [故障排查手册](./troubleshooting.md)