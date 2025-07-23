# xconnector Project

## WSL构建项目

- 如果是首次构建，需要创建新的虚拟环境，并安装依赖

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- 如果太卡了，可以考虑换源

```shell
mkdir -p ~/.pip
vim ~/.pip/pip.conf

# 添加如下内容
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
timeout = 120
```

# XConnector + AI-Dynamo 离线部署指南

## 1. 准备工作

### 1.1 环境要求
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime（如需GPU支持）
- 至少8GB可用内存
- 至少20GB磁盘空间

### 1.2 获取Dynamo镜像
```bash
# 方式1：从你的私有仓库拉取
docker pull your-registry/ai-dynamo:latest

# 方式2：如果有Dynamo源码，自行构建
cd /path/to/ai-dynamo
docker build -t ai-dynamo:latest .

# 设置环境变量
export DYNAMO_IMAGE=ai-dynamo:latest  # 或 your-registry/ai-dynamo:latest
```

## 2. 构建和导出镜像（在有网络的机器上）

```bash
cd xconnector/deployments

# 设置Dynamo镜像
export DYNAMO_IMAGE=your-registry/ai-dynamo:latest

# 构建所有镜像并导出
./offline-deploy.sh build

# 创建完整部署包
./offline-deploy.sh package
```

这会生成一个包含所有镜像和配置的部署包：`xconnector-dynamo-deployment-YYYYMMDD_HHMMSS.tar.gz`

## 3. 离线服务器部署

### 3.1 传输部署包
```bash
# 将部署包传输到离线服务器
scp xconnector-dynamo-deployment-*.tar.gz user@offline-server:/path/to/deploy/

# 在离线服务器上解压
cd /path/to/deploy/
tar -xzf xconnector-dynamo-deployment-*.tar.gz
cd xconnector-dynamo-deployment-*/
```

### 3.2 加载镜像
```bash
# 加载所有Docker镜像
./offline-deploy.sh load
```

### 3.3 配置调整
根据实际环境调整配置文件：

#### `docker/configs/disagg_xconnector_remote.yaml`
```yaml
# 调整模型路径和GPU配置
Common:
  model: /path/to/your/model  # 修改为实际模型路径
  
VllmWorker:
  ServiceArgs:
    workers: 1
    resources:
      gpu: 1  # 根据实际GPU数量调整
      
  extensions:
    xconnector:
      enabled: true
      service_mode: remote
      service_url: http://xconnector-service:8081
```

#### `docker/configs/xconnector_config.yaml`
```yaml
# 根据需要调整适配器配置
adapters:
  vllm:
    enabled: true
    config:
      model_name: /path/to/your/model  # 修改为实际模型路径
      tensor_parallel_size: 1  # 根据GPU数量调整
```

### 3.4 部署服务
```bash
# 部署所有服务
./offline-deploy.sh deploy
```

## 4. 验证部署

### 4.1 检查服务状态
```bash
cd docker
docker-compose ps
```

### 4.2 健康检查
```bash
# XConnector服务
curl http://localhost:8081/health

# etcd
curl http://localhost:2379/health

# NATS监控
curl http://localhost:8222/
```

### 4.3 测试推理
```bash
# 测试Dynamo API
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'
```

## 5. 目录结构和挂载说明

### 5.1 关键目录挂载
```yaml
volumes:
  # XConnector集成代码
  - ../../integrations/dynamo/extension_loader.py:/xconnector-integration/extension_loader.py:ro
  
  # 启动包装脚本
  - ./dynamo-wrapper/startup-wrapper.py:/workspace/startup-wrapper.py:ro
  
  # 配置文件
  - ./configs/disagg_xconnector_remote.yaml:/workspace/configs/disagg.yaml:ro
  
  # 模型缓存（持久化）
  - model-cache:/workspace/models
  
  # 日志目录（持久化）
  - dynamo-logs:/workspace/logs
  - xconnector-logs:/app/logs
```

### 5.2 代码集成方式
XConnector通过以下方式与Dynamo集成：

1. **启动包装脚本**：`startup-wrapper.py`
   - 在Dynamo启动前注入XConnector扩展
   - 自动发现并patch VllmWorker类
   - 支持远程和嵌入式两种模式

2. **扩展加载器**：`extension_loader.py`
   - 提供XConnector功能的接口封装
   - 处理KV缓存的收发
   - 管理与XConnector服务的通信

3. **配置集成**：
   - 通过Dynamo的配置文件启用XConnector
   - 支持动态配置和热重载

## 6. 故障排除

### 6.1 常见问题

#### 服务启动失败
```bash
# 查看详细日志
docker-compose logs -f xconnector-service
docker-compose logs -f dynamo-worker

# 检查配置文件
docker exec -it dynamo-worker cat /workspace/configs/disagg.yaml
```

#### XConnector连接失败
```bash
# 测试XConnector服务连接
docker exec -it dynamo-worker curl http://xconnector-service:8081/health

# 检查网络连接
docker network ls
docker network inspect dynamo-network
```

#### GPU资源问题
```bash
# 检查GPU可用性
docker exec -it dynamo-worker nvidia-smi

# 检查容器GPU访问权限
docker exec -it dynamo-worker python -c "import torch; print(torch.cuda.is_available())"
```

### 6.2 日志分析
```bash
# XConnector服务日志
docker-compose logs -f xconnector-service | grep ERROR

# Dynamo worker日志
docker-compose logs -f dynamo-worker | grep -E "(XConnector|ERROR)"

# 系统资源监控
docker stats
```

### 6.3 配置重载
```bash
# 重启特定服务
docker-compose restart xconnector-service
docker-compose restart dynamo-worker

# 重新加载配置
docker-compose down
docker-compose up -d
```

## 7. 性能优化建议

### 7.1 资源配置
```yaml
# 根据实际硬件调整资源限制
services:
  dynamo-worker:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 2  # 使用多GPU
```

### 7.2 缓存配置
```yaml
# 优化缓存设置
xconnector:
  adapters:
    lmcache:
      config:
        max_cache_size: 4096  # 增大缓存
        enable_compression: true
```

### 7.3 网络优化
```yaml
# 使用主机网络模式（可选）
services:
  dynamo-worker:
    network_mode: "host"  # 降低网络延迟
```

## 8. 监控和维护

### 8.1 监控指标
- XConnector API响应时间和成功率
- 缓存命中率
- GPU利用率和内存使用
- 网络延迟和吞吐量

### 8.2 日常维护
```bash
# 清理旧容器和镜像
docker system prune -f

# 备份配置文件
cp -r docker/configs/ backup/configs-$(date +%Y%m%d)/

# 更新镜像
docker-compose pull
docker-compose up -d
```