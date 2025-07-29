## 🛠️ **第二步：在 WSL 中执行构建**

### 1. 保存构建脚本

在你的 WSL 终端中，进入 xconnector 项目目录：

```bash
# 进入你的项目目录
cd /path/to/your/xconnector

# 创建构建脚本
nano deployments/build-wsl.sh

# 或者使用 vim
vim deployments/build-wsl.sh
```

将我刚才提供的 WSL 构建脚本内容复制粘贴进去并保存。

### 2. 给脚本执行权限

```bash
chmod +x deployments/build-wsl.sh
```

### 3. 验证项目结构

确保你的项目结构如下：

```bash
# 检查项目结构
ls -la
# 应该看到:
# xconnector/
# integrations/  
# deployments/
# requirements.txt
# setup.py

# 检查关键文件
ls -la deployments/docker/
# 应该看到:
# Dockerfile.xconnector-service
# docker-compose.yml
```

### 4. 执行构建

```bash
# 在项目根目录执行
# 1. 构建 XConnector 镜像
docker build -f deployments/docker/Dockerfile.xconnector-service -t xconnector-service:latest .

# 2. 导出镜像
docker save xconnector-service:latest | gzip > xconnector-service_latest.tar.gz

# 3. 手动上传镜像
```

### 5. 构建过程中你会看到：

```bash
====================================
XConnector WSL 构建工具
====================================
✓ WSL 环境检测成功
检查 Docker 环境...
✓ Docker 服务运行中
Docker 版本: Docker version 24.0.x
检查项目结构...
项目根目录: /path/to/xconnector
✓ 项目结构检查通过
开始构建 XConnector 服务镜像...
构建上下文: /path/to/xconnector
Dockerfile: deployments/docker/Dockerfile.xconnector-service
...
✓ XConnector 服务镜像构建成功
拉取依赖镜像...
...
✓ 依赖镜像拉取完成
导出 Docker 镜像...
...
✓ 镜像导出完成
创建部署包...
...
✓ 部署包创建完成: xconnector-deployment-20241223_143022
```

### 6. 构建完成后检查结果

```bash
# 查看生成的文件
ls -la
# 应该看到:
# docker-images/                     ← 镜像文件目录
# xconnector-deployment-YYYYMMDD_HHMMSS/  ← 部署包

# 查看镜像文件
ls -lh docker-images/
# 应该看到:
# xconnector-service_latest.tar.gz
# etcd_v3.5.9.tar.gz  
# nats_2.10-alpine.tar.gz

# 查看部署包内容
ls -la xconnector-deployment-*/
# 应该看到:
# docker/              ← Docker配置
# docker-images/       ← 镜像文件
# deploy-server.sh     ← 服务器部署脚本
# transfer-to-server.sh ← 传输脚本
# README.md           ← 说明文档
```

### . 在服务器上部署

在堡垒机上

```bash
# 1. 加载 XConnector 镜像
gunzip -c xconnector-service_latest.tar.gz | docker load

# 2. 部署
cd /path/to/xconnector
chmod +x deployments/deploy-offline.sh
./deployments/deploy-offline.sh deploy

# 3. 检查状态
./deployments/deploy-offline.sh status
```

### 9. 验证部署

```bash
# 检查容器状态
docker-compose ps

# 健康检查
curl http://localhost:8081/health

# 查看日志
docker-compose logs -f xconnector-service
```
