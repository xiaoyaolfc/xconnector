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
# 进入部署目录
cd deployments

# 执行完整构建流程
./build-wsl.sh all

# 或者分步执行：
# ./build-wsl.sh check     # 先检查环境
# ./build-wsl.sh build     # 构建镜像
# ./build-wsl.sh package   # 创建部署包
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

### 7. 传输到服务器

现在你有两种方式传输到服务器：

#### 方式A：使用提供的传输脚本
```bash
# 进入部署包目录
cd xconnector-deployment-*/

# 传输到服务器
./transfer-to-server.sh user@your-server:/path/to/deploy/

# 例如：
./transfer-to-server.sh ubuntu@192.168.1.100:/home/ubuntu/xconnector/
```

#### 方式B：手动传输
```bash
# 使用 scp 传输整个部署包
scp -r xconnector-deployment-* user@your-server:/path/to/deploy/

# 或者先压缩再传输
tar -czf xconnector-deployment.tar.gz xconnector-deployment-*/
scp xconnector-deployment.tar.gz user@your-server:/path/to/deploy/
```

### 8. 在服务器上部署

SSH 到你的服务器：

```bash
ssh user@your-server
cd /path/to/deploy/xconnector-deployment-*/

# 设置你的 AI-Dynamo 镜像名称
export DYNAMO_IMAGE=your-ai-dynamo-image:tag

# 执行部署
./deploy-server.sh
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

## 🎯 **现在就可以开始了！**

你只需要：

1. 在 WSL 中进入你的 xconnector 项目目录
2. 创建并运行 `deployments/build-wsl.sh` 脚本
3. 等待构建完成
4. 传输部署包到服务器
5. 在服务器上运行部署脚本

如果过程中遇到任何问题，告诉我具体的错误信息，我会帮你解决！