# XConnector-Dynamo 部署文件夹

该文件夹存放部署xconnector和ai-dynamo的相关配置启动文件

## setup

1. Set environment variables:
```bash
source scripts/setup_environment.sh
```
2. Install XConnector (if not already installed):
```bash
python deployments/scripts/deploy_xconnector.py --install 
```
## 部署步骤

1. 设置权限
```bash
chmod +x deployments/docker/start-xconnector.sh
```
2. 设置 Dynamo 镜像（如果使用私有镜像）
```bash
export DYNAMO_IMAGE="your-registry/ai-dynamo:latest"
```
3. 启动所有服务
```bash
./deployments/docker/start-xconnector.sh
```
4. 查看日志
```bash
docker-compose -f deployments/docker/docker-compose.yml logs -f
```
5. curl测试
```base
curl http://localhost:8081/health
curl http://localhost:8000/v1/models
```





