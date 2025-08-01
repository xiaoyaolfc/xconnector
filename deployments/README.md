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

# 1. 确保有相同的文件
```shell
deployments/docker/docker-compose.yml
integrations/dynamo/configs/disagg_with_xconnector.yaml

```

# 2. 部署命令
```shell
./deployments/pre-deployment-check.sh
docker-compose -f deployments/docker/docker-compose.yml down
docker-compose -f deployments/docker/docker-compose.yml up -d
./deployments/final-fix-verification.sh verify
```

# 3. 等待并检查
```shell
sleep 90
curl http://localhost:8000/health
```




