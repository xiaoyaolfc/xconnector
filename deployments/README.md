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

1. Full Deployment (Recommended)
```base
python deployments/scripts/deploy_xconnector.py --mode all --worker-config disagg.yaml --validate
```
2. Service-Only Deployment
```base
python deployments/scripts/deploy_xconnector.py --mode service
```
3. Workers-Only Deployment
```base
python deployments/scripts/deploy_xconnector.py --mode workers --worker-config disagg.yaml
```

## 配置文件和ai-dynamo补丁
放在了
```shell
integrations/dynamo/patches/workers
integrations/dynamo/configs/xconnector_config.yaml
```





