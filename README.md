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