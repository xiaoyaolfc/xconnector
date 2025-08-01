#!/bin/bash
# 部署前配置验证脚本

echo "=== XConnector 部署前配置检查 ==="

# 检查必要文件
echo "检查配置文件..."
files=(
    "deployments/docker/docker-compose.yml"
    "integrations/dynamo/configs/disagg_with_xconnector.yaml"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file"
    else
        echo "✗ 缺少文件: $file"
        exit 1
    fi
done

# 检查 Docker Compose 配置
echo "验证 Docker Compose 配置..."
if docker-compose -f deployments/docker/docker-compose.yml config > /dev/null; then
    echo "✓ Docker Compose 配置有效"
else
    echo "✗ Docker Compose 配置错误"
    exit 1
fi

# 检查服务地址配置
echo "检查服务地址配置..."
config_file="integrations/dynamo/configs/disagg_with_xconnector.yaml"

if grep -q "etcd:2379" "$config_file" && grep -q "nats:4222" "$config_file"; then
    echo "✓ 服务地址配置正确"
else
    echo "✗ 服务地址配置错误"
    echo "请确保配置文件包含："
    echo "  - etcd-url: \"http://etcd:2379\""
    echo "  - nats-url: \"nats://nats:4222\""
    exit 1
fi

echo "✅ 配置验证通过，可以开始部署"