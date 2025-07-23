#!/bin/bash
# 离线环境检查和准备脚本

echo "🔒 离线环境部署检查..."

cd /home/lfc/xconnector

# 1. 检查必要的镜像
echo "🐳 检查 Docker 镜像..."
echo "必需的镜像："

required_images=(
    "xconnector-service:latest"
    "dynamo:latest-vllm"
    "bitnami/etcd:auth-online"
    "nats:latest"
)

all_images_present=true

for image in "${required_images[@]}"; do
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^${image}$"; then
        echo "✅ $image"
    else
        echo "❌ $image - 缺失"
        all_images_present=false
    fi
done

if [ "$all_images_present" = false ]; then
    echo ""
    echo "❌ 缺少必要的镜像！"
    echo "请确保所有镜像都已从本地传输到服务器"
    exit 1
fi

# 2. 创建基本目录
echo ""
echo "📁 创建目录..."
mkdir -p logs

# 3. 清理可能冲突的网络
echo ""
echo "🔧 清理网络冲突..."
docker network prune -f || true

# 4. 检查端口占用
echo ""
echo "🔍 检查端口占用..."
ports=(8081 8000 2379 2380 4222 8222)

for port in "${ports[@]}"; do
    if netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "⚠️  端口 $port 已被占用"
        echo "   占用进程: $(netstat -tlnp 2>/dev/null | grep ":${port} " | awk '{print $7}')"
    else
        echo "✅ 端口 $port 可用"
    fi
done

# 5. 验证 docker-compose 配置
echo ""
echo "✅ 验证配置..."
if [ -f "docker-compose.yml" ]; then
    if docker-compose config >/dev/null 2>&1; then
        echo "✅ docker-compose.yml 配置有效"
    else
        echo "❌ docker-compose.yml 配置错误"
        docker-compose config
        exit 1
    fi
else
    echo "❌ 找不到 docker-compose.yml"
    exit 1
fi

# 6. 显示当前状态
echo ""
echo "📊 系统状态："
echo "- 工作目录: $(pwd)"
echo "- Docker 版本: $(docker --version)"
echo "- Docker Compose 版本: $(docker-compose --version)"
echo "- 可用磁盘空间: $(df -h . | tail -1 | awk '{print $4}')"

echo ""
echo "🎯 准备完成！"
echo ""
echo "▶️  启动服务: docker-compose up -d"
echo "🔍 查看状态: docker-compose ps"
echo "📋 查看日志: docker-compose logs -f"
echo ""
echo "🌐 服务地址（启动后）："
echo "   - XConnector: http://localhost:8081"
echo "   - Dynamo: http://localhost:8000"
echo "   - etcd: http://localhost:2379"
echo "   - NATS: http://localhost:8222"