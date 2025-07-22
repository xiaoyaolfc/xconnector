#!/bin/bash
set -e

# 配置
PROJECT_NAME="xconnector"
VERSION=${VERSION:-"latest"}
OUTPUT_DIR="dist"

echo "Packaging XConnector for deployment..."

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 创建临时目录
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${PROJECT_NAME}"

# 复制必要文件
echo "Copying files..."
mkdir -p ${PACKAGE_DIR}

# 复制源码（排除不必要的文件）
rsync -av \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='dist' \
  --exclude='*.egg-info' \
  --exclude='.pytest_cache' \
  --exclude='logs' \
  --exclude='data' \
  xconnector/ ${PACKAGE_DIR}/xconnector/

# 复制其他必要文件
cp -r integrations/ ${PACKAGE_DIR}/
cp -r deployments/ ${PACKAGE_DIR}/
cp -r csrc/ ${PACKAGE_DIR}/ 2>/dev/null || true
cp requirements.txt ${PACKAGE_DIR}/
cp setup.py ${PACKAGE_DIR}/
cp pyproject.toml ${PACKAGE_DIR}/ 2>/dev/null || true

# 创建部署信息文件
cat > ${PACKAGE_DIR}/deployment-info.json << EOF
{
  "version": "${VERSION}",
  "build_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

# 打包
OUTPUT_FILE="${OUTPUT_DIR}/${PROJECT_NAME}-${VERSION}.tar.gz"
echo "Creating archive: ${OUTPUT_FILE}"
tar -czf ${OUTPUT_FILE} -C ${TEMP_DIR} ${PROJECT_NAME}

# 创建部署脚本
cat > ${OUTPUT_DIR}/deploy-on-server.sh << 'EOF'
#!/bin/bash
set -e

# 解压
echo "Extracting package..."
tar -xzf xconnector-*.tar.gz

# 进入目录
cd xconnector

# 构建镜像
echo "Building Docker images..."
cd deployments/docker

# 构建 XConnector 服务镜像
docker build -f Dockerfile.xconnector-service -t xconnector-service:latest ../..

# 启动服务
echo "Starting services..."
./start-xconnector.sh

echo "Deployment complete!"
EOF

chmod +x ${OUTPUT_DIR}/deploy-on-server.sh

# 清理
rm -rf ${TEMP_DIR}

echo "Package created: ${OUTPUT_FILE}"
echo "Size: $(du -h ${OUTPUT_FILE} | cut -f1)"
echo ""
echo "To deploy:"
echo "1. Copy ${OUTPUT_FILE} and deploy-on-server.sh to your server"
echo "2. Run: ./deploy-on-server.sh"