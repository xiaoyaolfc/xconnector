#!/bin/bash
# WSL XConnector 构建脚本
# 文件名: build-wsl.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
XCONNECTOR_VERSION=${XCONNECTOR_VERSION:-"latest"}
BUILD_MODE=${BUILD_MODE:-"build"}
IMAGES_DIR="./docker-images"

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}XConnector WSL 构建工具${NC}"
echo -e "${GREEN}====================================${NC}"

# 检查是否在 WSL 中
check_wsl() {
    if [[ ! -f /proc/version ]] || ! grep -q "microsoft\|WSL" /proc/version; then
        echo -e "${YELLOW}警告: 未检测到 WSL 环境${NC}"
        echo -e "${YELLOW}当前环境: $(uname -a)${NC}"
        read -p "是否继续? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ WSL 环境检测成功${NC}"
    fi
}

# 检查 Docker
check_docker() {
    echo -e "${BLUE}检查 Docker 环境...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker 未安装${NC}"
        echo -e "${YELLOW}请安装 Docker 或启用 Docker Desktop WSL 集成${NC}"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}✗ Docker 服务未运行${NC}"
        echo -e "${YELLOW}请启动 Docker 服务${NC}"

        # 尝试启动 Docker 服务（如果是 WSL 中直接安装的）
        if command -v service &> /dev/null; then
            echo -e "${BLUE}尝试启动 Docker 服务...${NC}"
            sudo service docker start || true
            sleep 3

            if docker info &> /dev/null; then
                echo -e "${GREEN}✓ Docker 服务启动成功${NC}"
            else
                echo -e "${RED}✗ 无法启动 Docker 服务${NC}"
                echo -e "${YELLOW}请检查 Docker Desktop 是否启用了 WSL 集成${NC}"
                exit 1
            fi
        else
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Docker 服务运行中${NC}"
    fi

    echo -e "${GREEN}Docker 版本: $(docker --version)${NC}"
}

# 检查项目结构
check_project_structure() {
    echo -e "${BLUE}检查项目结构...${NC}"

    # 获取脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

    echo -e "${BLUE}项目根目录: $PROJECT_ROOT${NC}"

    # 检查必要文件
    local required_files=(
        "xconnector"
        "requirements.txt"
        "deployments/docker/Dockerfile.xconnector-service"
        "integrations/dynamo"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -e "$PROJECT_ROOT/$file" ]]; then
            echo -e "${RED}✗ 缺少必要文件/目录: $file${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}✓ 项目结构检查通过${NC}"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"
}

# 构建镜像
build_images() {
    echo -e "${BLUE}开始构建 XConnector 服务镜像...${NC}"

    # 显示构建上下文信息
    echo -e "${BLUE}构建上下文: $(pwd)${NC}"
    echo -e "${BLUE}Dockerfile: deployments/docker/Dockerfile.xconnector-service${NC}"

    # 构建 XConnector 服务镜像
    docker build \
        -f deployments/docker/Dockerfile.xconnector-service \
        -t xconnector-service:${XCONNECTOR_VERSION} \
        . \
        --progress=plain

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ XConnector 服务镜像构建成功${NC}"
    else
        echo -e "${RED}✗ XConnector 服务镜像构建失败${NC}"
        exit 1
    fi

    # 显示镜像信息
    echo -e "${BLUE}构建的镜像:${NC}"
    docker images | grep xconnector-service
}

# 拉取依赖镜像
pull_dependencies() {
    echo -e "${BLUE}拉取依赖镜像...${NC}"

    local deps=(
        "quay.io/coreos/etcd:v3.5.9"
        "nats:2.10-alpine"
    )

    for dep in "${deps[@]}"; do
        echo -e "${BLUE}拉取 $dep...${NC}"
        docker pull "$dep"
    done

    echo -e "${GREEN}✓ 依赖镜像拉取完成${NC}"
}

# 导出镜像
export_images() {
    echo -e "${BLUE}导出 Docker 镜像...${NC}"

    # 创建镜像目录
    mkdir -p "$IMAGES_DIR"

    # 导出 XConnector 服务镜像
    echo -e "${BLUE}导出 XConnector 服务镜像...${NC}"
    docker save "xconnector-service:${XCONNECTOR_VERSION}" | gzip > "$IMAGES_DIR/xconnector-service_${XCONNECTOR_VERSION}.tar.gz"

    # 导出依赖镜像
    echo -e "${BLUE}导出依赖镜像...${NC}"
    docker save "quay.io/coreos/etcd:v3.5.9" | gzip > "$IMAGES_DIR/etcd_v3.5.9.tar.gz"
    docker save "nats:2.10-alpine" | gzip > "$IMAGES_DIR/nats_2.10-alpine.tar.gz"

    echo -e "${GREEN}✓ 镜像导出完成${NC}"
    echo -e "${BLUE}镜像文件位置: $IMAGES_DIR${NC}"
    ls -lh "$IMAGES_DIR"/
}

# 创建部署包
create_deployment_package() {
    echo -e "${BLUE}创建部署包...${NC}"

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local package_name="xconnector-deployment-${timestamp}"

    # 创建部署包目录
    mkdir -p "$package_name"

    # 复制文件
    echo -e "${BLUE}复制部署文件...${NC}"
    cp -r deployments/docker "$package_name/"
    cp -r "$IMAGES_DIR" "$package_name/"

    # 创建部署脚本
    cat > "$package_name/deploy-server.sh" << 'EOF'
#!/bin/bash
# 服务器部署脚本

set -e

echo "=== XConnector 服务器部署 ==="

# 加载镜像
echo "加载 Docker 镜像..."
cd docker-images
for f in *.tar.gz; do
    echo "加载 $f"
    gunzip -c "$f" | docker load
done

cd ../docker

# 检查 AI-Dynamo 镜像
if [[ -z "$DYNAMO_IMAGE" ]]; then
    echo "警告: 未设置 DYNAMO_IMAGE 环境变量"
    echo "请设置你的 AI-Dynamo 镜像名称:"
    echo "export DYNAMO_IMAGE=your-dynamo-image:tag"
    echo ""
    echo "继续使用默认配置..."
    export DYNAMO_IMAGE="ai-dynamo:latest"
fi

echo "使用 Dynamo 镜像: $DYNAMO_IMAGE"

# 启动服务
echo "启动服务..."
docker-compose up -d

echo "等待服务启动..."
sleep 30

# 健康检查
echo "检查服务状态..."
docker-compose ps

echo ""
echo "服务访问地址:"
echo "  - XConnector API: http://localhost:8081"
echo "  - 健康检查: curl http://localhost:8081/health"

echo ""
echo "部署完成!"
EOF

    chmod +x "$package_name/deploy-server.sh"

    # 创建 Windows 传输脚本
    cat > "$package_name/transfer-to-server.sh" << 'EOF'
#!/bin/bash
# 传输到服务器的脚本
# 使用方法: ./transfer-to-server.sh user@server:/path/to/deploy/

if [[ $# -eq 0 ]]; then
    echo "使用方法: $0 user@server:/path/to/deploy/"
    echo "示例: $0 ubuntu@192.168.1.100:/home/ubuntu/xconnector/"
    exit 1
fi

REMOTE_PATH=$1
PACKAGE_DIR=$(dirname "$0")

echo "传输部署包到: $REMOTE_PATH"

# 使用 rsync 传输（保持权限和符号链接）
rsync -avz --progress "$PACKAGE_DIR/" "$REMOTE_PATH"

echo ""
echo "传输完成！"
echo "在服务器上运行:"
echo "  cd $REMOTE_PATH"
echo "  ./deploy-server.sh"
EOF

    chmod +x "$package_name/transfer-to-server.sh"

    # 创建 README
    cat > "$package_name/README.md" << EOF
# XConnector 部署包

构建时间: $(date)
构建环境: WSL $(cat /proc/version | grep -o 'WSL[0-9]*')

## 快速部署

### 方法1: 自动传输并部署
\`\`\`bash
./transfer-to-server.sh user@server:/path/to/deploy/
ssh user@server 'cd /path/to/deploy && ./deploy-server.sh'
\`\`\`

### 方法2: 手动部署
1. 将整个文件夹复制到服务器
2. 在服务器上运行: \`./deploy-server.sh\`

## 自定义 AI-Dynamo 镜像
\`\`\`bash
export DYNAMO_IMAGE=your-dynamo-image:tag
./deploy-server.sh
\`\`\`

## 服务访问
- XConnector API: http://localhost:8081
- 健康检查: \`curl http://localhost:8081/health\`
EOF

    echo -e "${GREEN}✓ 部署包创建完成: $package_name${NC}"
    echo -e "${BLUE}包含文件:${NC}"
    ls -la "$package_name"/
}

# 清理函数
cleanup() {
    echo -e "${BLUE}清理临时文件...${NC}"
    # 这里可以添加清理逻辑
}

# 主函数
main() {
    local action=${1:-"all"}

    case $action in
        "check")
            check_wsl
            check_docker
            check_project_structure
            ;;
        "build")
            check_wsl
            check_docker
            check_project_structure
            build_images
            ;;
        "export")
            pull_dependencies
            export_images
            ;;
        "package")
            check_wsl
            check_docker
            check_project_structure
            build_images
            pull_dependencies
            export_images
            create_deployment_package
            ;;
        "all")
            check_wsl
            check_docker
            check_project_structure
            build_images
            pull_dependencies
            export_images
            create_deployment_package
            cleanup
            ;;
        "help"|*)
            echo "用法: $0 {check|build|export|package|all}"
            echo ""
            echo "命令说明:"
            echo "  check   - 检查环境和项目结构"
            echo "  build   - 仅构建镜像"
            echo "  export  - 仅导出镜像"
            echo "  package - 创建完整部署包"
            echo "  all     - 执行完整流程（默认）"
            echo ""
            echo "环境变量:"
            echo "  XCONNECTOR_VERSION=latest"
            echo ""
            echo "示例:"
            echo "  $0 all                    # 完整构建流程"
            echo "  $0 check                  # 仅检查环境"
            echo "  XCONNECTOR_VERSION=v1.0 $0 build  # 指定版本构建"
            ;;
    esac
}

# 捕获中断信号
trap cleanup INT TERM

# 执行主函数
main "$@"