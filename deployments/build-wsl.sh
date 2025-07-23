#!/bin/bash
# 修复路径问题的 WSL 构建脚本
# 文件名: build-wsl-fixed.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
XCONNECTOR_VERSION=${XCONNECTOR_VERSION:-"latest"}
IMAGES_DIR="./docker-images"

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}XConnector WSL 构建工具 (修复版)${NC}"
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
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}✗ Docker 服务未运行${NC}"
        echo -e "${YELLOW}请启动 Docker Desktop 并确保 WSL 集成已启用${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Docker 服务运行中${NC}"
    fi

    echo -e "${GREEN}Docker 版本: $(docker --version)${NC}"
}

# 智能定位项目根目录
find_project_root() {
    local current_dir="$(pwd)"
    local script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

    echo -e "${BLUE}脚本位置: $script_dir${NC}"
    echo -e "${BLUE}当前目录: $current_dir${NC}"

    # 方法1: 基于脚本位置推断（假设脚本在 deployments/ 目录下）
    if [[ "$script_dir" == */deployments ]]; then
        PROJECT_ROOT="$(dirname "$script_dir")"
        echo -e "${BLUE}从脚本位置推断项目根目录: $PROJECT_ROOT${NC}"
    # 方法2: 从当前目录向上查找
    elif [[ -f "$current_dir/requirements.txt" && -d "$current_dir/xconnector" ]]; then
        PROJECT_ROOT="$current_dir"
        echo -e "${BLUE}当前目录就是项目根目录: $PROJECT_ROOT${NC}"
    # 方法3: 向上查找包含标志文件的目录
    else
        local search_dir="$current_dir"
        while [[ "$search_dir" != "/" ]]; do
            if [[ -f "$search_dir/requirements.txt" && -d "$search_dir/xconnector" && -d "$search_dir/deployments" ]]; then
                PROJECT_ROOT="$search_dir"
                echo -e "${BLUE}找到项目根目录: $PROJECT_ROOT${NC}"
                break
            fi
            search_dir="$(dirname "$search_dir")"
        done

        if [[ -z "$PROJECT_ROOT" ]]; then
            echo -e "${RED}✗ 无法找到项目根目录${NC}"
            echo -e "${YELLOW}请确保在 xconnector 项目目录或其子目录中运行此脚本${NC}"
            echo -e "${YELLOW}项目根目录应包含: requirements.txt, xconnector/, deployments/${NC}"
            exit 1
        fi
    fi
}

# 检查项目结构
check_project_structure() {
    echo -e "${BLUE}检查项目结构...${NC}"

    # 首先定位项目根目录
    find_project_root

    echo -e "${GREEN}项目根目录: $PROJECT_ROOT${NC}"

    # 检查必要文件
    local required_files=(
        "xconnector"
        "requirements.txt"
        "deployments/docker/Dockerfile.xconnector-service"
        "integrations/dynamo"
    )

    for file in "${required_files[@]}"; do
        local full_path="$PROJECT_ROOT/$file"
        if [[ ! -e "$full_path" ]]; then
            echo -e "${RED}✗ 缺少必要文件/目录: $file${NC}"
            echo -e "${RED}  完整路径: $full_path${NC}"
            echo -e "${YELLOW}请检查项目结构是否完整${NC}"
            exit 1
        else
            echo -e "${GREEN}✓ 找到: $file${NC}"
        fi
    done

    echo -e "${GREEN}✓ 项目结构检查通过${NC}"

    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    echo -e "${BLUE}已切换到项目根目录: $(pwd)${NC}"
}

# 构建镜像
build_images() {
    echo -e "${BLUE}开始构建 XConnector 服务镜像...${NC}"

    # 显示构建上下文信息
    echo -e "${BLUE}构建上下文: $(pwd)${NC}"
    echo -e "${BLUE}Dockerfile: deployments/docker/Dockerfile.xconnector-service${NC}"

    # 检查 Dockerfile 是否存在
    if [[ ! -f "deployments/docker/Dockerfile.xconnector-service" ]]; then
        echo -e "${RED}✗ Dockerfile 不存在: deployments/docker/Dockerfile.xconnector-service${NC}"
        exit 1
    fi

    # 构建 XConnector 服务镜像
    echo -e "${BLUE}执行构建命令...${NC}"
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
    docker images | grep xconnector-service || echo "未找到 xconnector-service 镜像"
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
        if [[ $? -ne 0 ]]; then
            echo -e "${YELLOW}警告: 拉取 $dep 失败，但继续执行${NC}"
        fi
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

    # 检查镜像是否存在再导出
    if docker images | grep -q "quay.io/coreos/etcd.*v3.5.9"; then
        docker save "quay.io/coreos/etcd:v3.5.9" | gzip > "$IMAGES_DIR/etcd_v3.5.9.tar.gz"
    else
        echo -e "${YELLOW}警告: etcd 镜像不存在，跳过导出${NC}"
    fi

    if docker images | grep -q "nats.*2.10-alpine"; then
        docker save "nats:2.10-alpine" | gzip > "$IMAGES_DIR/nats_2.10-alpine.tar.gz"
    else
        echo -e "${YELLOW}警告: nats 镜像不存在，跳过导出${NC}"
    fi

    echo -e "${GREEN}✓ 镜像导出完成${NC}"
    echo -e "${BLUE}镜像文件位置: $IMAGES_DIR${NC}"
    ls -lh "$IMAGES_DIR"/ || echo "镜像目录为空"
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

    # 检查并复制 docker 配置
    if [[ -d "deployments/docker" ]]; then
        cp -r deployments/docker "$package_name/"
        echo -e "${GREEN}✓ 复制 docker 配置${NC}"
    else
        echo -e "${RED}✗ deployments/docker 目录不存在${NC}"
        exit 1
    fi

    # 检查并复制镜像
    if [[ -d "$IMAGES_DIR" ]]; then
        cp -r "$IMAGES_DIR" "$package_name/"
        echo -e "${GREEN}✓ 复制镜像文件${NC}"
    else
        echo -e "${YELLOW}警告: 镜像目录不存在，创建空目录${NC}"
        mkdir -p "$package_name/$IMAGES_DIR"
    fi

    # 创建部署脚本
    cat > "$package_name/deploy-server.sh" << 'EOF'
#!/bin/bash
# 服务器部署脚本

set -e

echo "=== XConnector 服务器部署 ==="

# 检查当前目录
if [[ ! -d "docker-images" || ! -d "docker" ]]; then
    echo "错误: 请在部署包目录中运行此脚本"
    exit 1
fi

# 加载镜像
echo "加载 Docker 镜像..."
cd docker-images

if [[ -n "$(ls -A . 2>/dev/null)" ]]; then
    for f in *.tar.gz; do
        if [[ -f "$f" ]]; then
            echo "加载 $f"
            gunzip -c "$f" | docker load
        fi
    done
else
    echo "警告: 镜像目录为空"
fi

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

    # 创建 README
    cat > "$package_name/README.md" << EOF
# XConnector 部署包

构建时间: $(date)
构建环境: WSL

## 部署步骤

1. 设置 AI-Dynamo 镜像名称：
\`\`\`bash
export DYNAMO_IMAGE=your-ai-dynamo-image:tag
\`\`\`

2. 运行部署脚本：
\`\`\`bash
./deploy-server.sh
\`\`\`

## 服务访问
- XConnector API: http://localhost:8081
- 健康检查: \`curl http://localhost:8081/health\`

## 文件说明
- docker/: Docker Compose 配置
- docker-images/: Docker 镜像文件
- deploy-server.sh: 部署脚本
EOF

    echo -e "${GREEN}✓ 部署包创建完成: $package_name${NC}"
    echo -e "${BLUE}包含文件:${NC}"
    ls -la "$package_name"/ || echo "无法列出部署包内容"
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
            ;;
    esac
}

# 执行主函数
main "$@"