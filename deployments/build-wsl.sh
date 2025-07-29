#!/bin/bash
# Docker 镜像离线打包脚本 (WSL Ubuntu 环境)

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
PACKAGE_NAME="xconnector-dynamo-$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./docker-packages"
COMPOSE_FILE="docker-compose-local.yml"

echo -e "${GREEN}=== Docker 镜像离线打包工具 (WSL) ===${NC}"

# 检查 Docker 环境
check_docker() {
    echo -e "${YELLOW}检查 Docker 环境...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker 未安装${NC}"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}✗ Docker 服务未运行${NC}"
        echo -e "${YELLOW}请启动 Docker Desktop${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Docker 环境正常${NC}"
    echo -e "${BLUE}Docker 版本: $(docker --version)${NC}"
}

# 检查磁盘空间
check_disk_space() {
    echo -e "${YELLOW}检查磁盘空间...${NC}"

    available_space=$(df . | awk 'NR==2 {print $4}')
    available_gb=$((available_space / 1024 / 1024))

    echo -e "${BLUE}可用空间: ${available_gb} GB${NC}"

    if [ $available_gb -lt 10 ]; then
        echo -e "${YELLOW}⚠ 可用空间不足 10GB，可能导致打包失败${NC}"
        read -p "是否继续? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 创建输出目录
create_output_dir() {
    echo -e "${YELLOW}创建输出目录...${NC}"

    mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME"
    echo -e "${GREEN}✓ 输出目录: $OUTPUT_DIR/$PACKAGE_NAME${NC}"
}

# 重新构建 XConnector 镜像
rebuild_xconnector() {
    echo -e "${YELLOW}重新构建 XConnector 镜像...${NC}"

    # 检查 Dockerfile 是否存在
    if [[ ! -f "deployments/docker/Dockerfile.xconnector-service" ]]; then
        echo -e "${RED}✗ 未找到 Dockerfile: deployments/docker/Dockerfile.xconnector-service${NC}"
        exit 1
    fi

    # 构建镜像
    echo -e "${BLUE}构建 XConnector 服务镜像...${NC}"
    docker build \
        -f deployments/docker/Dockerfile.xconnector-service \
        -t xconnector-service:latest \
        . \
        --no-cache

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ XConnector 镜像构建成功${NC}"
    else
        echo -e "${RED}✗ XConnector 镜像构建失败${NC}"
        exit 1
    fi
}

# 列出所需镜像
list_required_images() {
    echo -e "${YELLOW}检查所需镜像...${NC}"

    # 从 docker-compose 文件提取镜像列表
    if [[ -f "$COMPOSE_FILE" ]]; then
        echo -e "${BLUE}从 $COMPOSE_FILE 提取镜像列表${NC}"
        required_images=($(grep "image:" "$COMPOSE_FILE" | awk '{print $2}' | sort -u))
    else
        echo -e "${YELLOW}未找到 $COMPOSE_FILE，使用默认镜像列表${NC}"
        required_images=(
            "xconnector-service:latest"
            "dynamo-nvidia:v0.3.0-vllm0.8.4-lmcache0.2.1-inline"
            "bitnami/etcd:auth-online"
            "nats:2.10-alpine"
        )
    fi

    echo -e "${BLUE}所需镜像列表:${NC}"
    for image in "${required_images[@]}"; do
        if docker image inspect "$image" &> /dev/null; then
            size=$(docker image inspect "$image" --format='{{.Size}}' | awk '{print int($1/1024/1024/1024*100)/100}')
            echo -e "${GREEN}✓ $image (${size}GB)${NC}"
        else
            echo -e "${RED}✗ $image (不存在)${NC}"
        fi
    done

    echo "${required_images[@]}"
}

# 导出 Docker 镜像
export_images() {
    echo -e "${YELLOW}导出 Docker 镜像...${NC}"

    # 获取镜像列表
    images=($(list_required_images))

    # 导出每个镜像
    for image in "${images[@]}"; do
        if docker image inspect "$image" &> /dev/null; then
            echo -e "${BLUE}导出镜像: $image${NC}"

            # 生成安全的文件名
            safe_name=$(echo "$image" | sed 's/[\/:]/_/g')
            output_file="$OUTPUT_DIR/$PACKAGE_NAME/${safe_name}.tar.gz"

            # 导出并压缩
            docker save "$image" | gzip > "$output_file"

            if [[ $? -eq 0 ]]; then
                size=$(du -h "$output_file" | cut -f1)
                echo -e "${GREEN}✓ 导出完成: $output_file (${size})${NC}"
            else
                echo -e "${RED}✗ 导出失败: $image${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ 跳过不存在的镜像: $image${NC}"
        fi
    done
}

# 复制配置文件
copy_configs() {
    echo -e "${YELLOW}复制配置文件...${NC}"

    # 复制 Docker Compose 文件
    if [[ -f "$COMPOSE_FILE" ]]; then
        cp "$COMPOSE_FILE" "$OUTPUT_DIR/$PACKAGE_NAME/"
        echo -e "${GREEN}✓ 复制: $COMPOSE_FILE${NC}"
    fi

    # 复制配置目录
    if [[ -d "configs" ]]; then
        cp -r configs "$OUTPUT_DIR/$PACKAGE_NAME/"
        echo -e "${GREEN}✓ 复制: configs/目录${NC}"
    fi

    # 复制集成脚本
    if [[ -d "xconnector-integration" ]]; then
        cp -r xconnector-integration "$OUTPUT_DIR/$PACKAGE_NAME/"
        echo -e "${GREEN}✓ 复制: xconnector-integration/目录${NC}"
    fi

    # 复制启动脚本
    if [[ -f "start_xconnector_dynamo.sh" ]]; then
        cp start_xconnector_dynamo.sh "$OUTPUT_DIR/$PACKAGE_NAME/"
        chmod +x "$OUTPUT_DIR/$PACKAGE_NAME/start_xconnector_dynamo.sh"
        echo -e "${GREEN}✓ 复制: start_xconnector_dynamo.sh${NC}"
    fi
}

# 创建加载脚本
create_load_script() {
    echo -e "${YELLOW}创建镜像加载脚本...${NC}"

    cat > "$OUTPUT_DIR/$PACKAGE_NAME/load_images.sh" << 'EOF'
#!/bin/bash
# Docker 镜像加载脚本

set -e

echo "=== Docker 镜像加载工具 ==="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "错误: Docker 服务未运行"
    exit 1
fi

# 加载所有镜像
echo "开始加载 Docker 镜像..."

for file in *.tar.gz; do
    if [[ -f "$file" ]]; then
        echo "加载镜像: $file"
        gunzip -c "$file" | docker load

        if [[ $? -eq 0 ]]; then
            echo "✓ 加载成功: $file"
        else
            echo "✗ 加载失败: $file"
        fi
    fi
done

echo ""
echo "镜像加载完成！"
echo ""
echo "查看已加载的镜像:"
docker images | grep -E "(xconnector|dynamo|etcd|nats)"

echo ""
echo "下一步："
echo "1. 启动服务: ./start_xconnector_dynamo.sh start"
echo "2. 查看状态: ./start_xconnector_dynamo.sh status"
EOF

    chmod +x "$OUTPUT_DIR/$PACKAGE_NAME/load_images.sh"
    echo -e "${GREEN}✓ 创建: load_images.sh${NC}"
}

# 创建部署说明
create_readme() {
    echo -e "${YELLOW}创建部署说明...${NC}"

    cat > "$OUTPUT_DIR/$PACKAGE_NAME/README.md" << EOF
# XConnector + Dynamo 离线部署包

**打包时间**: $(date)
**打包环境**: WSL Ubuntu
**包名称**: $PACKAGE_NAME

## 部署步骤

### 1. 传输到服务器
\`\`\`bash
# 方式1: 使用 scp
scp -r $PACKAGE_NAME user@server:/path/to/deploy/

# 方式2: 先压缩再传输
tar -czf $PACKAGE_NAME.tar.gz $PACKAGE_NAME/
scp $PACKAGE_NAME.tar.gz user@server:/path/to/deploy/

# 在服务器上解压
tar -xzf $PACKAGE_NAME.tar.gz
cd $PACKAGE_NAME/
\`\`\`

### 2. 加载 Docker 镜像
\`\`\`bash
# 执行镜像加载脚本
./load_images.sh
\`\`\`

### 3. 启动服务
\`\`\`bash
# 启动所有服务
./start_xconnector_dynamo.sh start

# 查看服务状态
./start_xconnector_dynamo.sh status

# 查看日志
./start_xconnector_dynamo.sh logs
\`\`\`

## 服务访问

- **XConnector API**: http://localhost:8081
- **Dynamo Frontend**: http://localhost:8000
- **etcd**: http://localhost:2379
- **NATS 监控**: http://localhost:8222

## 测试命令

\`\`\`bash
# 测试 XConnector
curl http://localhost:8081/health

# 测试 Dynamo
curl http://localhost:8000/health

# 运行联调测试
./start_xconnector_dynamo.sh test
\`\`\`

## 故障排除

\`\`\`bash
# 查看特定服务日志
docker-compose -f docker-compose-local.yml logs xconnector-service
docker-compose -f docker-compose-local.yml logs dynamo-worker

# 重启服务
./start_xconnector_dynamo.sh restart

# 停止服务
./start_xconnector_dynamo.sh stop
\`\`\`

## 文件说明

- \`*.tar.gz\`: Docker 镜像文件
- \`docker-compose-local.yml\`: 服务编排配置
- \`configs/\`: 配置文件目录
- \`xconnector-integration/\`: 集成脚本
- \`load_images.sh\`: 镜像加载脚本
- \`start_xconnector_dynamo.sh\`: 服务管理脚本
EOF

    echo -e "${GREEN}✓ 创建: README.md${NC}"
}

# 创建传输脚本
create_transfer_script() {
    echo -e "${YELLOW}创建传输脚本...${NC}"

    cat > "$OUTPUT_DIR/transfer_to_server.sh" << 'EOF'
#!/bin/bash
# 传输部署包到服务器脚本

if [[ $# -ne 2 ]]; then
    echo "用法: $0 <部署包目录> <服务器地址:路径>"
    echo "示例: $0 xconnector-dynamo-20241223_143022 user@192.168.1.100:/home/user/deploy/"
    exit 1
fi

PACKAGE_DIR="$1"
SERVER_TARGET="$2"

if [[ ! -d "$PACKAGE_DIR" ]]; then
    echo "错误: 部署包目录不存在: $PACKAGE_DIR"
    exit 1
fi

echo "=== 传输部署包到服务器 ==="
echo "部署包: $PACKAGE_DIR"
echo "目标: $SERVER_TARGET"
echo ""

# 压缩部署包
echo "压缩部署包..."
tar -czf "${PACKAGE_DIR}.tar.gz" "$PACKAGE_DIR"

if [[ $? -eq 0 ]]; then
    echo "✓ 压缩完成: ${PACKAGE_DIR}.tar.gz"
    echo "压缩包大小: $(du -h ${PACKAGE_DIR}.tar.gz | cut -f1)"
else
    echo "✗ 压缩失败"
    exit 1
fi

# 传输到服务器
echo ""
echo "传输到服务器..."
scp "${PACKAGE_DIR}.tar.gz" "$SERVER_TARGET"

if [[ $? -eq 0 ]]; then
    echo "✓ 传输完成"
    echo ""
    echo "在服务器上执行以下命令:"
    echo "  tar -xzf ${PACKAGE_DIR}.tar.gz"
    echo "  cd ${PACKAGE_DIR}/"
    echo "  ./load_images.sh"
    echo "  ./start_xconnector_dynamo.sh start"
else
    echo "✗ 传输失败"
    exit 1
fi
EOF

    chmod +x "$OUTPUT_DIR/transfer_to_server.sh"
    echo -e "${GREEN}✓ 创建: transfer_to_server.sh${NC}"
}

# 显示包信息
show_package_info() {
    echo -e "\n${GREEN}=== 打包完成 ===${NC}"

    # 计算包大小
    package_size=$(du -sh "$OUTPUT_DIR/$PACKAGE_NAME" | cut -f1)

    echo -e "${BLUE}部署包信息:${NC}"
    echo -e "  路径: $OUTPUT_DIR/$PACKAGE_NAME"
    echo -e "  大小: $package_size"

    echo -e "\n${BLUE}包含文件:${NC}"
    ls -la "$OUTPUT_DIR/$PACKAGE_NAME/" | head -20

    if [[ $(ls "$OUTPUT_DIR/$PACKAGE_NAME/" | wc -l) -gt 18 ]]; then
        echo -e "  ... (更多文件)"
    fi

    echo -e "\n${GREEN}=== 后续步骤 ===${NC}"
    echo -e "1. 传输到服务器:"
    echo -e "   ${BLUE}$OUTPUT_DIR/transfer_to_server.sh $PACKAGE_NAME user@server:/path/${NC}"
    echo -e ""
    echo -e "2. 在服务器上部署:"
    echo -e "   ${BLUE}tar -xzf $PACKAGE_NAME.tar.gz${NC}"
    echo -e "   ${BLUE}cd $PACKAGE_NAME/${NC}"
    echo -e "   ${BLUE}./load_images.sh${NC}"
    echo -e "   ${BLUE}./start_xconnector_dynamo.sh start${NC}"
}

# 主函数
main() {
    local action=${1:-"all"}

    case $action in
        "build")
            check_docker
            rebuild_xconnector
            ;;
        "export")
            check_docker
            check_disk_space
            create_output_dir
            export_images
            ;;
        "package")
            check_docker
            check_disk_space
            create_output_dir
            copy_configs
            create_load_script
            create_readme
            create_transfer_script
            ;;
        "all")
            check_docker
            check_disk_space
            create_output_dir
            rebuild_xconnector
            export_images
            copy_configs
            create_load_script
            create_readme
            create_transfer_script
            show_package_info
            ;;
        "help"|*)
            echo "用法: $0 {build|export|package|all}"
            echo ""
            echo "命令说明:"
            echo "  build   - 重新构建 XConnector 镜像"
            echo "  export  - 仅导出镜像"
            echo "  package - 仅打包配置文件和脚本"
            echo "  all     - 执行完整打包流程（默认）"
            echo ""
            echo "示例:"
            echo "  $0 all                    # 完整打包"
            echo "  $0 build                  # 仅重新构建镜像"
            echo "  $0 export                 # 仅导出镜像"
            ;;
    esac
}

# 执行主函数
main "$@"