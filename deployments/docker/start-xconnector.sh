#!/bin/bash
# offline-deploy.sh - 离线环境部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
XCONNECTOR_VERSION=${XCONNECTOR_VERSION:-"latest"}
DYNAMO_IMAGE=${DYNAMO_IMAGE:-"ai-dynamo:latest"}
BUILD_MODE=${BUILD_MODE:-"build"}  # build | load
IMAGES_DIR="./docker-images"

echo -e "${GREEN}=== XConnector + AI-Dynamo 离线部署工具 ===${NC}"

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}检查依赖...${NC}"

    local missing_deps=()

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}缺少依赖: ${missing_deps[*]}${NC}"
        exit 1
    fi

    echo -e "${GREEN}依赖检查通过${NC}"
}

# 构建镜像
build_images() {
    echo -e "${YELLOW}构建Docker镜像...${NC}"

    # 进入部署目录
    cd "$(dirname "$0")"

    # 构建XConnector服务镜像
    echo -e "${BLUE}构建XConnector服务镜像...${NC}"
    docker build \
        -f docker/Dockerfile.xconnector-service \
        -t xconnector-service:${XCONNECTOR_VERSION} \
        ../..

    echo -e "${GREEN}XConnector服务镜像构建完成${NC}"

    # 检查Dynamo镜像是否存在
    if ! docker image inspect ${DYNAMO_IMAGE} &> /dev/null; then
        echo -e "${YELLOW}警告: Dynamo镜像 ${DYNAMO_IMAGE} 不存在${NC}"
        echo -e "${YELLOW}请确保已经构建或加载了Dynamo镜像${NC}"
    else
        echo -e "${GREEN}发现Dynamo镜像: ${DYNAMO_IMAGE}${NC}"
    fi
}

# 导出镜像
export_images() {
    echo -e "${YELLOW}导出Docker镜像...${NC}"

    mkdir -p ${IMAGES_DIR}

    # 导出XConnector服务镜像
    echo -e "${BLUE}导出XConnector服务镜像...${NC}"
    docker save xconnector-service:${XCONNECTOR_VERSION} | gzip > ${IMAGES_DIR}/xconnector-service_${XCONNECTOR_VERSION}.tar.gz

    # 导出依赖镜像
    echo -e "${BLUE}导出依赖镜像...${NC}"

    # etcd
    docker pull quay.io/coreos/etcd:v3.5.9
    docker save quay.io/coreos/etcd:v3.5.9 | gzip > ${IMAGES_DIR}/etcd_v3.5.9.tar.gz

    # NATS
    docker pull nats:2.10-alpine
    docker save nats:2.10-alpine | gzip > ${IMAGES_DIR}/nats_2.10-alpine.tar.gz

    # 导出Dynamo镜像（如果存在）
    if docker image inspect ${DYNAMO_IMAGE} &> /dev/null; then
        echo -e "${BLUE}导出Dynamo镜像...${NC}"
        docker save ${DYNAMO_IMAGE} | gzip > ${IMAGES_DIR}/dynamo_${DYNAMO_IMAGE##*:}.tar.gz
    fi

    echo -e "${GREEN}镜像导出完成，保存在: ${IMAGES_DIR}${NC}"
    ls -lh ${IMAGES_DIR}/
}

# 加载镜像
load_images() {
    echo -e "${YELLOW}加载Docker镜像...${NC}"

    if [ ! -d "${IMAGES_DIR}" ]; then
        echo -e "${RED}镜像目录不存在: ${IMAGES_DIR}${NC}"
        exit 1
    fi

    # 加载所有镜像
    for image_file in ${IMAGES_DIR}/*.tar.gz; do
        if [ -f "$image_file" ]; then
            echo -e "${BLUE}加载镜像: $(basename $image_file)${NC}"
            gunzip -c "$image_file" | docker load
        fi
    done

    echo -e "${GREEN}镜像加载完成${NC}"
}

# 准备配置文件
prepare_configs() {
    echo -e "${YELLOW}准备配置文件...${NC}"

    # 确保配置目录存在
    mkdir -p docker/configs
    mkdir -p docker/dynamo-wrapper

    # 检查必要的配置文件
    local config_files=(
        "configs/disagg_xconnector_remote.yaml"
        "configs/xconnector_config.yaml"
        "dynamo-wrapper/startup-wrapper.py"
    )

    for config_file in "${config_files[@]}"; do
        if [ ! -f "docker/${config_file}" ]; then
            echo -e "${RED}缺少配置文件: docker/${config_file}${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}配置文件检查通过${NC}"
}

# 部署服务
deploy_services() {
    echo -e "${YELLOW}部署服务...${NC}"

    # 进入Docker目录
    cd docker

    # 设置环境变量
    export DYNAMO_IMAGE=${DYNAMO_IMAGE}
    export COMPOSE_PROJECT_NAME="xconnector-dynamo"

    # 创建必要的目录
    mkdir -p logs data

    # 停止已存在的服务
    echo -e "${BLUE}停止现有服务...${NC}"
    docker-compose down -v || true

    # 启动服务
    echo -e "${BLUE}启动服务...${NC}"
    docker-compose up -d

    # 等待服务启动
    echo -e "${BLUE}等待服务启动...${NC}"
    sleep 30

    # 检查服务状态
    check_services_health
}

# 检查服务健康状态
check_services_health() {
    echo -e "${YELLOW}检查服务健康状态...${NC}"

    local services=(
        "http://localhost:8081/health:XConnector服务"
        "http://localhost:2379/health:etcd"
        "http://localhost:8222/:NATS监控"
    )

    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"

        echo -n "检查 $name ... "

        max_retries=10
        for i in $(seq 1 $max_retries); do
            if curl -f -s "$url" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ 健康${NC}"
                break
            elif [ $i -eq $max_retries ]; then
                echo -e "${RED}✗ 不健康${NC}"
            else
                sleep 3
            fi
        done
    done

    # 显示容器状态
    echo -e "\n${BLUE}容器状态:${NC}"
    docker-compose ps
}

# 显示部署信息
show_deployment_info() {
    echo -e "\n${GREEN}=== 部署完成 ===${NC}"
    echo -e "服务访问地址:"
    echo -e "  - XConnector API: ${BLUE}http://localhost:8081${NC}"
    echo -e "  - Dynamo Frontend: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - etcd: ${BLUE}http://localhost:2379${NC}"
    echo -e "  - NATS监控: ${BLUE}http://localhost:8222${NC}"
    echo -e ""
    echo -e "常用命令:"
    echo -e "  - 查看日志: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "  - 停止服务: ${YELLOW}docker-compose down${NC}"
    echo -e "  - 重启服务: ${YELLOW}docker-compose restart${NC}"
    echo -e ""
    echo -e "配置文件位置:"
    echo -e "  - XConnector配置: ${BLUE}docker/configs/xconnector_config.yaml${NC}"
    echo -e "  - Dynamo配置: ${BLUE}docker/configs/disagg_xconnector_remote.yaml${NC}"
}

# 创建部署包
create_deployment_package() {
    echo -e "${YELLOW}创建部署包...${NC}"

    local package_name="xconnector-dynamo-deployment-$(date +%Y%m%d_%H%M%S)"
    local package_dir="/tmp/${package_name}"

    mkdir -p ${package_dir}

    # 复制部署文件
    cp -r docker/ ${package_dir}/
    cp -r ${IMAGES_DIR}/ ${package_dir}/
    cp "$0" ${package_dir}/

    # 创建部署说明
    cat > ${package_dir}/README.md << 'EOF'
# XConnector + AI-Dynamo 离线部署包

## 部署步骤

1. 加载Docker镜像:
   ```bash
   ./offline-deploy.sh load
   ```

2. 部署服务:
   ```bash
   ./offline-deploy.sh deploy
   ```

3. 检查服务状态:
   ```bash
   docker-compose -f docker/docker-compose.yml ps
   ```

## 配置文件

- `docker/configs/xconnector_config.yaml`: XConnector服务配置
- `docker/configs/disagg_xconnector_remote.yaml`: Dynamo配置

## 故障排除

- 查看日志: `docker-compose -f docker/docker-compose.yml logs -f`
- 重启服务: `docker-compose -f docker/docker-compose.yml restart`
EOF

    # 打包
    tar -czf ${package_name}.tar.gz -C /tmp ${package_name}

    echo -e "${GREEN}部署包已创建: ${package_name}.tar.gz${NC}"
    echo -e "包大小: $(du -h ${package_name}.tar.gz | cut -f1)"
}

# 主函数
main() {
    local action=${1:-"help"}

    case $action in
        "build")
            check_dependencies
            prepare_configs
            build_images
            export_images
            echo -e "${GREEN}构建完成！${NC}"
            ;;
        "load")
            check_dependencies
            load_images
            ;;
        "deploy")
            check_dependencies
            prepare_configs
            deploy_services
            show_deployment_info
            ;;
        "package")
            check_dependencies
            prepare_configs
            build_images
            export_images
            create_deployment_package
            ;;
        "all")
            check_dependencies
            prepare_configs
            build_images
            export_images
            deploy_services
            show_deployment_info
            ;;
        "help"|*)
            echo "用法: $0 {build|load|deploy|package|all}"
            echo ""
            echo "命令说明:"
            echo "  build   - 构建镜像并导出"
            echo "  load    - 从文件加载镜像"
            echo "  deploy  - 部署服务"
            echo "  package - 创建完整部署包"
            echo "  all     - 执行完整流程（构建+部署）"
            echo ""
            echo "环境变量:"
            echo "  DYNAMO_IMAGE=ai-dynamo:latest"
            echo "  XCONNECTOR_VERSION=latest"
            ;;
    esac
}

# 执行主函数
main "$@"