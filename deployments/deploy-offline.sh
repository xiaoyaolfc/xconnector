#!/bin/bash
# 离线服务器部署脚本 - 使用服务器现有镜像

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
COMPOSE_FILE="docker/docker-compose.yml"
PROJECT_NAME="xconnector-dynamo"

echo -e "${GREEN}=== XConnector 离线部署脚本 ===${NC}"

# 检查必要的镜像是否存在
check_required_images() {
    echo -e "${YELLOW}检查必需镜像...${NC}"
    
    local required_images=(
        "xconnector-service:latest"                              # 从本地上传
        "dynamo-nvidia:v0.3.0-vllm0.8.4-lmcache0.2.1-inline"  # 服务器现有
        "bitnami/etcd:auth-online"                              # 服务器现有
        "nats:latest"                                           # 服务器现有
    )
    
    local missing_images=()
    for image in "${required_images[@]}"; do
        if docker image inspect "$image" &> /dev/null; then
            echo -e "${GREEN}✓ $image${NC}"
        else
            echo -e "${RED}✗ $image${NC}"
            missing_images+=("$image")
        fi
    done
    
    if [ ${#missing_images[@]} -gt 0 ]; then
        echo -e "${RED}缺少以下镜像:${NC}"
        for img in "${missing_images[@]}"; do
            echo -e "  - $img"
        done
        
        # 特别提示 XConnector 镜像
        if [[ " ${missing_images[@]} " =~ " xconnector-service:latest " ]]; then
            echo -e "${YELLOW}请先加载 XConnector 镜像:${NC}"
            echo -e "  ${BLUE}gunzip -c xconnector-service_latest.tar.gz | docker load${NC}"
        fi
        
        return 1
    fi
    
    echo -e "${GREEN}✓ 所有必需镜像都已准备就绪${NC}"
    return 0
}

# 检查项目文件结构
check_project_structure() {
    echo -e "${YELLOW}检查项目结构...${NC}"
    
    local required_files=(
        "$COMPOSE_FILE"
        "integrations/dynamo/configs/disagg_with_xconnector.yaml"
        "integrations/dynamo/configs/xconnector_config.yaml"
        "integrations/dynamo/startup-wrapper.py"
        "integrations/dynamo/extension_loader.py"
        "integrations/dynamo/registry.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}✗ 缺少文件: $file${NC}"
            return 1
        fi
        echo -e "${GREEN}✓ $file${NC}"
    done
    
    # 检查 __init__.py 文件
    if [[ ! -f "integrations/dynamo/__init__.py" ]]; then
        echo -e "${YELLOW}创建 integrations/dynamo/__init__.py${NC}"
        touch integrations/dynamo/__init__.py
    fi
    
    return 0
}

# 停止现有服务
stop_services() {
    echo -e "${YELLOW}停止现有服务...${NC}"
    
    if docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps -q | grep -q .; then
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans
        echo -e "${GREEN}✓ 现有服务已停止${NC}"
    else
        echo -e "${BLUE}没有运行中的服务${NC}"
    fi
}

# 启动服务
start_services() {
    echo -e "${YELLOW}启动服务...${NC}"
    
    # 设置环境变量
    export COMPOSE_PROJECT_NAME="$PROJECT_NAME"
    
    # 启动服务
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
    
    echo -e "${GREEN}✓ 服务启动完成${NC}"
}

# 等待服务就绪
wait_for_services() {
    echo -e "${YELLOW}等待服务就绪...${NC}"
    
    local max_retries=30
    
    # 等待 XConnector 服务
    echo -n "等待 XConnector 服务..."
    for i in $(seq 1 $max_retries); do
        if curl -f -s http://localhost:8081/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq $max_retries ]]; then
            echo -e " ${RED}✗ 超时${NC}"
            return 1
        else
            echo -n "."
            sleep 2
        fi
    done
    
    # 等待 etcd 服务
    echo -n "等待 etcd 服务..."
    for i in $(seq 1 $max_retries); do
        if curl -f -s http://localhost:2379/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq $max_retries ]]; then
            echo -e " ${RED}✗ 超时${NC}"
            return 1
        else
            echo -n "."
            sleep 2
        fi
    done
    
    # 等待 Dynamo 服务（需要更长时间）
    echo -n "等待 Dynamo 服务..."
    local dynamo_retries=60
    for i in $(seq 1 $dynamo_retries); do
        if curl -f -s http://localhost:8000/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq $dynamo_retries ]]; then
            echo -e " ${YELLOW}⚠ 可能仍在加载模型${NC}"
            break
        else
            echo -n "."
            sleep 3
        fi
    done
}

# 显示服务状态
show_status() {
    echo -e "\n${GREEN}=== 服务状态 ===${NC}"
    
    # 显示容器状态
    echo -e "${BLUE}容器状态:${NC}"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    
    echo -e "\n${BLUE}服务访问信息:${NC}"
    echo -e "XConnector API: ${BLUE}http://localhost:8081${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8081/health${NC}"
    echo -e "  - 服务状态: ${BLUE}curl http://localhost:8081/status${NC}"
    echo -e ""
    echo -e "Dynamo API: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8000/health${NC}"
    echo -e ""
    echo -e "etcd: ${BLUE}http://localhost:2379${NC}"
    echo -e "NATS 监控: ${BLUE}http://localhost:8222${NC}"
    
    echo -e "\n${GREEN}=== 测试命令 ===${NC}"
    echo -e "查看日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f${NC}"
    echo -e "重启服务: ${YELLOW}$0 restart${NC}"
    echo -e "停止服务: ${YELLOW}$0 stop${NC}"
}

# 主函数
main() {
    local action=${1:-"deploy"}
    
    case $action in
        "deploy")
            if ! check_required_images; then
                echo -e "${RED}镜像检查失败，部署终止${NC}"
                exit 1
            fi
            if ! check_project_structure; then
                echo -e "${RED}项目结构检查失败，部署终止${NC}"
                exit 1
            fi
            stop_services
            start_services
            wait_for_services
            show_status
            ;;
        "start")
            start_services
            wait_for_services
            show_status
            ;;
        "stop")
            stop_services
            echo -e "${GREEN}✓ 服务已停止${NC}"
            ;;
        "restart")
            stop_services
            start_services
            wait_for_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
            ;;
        "check")
            check_required_images
            check_project_structure
            ;;
        "help"|*)
            echo "用法: $0 {deploy|start|stop|restart|status|logs|check}"
            echo ""
            echo "命令说明:"
            echo "  deploy  - 完整部署流程（检查+停止+启动）"
            echo "  start   - 启动服务"
            echo "  stop    - 停止服务"
            echo "  restart - 重启服务"
            echo "  status  - 查看服务状态"
            echo "  logs    - 查看实时日志"
            echo "  check   - 检查镜像和文件"
            echo ""
            echo "部署前请确保:"
            echo "  1. 已加载 XConnector 镜像"
            echo "  2. 服务器有 dynamo-nvidia、etcd、nats 镜像"
            echo "  3. 在项目根目录运行"
            ;;
    esac
}

# 执行主函数
main "$@"