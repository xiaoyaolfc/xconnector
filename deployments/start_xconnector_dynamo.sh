#!/bin/bash
# XConnector + Dynamo 联调启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
COMPOSE_FILE="docker-compose-local.yml"
PROJECT_NAME="xconnector-dynamo"

echo -e "${GREEN}=== XConnector + Dynamo 联调启动脚本 ===${NC}"

# 检查必要文件
check_files() {
    echo -e "${YELLOW}检查必要文件...${NC}"

    local required_files=(
        "$COMPOSE_FILE"
        "configs/disagg_with_xconnector.yaml"
        "configs/xconnector_config.yaml"
        "xconnector-integration/startup_wrapper.py"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}✗ 缺少文件: $file${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 找到: $file${NC}"
    done
}

# 检查 Docker 镜像
check_images() {
    echo -e "${YELLOW}检查 Docker 镜像...${NC}"

    local required_images=(
        "xconnector-service:latest"
        "dynamo-nvidia:v0.3.0-vllm0.8.4-lmcache0.2.1-inline"
        "bitnami/etcd:auth-online"
    )

    for image in "${required_images[@]}"; do
        if docker image inspect "$image" &> /dev/null; then
            echo -e "${GREEN}✓ 镜像存在: $image${NC}"
        else
            echo -e "${RED}✗ 镜像不存在: $image${NC}"
            exit 1
        fi
    done
}

# 创建必要目录
create_directories() {
    echo -e "${YELLOW}创建必要目录...${NC}"

    mkdir -p configs
    mkdir -p xconnector-integration
    mkdir -p logs

    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 停止现有服务
stop_services() {
    echo -e "${YELLOW}停止现有服务...${NC}"

    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans || true

    # 清理悬空容器
    docker container prune -f || true

    echo -e "${GREEN}✓ 服务停止完成${NC}"
}

# 启动服务
start_services() {
    echo -e "${YELLOW}启动服务...${NC}"

    # 设置环境变量
    export COMPOSE_PROJECT_NAME="$PROJECT_NAME"

    # 启动服务
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d

    echo -e "${GREEN}✓ 服务启动命令执行完成${NC}"
}

# 等待服务就绪
wait_for_services() {
    echo -e "${YELLOW}等待服务就绪...${NC}"

    # 等待 XConnector 服务
    echo -n "等待 XConnector 服务启动..."
    max_retries=30
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
    echo -n "等待 etcd 服务启动..."
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

    # 等待 Dynamo 服务（可能需要更长时间）
    echo -n "等待 Dynamo 服务启动..."
    max_retries=60  # Dynamo 需要更长时间启动
    for i in $(seq 1 $max_retries); do
        if curl -f -s http://localhost:8000/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq $max_retries ]]; then
            echo -e " ${YELLOW}⚠ Dynamo 可能仍在启动中${NC}"
            break
        else
            echo -n "."
            sleep 3
        fi
    done
}

# 检查服务状态
check_service_status() {
    echo -e "${YELLOW}检查服务状态...${NC}"

    # 显示容器状态
    echo -e "${BLUE}容器状态:${NC}"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps

    echo -e "\n${BLUE}服务健康检查:${NC}"

    # XConnector 服务
    if curl -f -s http://localhost:8081/health &> /dev/null; then
        echo -e "XConnector 服务: ${GREEN}✓ 健康${NC}"

        # 获取详细状态
        status=$(curl -s http://localhost:8081/status 2>/dev/null || echo "{}")
        echo -e "  服务状态: $(echo "$status" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('service',{}).get('status','unknown'))" 2>/dev/null || echo "unknown")"
    else
        echo -e "XConnector 服务: ${RED}✗ 不健康${NC}"
    fi

    # etcd 服务
    if curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e "etcd 服务: ${GREEN}✓ 健康${NC}"
    else
        echo -e "etcd 服务: ${RED}✗ 不健康${NC}"
    fi

    # NATS 服务
    if curl -f -s http://localhost:8222/ &> /dev/null; then
        echo -e "NATS 服务: ${GREEN}✓ 健康${NC}"
    else
        echo -e "NATS 服务: ${RED}✗ 不健康${NC}"
    fi

    # Dynamo 服务
    if curl -f -s http://localhost:8000/health &> /dev/null; then
        echo -e "Dynamo 服务: ${GREEN}✓ 健康${NC}"
    else
        echo -e "Dynamo 服务: ${YELLOW}⚠ 可能仍在启动或有问题${NC}"
        echo -e "  建议检查日志: ${BLUE}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs dynamo-worker${NC}"
    fi
}

# 显示访问信息
show_access_info() {
    echo -e "\n${GREEN}=== 服务访问信息 ===${NC}"
    echo -e "XConnector API: ${BLUE}http://localhost:8081${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8081/health${NC}"
    echo -e "  - 服务状态: ${BLUE}curl http://localhost:8081/status${NC}"
    echo -e "  - 适配器列表: ${BLUE}curl http://localhost:8081/adapters${NC}"
    echo -e ""
    echo -e "Dynamo Frontend: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8000/health${NC}"
    echo -e "  - API 文档: ${BLUE}http://localhost:8000/docs${NC}"
    echo -e ""
    echo -e "etcd: ${BLUE}http://localhost:2379${NC}"
    echo -e "NATS 监控: ${BLUE}http://localhost:8222${NC}"
    echo -e ""
    echo -e "${GREEN}=== 常用命令 ===${NC}"
    echo -e "查看所有日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f${NC}"
    echo -e "查看 XConnector 日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f xconnector-service${NC}"
    echo -e "查看 Dynamo 日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f dynamo-worker${NC}"
    echo -e "重启服务: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart${NC}"
    echo -e "停止服务: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down${NC}"
    echo -e ""
    echo -e "${GREEN}=== 测试联调 ===${NC}"
    echo -e "测试 XConnector 状态:"
    echo -e "  ${BLUE}curl -s http://localhost:8081/status | python3 -m json.tool${NC}"
    echo -e ""
    echo -e "测试 Dynamo 推理 (需要等待模型加载完成):"
    echo -e "  ${BLUE}curl -X POST http://localhost:8000/v1/chat/completions \\${NC}"
    echo -e "  ${BLUE}  -H \"Content-Type: application/json\" \\${NC}"
    echo -e "  ${BLUE}  -d '{\"model\": \"deepseek-ai/DeepSeek-R1-Distill-Llama-8B\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'${NC}"
}

# 显示日志监控
show_logs() {
    echo -e "${YELLOW}显示服务日志（按 Ctrl+C 退出）...${NC}"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
}

# 主函数
main() {
    local action=${1:-"start"}

    case $action in
        "start")
            check_files
            check_images
            create_directories
            stop_services
            start_services
            wait_for_services
            check_service_status
            show_access_info
            ;;
        "stop")
            stop_services
            echo -e "${GREEN}✓ 服务已停止${NC}"
            ;;
        "restart")
            stop_services
            start_services
            wait_for_services
            check_service_status
            show_access_info
            ;;
        "status")
            check_service_status
            show_access_info
            ;;
        "logs")
            show_logs
            ;;
        "clean")
            stop_services
            docker system prune -f
            echo -e "${GREEN}✓ 清理完成${NC}"
            ;;
        "test")
            echo -e "${YELLOW}运行联调测试...${NC}"

            # 测试 XConnector
            echo -e "${BLUE}测试 XConnector 服务:${NC}"
            curl -s http://localhost:8081/status | python3 -m json.tool || echo "XConnector 不可访问"

            echo -e "\n${BLUE}测试 XConnector 适配器:${NC}"
            curl -s http://localhost:8081/adapters | python3 -m json.tool || echo "适配器信息不可访问"

            # 测试 Dynamo
            echo -e "\n${BLUE}测试 Dynamo 服务:${NC}"
            curl -s http://localhost:8000/health || echo "Dynamo 不可访问"

            echo -e "\n${BLUE}测试完成${NC}"
            ;;
        "help"|*)
            echo "用法: $0 {start|stop|restart|status|logs|clean|test}"
            echo ""
            echo "命令说明:"
            echo "  start   - 启动所有服务（默认）"
            echo "  stop    - 停止所有服务"
            echo "  restart - 重启所有服务"
            echo "  status  - 查看服务状态"
            echo "  logs    - 查看实时日志"
            echo "  clean   - 停止服务并清理资源"
            echo "  test    - 运行联调测试"
            echo "  help    - 显示帮助信息"
            ;;
    esac
}

# 执行主函数
main "$@"