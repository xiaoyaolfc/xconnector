#!/bin/bash
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting XConnector with AI-Dynamo (Separated Mode)${NC}"

# 检查环境
check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"

    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi

    # 检查 NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${RED}NVIDIA Docker runtime is not available${NC}"
        exit 1
    fi

    echo -e "${GREEN}All requirements met${NC}"
}

# 设置环境变量
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"

    # 导出必要的环境变量
    export DYNAMO_IMAGE=${DYNAMO_IMAGE:-"nvcr.io/nvidia/pytorch:24.03-py3"}
    export COMPOSE_PROJECT_NAME="xconnector-dynamo"

    # 创建必要的目录
    mkdir -p logs
    mkdir -p data

    echo -e "${GREEN}Environment ready${NC}"
}

# 启动服务
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"

    # 进入正确的目录
    cd "$(dirname "$0")"

    # 启动所有服务
    docker-compose up -d

    # 等待服务就绪
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10

    # 检查服务状态
    echo -e "${YELLOW}Checking service status...${NC}"
    docker-compose ps

    # 验证 XConnector 服务
    if curl -f http://localhost:8081/health > /dev/null 2>&1; then
        echo -e "${GREEN}XConnector service is healthy${NC}"
    else
        echo -e "${RED}XConnector service is not responding${NC}"
    fi
}

# 显示日志
show_logs() {
    echo -e "${YELLOW}Recent logs:${NC}"
    docker-compose logs --tail=50
}

# 主函数
main() {
    check_requirements
    setup_environment
    start_services
    show_logs

    echo -e "${GREEN}Deployment complete!${NC}"
    echo -e "Services:"
    echo -e "  - XConnector API: http://localhost:8081"
    echo -e "  - Dynamo Frontend: http://localhost:8000"
    echo -e "  - NATS Monitor: http://localhost:8222"
    echo -e ""
    echo -e "To view logs: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "To stop: ${YELLOW}docker-compose down${NC}"
}

# 执行主函数
main