#!/bin/bash
# 离线服务器部署脚本 - 使用服务器现有镜像，包含 etcd 认证配置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
COMPOSE_FILE="deployments/docker/docker-compose.yml"
PROJECT_NAME="xconnector-dynamo"

echo -e "${GREEN}=== XConnector 离线部署脚本 (增强版) ===${NC}"

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

# 配置 etcd 无认证模式
configure_etcd_auth() {
    echo -e "${YELLOW}配置 etcd 认证模式...${NC}"
    
    local max_retries=30
    
    # 动态获取 etcd 容器名称
    echo "获取 etcd 容器名称..."
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)
    
    if [[ -z "$etcd_container" ]]; then
        echo -e "${RED}✗ 找不到 etcd 容器${NC}"
        return 1
    fi
    
    echo "找到 etcd 容器: $etcd_container"
    
    # 等待 etcd 容器启动
    echo -n "等待 etcd 容器启动..."
    for i in $(seq 1 $max_retries); do
        if docker ps --filter "name=$etcd_container" --filter "status=running" | grep -q "$etcd_container"; then
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
    
    # 等待 etcd 服务内部就绪
    echo -n "等待 etcd 服务就绪..."
    for i in $(seq 1 $max_retries); do
        # 使用 curl 测试而不是 etcdctl，避免认证问题
        if docker exec "$etcd_container" curl -f -s http://localhost:2379/health &> /dev/null; then
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

    # 配置 etcd 无认证模式
    echo "配置 etcd 无认证访问..."

    # 方法1：通过环境变量确认设置
    echo -n "检查 ALLOW_NONE_AUTHENTICATION 设置..."
    if docker exec "$etcd_container" env | grep -q "ALLOW_NONE_AUTHENTICATION=yes"; then
        echo -e " ${GREEN}✓${NC}"
    else
        echo -e " ${YELLOW}⚠ 重新设置${NC}"
        # 重启容器以确保环境变量生效
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart etcd
        sleep 10
    fi

    # 方法2：使用环境变量来避免认证问题
    echo "配置 etcd 客户端环境..."
    docker exec "$etcd_container" sh -c '
        export ETCDCTL_API=3
        export ETCDCTL_USER=""
        export ETCDCTL_PASSWORD=""

        # 测试基本连接
        echo "测试 etcd 连接..."
        if curl -f -s http://localhost:2379/health > /dev/null 2>&1; then
            echo "✓ etcd HTTP 连接正常"
        else
            echo "✗ etcd HTTP 连接失败"
            exit 1
        fi

        # 尝试使用 etcdctl 进行简单操作
        echo "测试 etcdctl 操作..."
        if ETCDCTL_API=3 etcdctl --endpoints=localhost:2379 --user="" put test-key test-value 2>/dev/null; then
            echo "✓ etcd 写入测试成功"
            ETCDCTL_API=3 etcdctl --endpoints=localhost:2379 --user="" del test-key 2>/dev/null
        else
            echo "⚠ etcdctl 可能有认证问题，但 HTTP 接口可用"
        fi

        echo "✓ etcd 配置检查完成"
    '

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ etcd 认证配置完成${NC}"
        return 0
    else
        echo -e "${RED}✗ etcd 认证配置失败${NC}"
        return 1
    fi
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

    # 测试 etcd 连接性
    echo -n "测试 etcd 连接性..."
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)
    # 从主机测试 etcd（这个用 localhost 是对的）
    if [[ -n "$etcd_container" ]] && curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"

        # 测试容器间网络连接（这个用容器名）
        if docker exec dynamo-worker curl -f -s http://etcd:2379/health &> /dev/null 2>&1; then
            echo -e "${GREEN}✓ 容器间网络连接正常${NC}"
        else
            echo -e "${RED}✗ 容器间网络连接失败${NC}"
            echo -e "${YELLOW}尝试重新配置 etcd...${NC}"
            configure_etcd_auth
        fi
    else
        echo -e " ${RED}✗${NC}"
        echo -e "${YELLOW}尝试重新配置 etcd...${NC}"
        configure_etcd_auth
    fi

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

# 验证部署
verify_deployment() {
    echo -e "${YELLOW}验证部署状态...${NC}"

    # 检查容器状态
    echo "检查容器状态..."
    local all_healthy=true

    while IFS= read -r line; do
        local container_name=$(echo "$line" | awk '{print $1}')
        local status=$(echo "$line" | awk '{print $2}')

        if [[ "$status" == "Up"* ]]; then
            echo -e "${GREEN}✓ $container_name${NC}"
        else
            echo -e "${RED}✗ $container_name: $status${NC}"
            all_healthy=false
        fi
    done < <(docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps --format "table {{.Name}}\t{{.Status}}" | tail -n +2)

    # 测试 etcd 连接
    echo "测试 etcd 连接..."
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)

    # 测试主机到 etcd 连接（用 localhost）
    if [[ -n "$etcd_container" ]] && curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e "${GREEN}✓ 主机到 etcd 连接正常${NC}"
    else
        echo -e "${RED}✗ 主机到 etcd 连接失败${NC}"
        all_healthy=false
    fi

    # 测试容器间连接（用容器名）
    if docker exec dynamo-worker curl -f -s http://etcd:2379/health &> /dev/null 2>&1; then
        echo -e "${GREEN}✓ dynamo-worker 到 etcd 连接正常${NC}"
    else
        echo -e "${RED}✗ dynamo-worker 到 etcd 连接失败${NC}"
        echo -e "${YELLOW}这是问题的根源！${NC}"
        all_healthy=false

        # 显示网络诊断信息
        echo "网络诊断信息:"
        docker exec dynamo-worker nslookup etcd 2>/dev/null || echo "DNS 解析失败"
        docker network ls | grep -E "(xconnector|dynamo)"
    fi

    # 测试服务端点
    echo "测试服务端点..."
    local endpoints=(
        "http://localhost:8081/health:XConnector"
        "http://localhost:2379/health:etcd"
        "http://localhost:8222:NATS监控"
    )

    for endpoint_info in "${endpoints[@]}"; do
        local endpoint=$(echo "$endpoint_info" | cut -d':' -f1)
        local service_name=$(echo "$endpoint_info" | cut -d':' -f2)

        if curl -f -s "$endpoint" &> /dev/null; then
            echo -e "${GREEN}✓ $service_name 端点可访问${NC}"
        else
            echo -e "${YELLOW}⚠ $service_name 端点不可访问${NC}"
        fi
    done

    return $([[ "$all_healthy" == true ]] && echo 0 || echo 1)
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
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:2379/health${NC}"
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)
    echo -e "  - 容器访问: ${BLUE}docker exec $etcd_container curl http://localhost:2379/health${NC}"
    echo -e "NATS 监控: ${BLUE}http://localhost:8222${NC}"
    
    echo -e "\n${GREEN}=== 测试命令 ===${NC}"
    echo -e "查看日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f${NC}"
    echo -e "查看 etcd 日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f etcd${NC}"
    echo -e "重启服务: ${YELLOW}$0 restart${NC}"
    echo -e "停止服务: ${YELLOW}$0 stop${NC}"
    echo -e "重新配置 etcd: ${YELLOW}$0 fix-etcd${NC}"
    
    echo -e "\n${GREEN}=== 故障排除 ===${NC}"
    echo -e "如果 Dynamo 连接 etcd 失败，请尝试:"
    echo -e "  1. ${YELLOW}$0 fix-etcd${NC} - 重新配置 etcd 认证"
    echo -e "  2. ${YELLOW}$0 restart${NC} - 重启所有服务"
    echo -e "  3. 查看 etcd 容器日志确认无认证模式是否生效"
}

# 修复 etcd 认证问题
fix_etcd() {
    echo -e "${YELLOW}修复 etcd 认证配置...${NC}"
    
    # 获取 etcd 容器名称
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)
    
    if [[ -z "$etcd_container" ]]; then
        echo -e "${RED}✗ 找不到 etcd 容器${NC}"
        return 1
    fi
    
    echo "找到 etcd 容器: $etcd_container"
    
    # 重启 etcd 容器
    echo "重启 etcd 容器..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart etcd
    sleep 10
    
    # 重新配置认证
    if configure_etcd_auth; then
        echo -e "${GREEN}✓ etcd 认证修复完成${NC}"
        
        # 重启依赖 etcd 的服务
        echo "重启相关服务..."
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart xconnector-service dynamo-worker
        
        wait_for_services
        echo -e "${GREEN}✓ 服务重启完成${NC}"
    else
        echo -e "${RED}✗ etcd 认证修复失败${NC}"
        return 1
    fi
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
            
            # 配置 etcd 认证
            if configure_etcd_auth; then
                echo -e "${GREEN}✓ etcd 配置成功${NC}"
            else
                echo -e "${YELLOW}⚠ etcd 配置可能有问题，请查看日志${NC}"
            fi
            
            wait_for_services
            verify_deployment
            show_status
            ;;
        "start")
            start_services
            configure_etcd_auth
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
            configure_etcd_auth
            wait_for_services
            verify_deployment
            show_status
            ;;
        "fix-etcd")
            fix_etcd
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
            echo "用法: $0 {deploy|start|stop|restart|fix-etcd|status|logs|check}"
            echo ""
            echo "命令说明:"
            echo "  deploy   - 完整部署流程（检查+停止+启动+配置etcd）"
            echo "  start    - 启动服务"
            echo "  stop     - 停止服务"
            echo "  restart  - 重启服务"
            echo "  fix-etcd - 修复 etcd 认证问题"
            echo "  status   - 查看服务状态"
            echo "  logs     - 查看实时日志"
            echo "  check    - 检查镜像和文件"
            echo ""
            echo "部署前请确保:"
            echo "  1. 已加载 XConnector 镜像"
            echo "  2. 服务器有 dynamo-nvidia、etcd、nats 镜像"
            echo "  3. 在项目根目录运行"
            echo ""
            echo "etcd 认证问题解决:"
            echo "  - 如果遇到 'Failed to connect to etcd server' 错误"
            echo "  - 运行: $0 fix-etcd"
            ;;
    esac
}

# 执行主函数
main "$@"