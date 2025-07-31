#!/bin/bash
# 增强版离线部署脚本 - 专门处理 etcd 认证问题

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

echo -e "${GREEN}=== XConnector 增强部署脚本 (etcd 认证修复版) ===${NC}"

# 预检查 etcd 镜像配置
precheck_etcd_image() {
    echo -e "${YELLOW}预检查 etcd 镜像配置...${NC}"

    # 检查是否存在正确的 etcd 镜像
    if docker image inspect "bitnami/etcd:auth-online" &> /dev/null; then
        echo -e "${GREEN}✓ 找到 etcd 镜像: bitnami/etcd:auth-online${NC}"
    else
        echo -e "${YELLOW}⚠ etcd 镜像不存在，尝试拉取...${NC}"

        # 尝试拉取或使用其他版本
        if docker pull bitnami/etcd:latest; then
            docker tag bitnami/etcd:latest bitnami/etcd:auth-online
            echo -e "${GREEN}✓ etcd 镜像准备就绪${NC}"
        else
            echo -e "${RED}✗ 无法获取 etcd 镜像${NC}"
            return 1
        fi
    fi

    return 0
}

# 创建 etcd 专用网络（如果需要）
setup_etcd_network() {
    echo -e "${YELLOW}设置 etcd 网络配置...${NC}"

    # 确保网络存在
    if ! docker network ls | grep -q "xconnector-net"; then
        docker network create xconnector-net --driver bridge --subnet=172.28.0.0/16 || true
        echo -e "${GREEN}✓ 创建网络: xconnector-net${NC}"
    else
        echo -e "${GREEN}✓ 网络已存在: xconnector-net${NC}"
    fi
}

# 停止并清理现有服务
clean_existing_services() {
    echo -e "${YELLOW}清理现有服务...${NC}"

    # 停止 docker-compose 服务
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans || true

    # 清理可能残留的容器
    for container in etcd xconnector-service dynamo-worker nats; do
        if docker ps -aq -f name="$container" | grep -q .; then
            echo "清理容器: $container"
            docker rm -f "$container" || true
        fi
    done

    # 清理悬空容器和网络
    docker container prune -f || true
    docker network prune -f || true

    echo -e "${GREEN}✓ 清理完成${NC}"
}

# 分步启动服务（etcd 优先）
start_etcd_first() {
    echo -e "${YELLOW}优先启动 etcd 服务...${NC}"

    # 只启动 etcd
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d etcd

    # 等待 etcd 启动并验证
    echo -n "等待 etcd 启动..."
    max_retries=30
    for i in $(seq 1 $max_retries); do
        if docker ps --filter "name=etcd" --filter "status=running" | grep -q "etcd"; then
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

    # 额外等待 etcd 内部初始化
    echo -n "等待 etcd 内部初始化..."
    sleep 10

    # 验证 etcd 健康状态
    for i in $(seq 1 20); do
        if curl -f -s http://localhost:2379/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq 20 ]]; then
            echo -e " ${RED}✗ etcd 健康检查失败${NC}"
            return 1
        else
            echo -n "."
            sleep 3
        fi
    done

    return 0
}

# 配置 etcd 无认证模式
configure_etcd_no_auth() {
    echo -e "${YELLOW}配置 etcd 无认证模式...${NC}"

    # 获取 etcd 容器名
    etcd_container=$(docker ps --filter "name=etcd" --format "{{.Names}}" | head -1)

    if [[ -z "$etcd_container" ]]; then
        echo -e "${RED}✗ 找不到 etcd 容器${NC}"
        return 1
    fi

    echo "找到 etcd 容器: $etcd_container"

    # 在容器内配置无认证模式
    docker exec "$etcd_container" sh -c '
        # 设置环境变量
        export ETCDCTL_API=3
        export ALLOW_NONE_AUTHENTICATION=yes
        export ETCD_AUTH_TOKEN=""
        export ETCD_ROOT_PASSWORD=""

        echo "配置 etcd 无认证模式..."

        # 禁用认证（如果已启用）
        etcdctl --endpoints=localhost:2379 auth disable 2>/dev/null || echo "认证未启用或已禁用"

        # 测试无认证访问
        echo "测试无认证访问..."
        if etcdctl --endpoints=localhost:2379 put test-auth-key test-value 2>/dev/null; then
            echo "✓ 无认证模式配置成功"
            etcdctl --endpoints=localhost:2379 del test-auth-key 2>/dev/null
        else
            echo "⚠ etcdctl 测试失败，但 HTTP 接口可能仍然可用"
        fi

        # 测试 HTTP 接口
        echo "测试 HTTP 接口..."
        if curl -f -s http://localhost:2379/health > /dev/null; then
            echo "✓ HTTP 接口可用"
        else
            echo "✗ HTTP 接口不可用"
            exit 1
        fi
    '

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ etcd 无认证模式配置完成${NC}"

        # 从主机测试连接
        echo -n "主机连接测试..."
        if curl -f -s http://localhost:2379/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
        else
            echo -e " ${RED}✗${NC}"
            return 1
        fi

        return 0
    else
        echo -e "${RED}✗ etcd 配置失败${NC}"
        return 1
    fi
}

# 启动其他服务
start_remaining_services() {
    echo -e "${YELLOW}启动其他服务...${NC}"

    # 启动 NATS
    echo "启动 NATS..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d nats

    # 等待 NATS 启动
    echo -n "等待 NATS 启动..."
    for i in $(seq 1 15); do
        if curl -f -s http://localhost:8222/ &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq 15 ]]; then
            echo -e " ${YELLOW}⚠ NATS 可能未完全就绪${NC}"
        else
            echo -n "."
            sleep 2
        fi
    done

    # 启动 XConnector 服务
    echo "启动 XConnector 服务..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d xconnector-service

    # 等待 XConnector 启动
    echo -n "等待 XConnector 启动..."
    for i in $(seq 1 30); do
        if curl -f -s http://localhost:8081/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        elif [[ $i -eq 30 ]]; then
            echo -e " ${RED}✗ XConnector 启动超时${NC}"
            return 1
        else
            echo -n "."
            sleep 3
        fi
    done

    return 0
}

# 启动 Dynamo 服务
start_dynamo_service() {
    echo -e "${YELLOW}启动 Dynamo 服务...${NC}"

    # 确保前置服务都正常
    echo "验证前置服务状态..."
    local services_ok=true

    # 检查 etcd
    if ! curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e "${RED}✗ etcd 服务不可用${NC}"
        services_ok=false
    fi

    # 检查 XConnector
    if ! curl -f -s http://localhost:8081/health &> /dev/null; then
        echo -e "${RED}✗ XConnector 服务不可用${NC}"
        services_ok=false
    fi

    if [[ "$services_ok" != "true" ]]; then
        echo -e "${RED}前置服务未就绪，无法启动 Dynamo${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ 前置服务验证通过${NC}"

    # 启动 Dynamo
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d dynamo-worker

    # Dynamo 需要更长的启动时间
    echo -n "等待 Dynamo 启动（可能需要几分钟）..."
    for i in $(seq 1 60); do
        if curl -f -s http://localhost:8000/health &> /dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        elif [[ $i -eq 60 ]]; then
            echo -e " ${YELLOW}⚠ Dynamo 可能仍在加载模型${NC}"
            return 0
        else
            echo -n "."
            sleep 5
        fi
    done
}

# 验证完整部署
validate_deployment() {
    echo -e "${YELLOW}验证部署状态...${NC}"

    # 检查所有容器状态
    echo -e "${BLUE}容器状态:${NC}"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps

    echo -e "\n${BLUE}服务健康检查:${NC}"

    # etcd
    if curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e "etcd: ${GREEN}✓ 健康${NC}"
    else
        echo -e "etcd: ${RED}✗ 不健康${NC}"
        return 1
    fi

    # XConnector
    if curl -f -s http://localhost:8081/health &> /dev/null; then
        echo -e "XConnector: ${GREEN}✓ 健康${NC}"

        # 获取 XConnector 状态
        status=$(curl -s http://localhost:8081/status 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('service',{}).get('status','unknown'))" 2>/dev/null || echo "unknown")
        echo -e "  状态: $status"
    else
        echo -e "XConnector: ${RED}✗ 不健康${NC}"
        return 1
    fi

    # NATS
    if curl -f -s http://localhost:8222/ &> /dev/null; then
        echo -e "NATS: ${GREEN}✓ 健康${NC}"
    else
        echo -e "NATS: ${YELLOW}⚠ 不健康${NC}"
    fi

    # Dynamo
    if curl -f -s http://localhost:8000/health &> /dev/null; then
        echo -e "Dynamo: ${GREEN}✓ 健康${NC}"
    else
        echo -e "Dynamo: ${YELLOW}⚠ 可能仍在启动${NC}"
    fi

    # 测试跨服务连接
    echo -e "\n${BLUE}跨服务连接测试:${NC}"

    # 测试 Dynamo 到 etcd 的连接
    if docker exec dynamo-worker curl -f -s http://etcd:2379/health &> /dev/null 2>&1; then
        echo -e "Dynamo -> etcd: ${GREEN}✓ 连接正常${NC}"
    else
        echo -e "Dynamo -> etcd: ${RED}✗ 连接失败${NC}"
        echo -e "${YELLOW}这是导致 'Failed to connect to etcd server' 错误的根本原因${NC}"
        return 1
    fi

    # 测试 XConnector 到 etcd 的连接
    if docker exec xconnector-service curl -f -s http://etcd:2379/health &> /dev/null 2>&1; then
        echo -e "XConnector -> etcd: ${GREEN}✓ 连接正常${NC}"
    else
        echo -e "XConnector -> etcd: ${YELLOW}⚠ 连接可能有问题${NC}"
    fi

    return 0
}

# 故障诊断
diagnose_issues() {
    echo -e "${YELLOW}运行故障诊断...${NC}"

    # 检查网络连通性
    echo -e "${BLUE}网络诊断:${NC}"

    # 检查 Docker 网络
    echo "Docker 网络信息:"
    docker network ls | grep -E "(xconnector|bridge)"

    # 检查容器网络配置
    echo -e "\n容器网络配置:"
    for container in etcd xconnector-service dynamo-worker; do
        if docker ps --filter "name=$container" | grep -q "$container"; then
            echo "  $container:"
            docker exec "$container" ip addr show eth0 2>/dev/null | grep "inet " || echo "    无法获取 IP"
        fi
    done

    # 检查 DNS 解析
    echo -e "\nDNS 解析测试:"
    if docker exec dynamo-worker nslookup etcd &> /dev/null; then
        echo -e "  dynamo-worker -> etcd: ${GREEN}✓${NC}"
    else
        echo -e "  dynamo-worker -> etcd: ${RED}✗${NC}"
    fi

    # 显示关键日志
    echo -e "\n${BLUE}关键错误日志:${NC}"

    echo "etcd 日志 (最近 10 行):"
    docker logs etcd --tail 10 2>/dev/null | grep -E "(error|Error|ERROR|fail|Fail|FAIL)" || echo "  无明显错误"

    echo -e "\nDynamo 日志 (最近 10 行):"
    docker logs dynamo-worker --tail 10 2>/dev/null | grep -E "(error|Error|ERROR|fail|Fail|FAIL|etcd)" || echo "  无明显错误"
}

# 修复建议
suggest_fixes() {
    echo -e "\n${GREEN}=== 修复建议 ===${NC}"

    if ! curl -f -s http://localhost:2379/health &> /dev/null; then
        echo -e "${YELLOW}etcd 服务问题:${NC}"
        echo -e "  1. 重启 etcd: ${BLUE}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart etcd${NC}"
        echo -e "  2. 检查认证配置: ${BLUE}./fix-etcd-auth.sh check${NC}"
        echo -e "  3. 重新配置 etcd: ${BLUE}./fix-etcd-auth.sh reconfigure${NC}"
    fi

    if docker exec dynamo-worker curl -f -s http://etcd:2379/health &> /dev/null 2>&1; then
        echo -e "${GREEN}✓ 网络连接正常${NC}"
    else
        echo -e "${YELLOW}网络连接问题:${NC}"
        echo -e "  1. 重建网络: ${BLUE}docker network rm xconnector-net && docker network create xconnector-net${NC}"
        echo -e "  2. 重启所有服务: ${BLUE}$0 restart${NC}"
        echo -e "  3. 检查防火墙配置"
    fi

    echo -e "\n${YELLOW}如果问题仍然存在:${NC}"
    echo -e "  1. 查看完整日志: ${BLUE}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs${NC}"
    echo -e "  2. 手动测试连接: ${BLUE}docker exec dynamo-worker ping etcd${NC}"
    echo -e "  3. 清理重新部署: ${BLUE}$0 clean && $0 deploy${NC}"
}

# 主函数
main() {
    local action=${1:-"deploy"}

    case $action in
        "deploy")
            precheck_etcd_image || exit 1
            setup_etcd_network
            clean_existing_services

            # 分步启动
            if start_etcd_first; then
                echo -e "${GREEN}✓ etcd 启动成功${NC}"
            else
                echo -e "${RED}✗ etcd 启动失败${NC}"
                exit 1
            fi

            if configure_etcd_no_auth; then
                echo -e "${GREEN}✓ etcd 认证配置成功${NC}"
            else
                echo -e "${RED}✗ etcd 认证配置失败${NC}"
                exit 1
            fi

            if start_remaining_services; then
                echo -e "${GREEN}✓ 基础服务启动成功${NC}"
            else
                echo -e "${RED}✗ 基础服务启动失败${NC}"
                exit 1
            fi

            start_dynamo_service

            # 验证部署
            if validate_deployment; then
                echo -e "\n${GREEN}=== 部署成功 ===${NC}"
                show_access_info
            else
                echo -e "\n${RED}=== 部署验证失败 ===${NC}"
                diagnose_issues
                suggest_fixes
                exit 1
            fi
            ;;
        "restart")
            echo -e "${YELLOW}重启所有服务...${NC}"
            docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
            main deploy
            ;;
        "fix-etcd")
            # 专门修复 etcd 问题
            if start_etcd_first && configure_etcd_no_auth; then
                echo -e "${GREEN}✓ etcd 修复完成${NC}"
                # 重启依赖服务
                docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" restart xconnector-service dynamo-worker
            else
                echo -e "${RED}✗ etcd 修复失败${NC}"
                exit 1
            fi
            ;;
        "diagnose")
            diagnose_issues
            suggest_fixes
            ;;
        "clean")
            clean_existing_services
            echo -e "${GREEN}✓ 清理完成${NC}"
            ;;
        "status")
            validate_deployment
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
            ;;
        "help"|*)
            echo "用法: $0 {deploy|restart|fix-etcd|diagnose|clean|status|logs}"
            echo ""
            echo "命令说明:"
            echo "  deploy    - 完整部署流程（推荐）"
            echo "  restart   - 重启所有服务"
            echo "  fix-etcd  - 专门修复 etcd 认证问题"
            echo "  diagnose  - 运行故障诊断"
            echo "  clean     - 清理所有服务"
            echo "  status    - 检查服务状态"
            echo "  logs      - 查看实时日志"
            echo ""
            echo "推荐执行流程:"
            echo "  1. $0 deploy     # 首次部署"
            echo "  2. $0 status     # 检查状态"
            echo "  3. $0 fix-etcd   # 如果遇到 etcd 连接问题"
            echo "  4. $0 diagnose   # 如果仍有问题"
            ;;
    esac
}

# 显示访问信息
show_access_info() {
    echo -e "\n${GREEN}=== 服务访问信息 ===${NC}"
    echo -e "XConnector API: ${BLUE}http://localhost:8081${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8081/health${NC}"
    echo -e "  - 服务状态: ${BLUE}curl http://localhost:8081/status${NC}"
    echo -e ""
    echo -e "Dynamo API: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:8000/health${NC}"
    echo -e ""
    echo -e "etcd: ${BLUE}http://localhost:2379${NC}"
    echo -e "  - 健康检查: ${BLUE}curl http://localhost:2379/health${NC}"
    echo -e ""
    echo -e "NATS 监控: ${BLUE}http://localhost:8222${NC}"
    echo -e ""
    echo -e "${GREEN}=== 测试命令 ===${NC}"
    echo -e "测试 etcd 连接: ${YELLOW}docker exec dynamo-worker curl http://etcd:2379/health${NC}"
    echo -e "查看所有日志: ${YELLOW}docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f${NC}"
    echo -e ""
    echo -e "${GREEN}=== 故障排除 ===${NC}"
    echo -e "如果遇到 'Failed to connect to etcd server' 错误:"
    echo -e "  1. ${YELLOW}$0 fix-etcd${NC}    # 修复 etcd 认证"
    echo -e "  2. ${YELLOW}$0 diagnose${NC}    # 运行诊断"
    echo -e "  3. ${YELLOW}$0 restart${NC}     # 重启服务"
}

# 执行主函数
main "$@"