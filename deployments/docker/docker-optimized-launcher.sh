#!/bin/bash
# Docker优化的Dynamo启动脚本 - 智能配置选择版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Docker Dynamo + XConnector 智能启动器 ===${NC}"

# 配置映射表 - 修复文件名
declare -A CONFIG_MAPPING=(
    ["agg.yaml"]="agg_with_xconnector.yaml"
    ["agg_router.yaml"]="agg_router_with_xconnector.yaml"
    ["disagg.yaml"]="disagg_with_xconnector.yaml"
    ["disagg_router.yaml"]="disagg_router_with_xconnector.yaml"
)

# 检查服务依赖
check_service_dependencies() {
    echo -e "${YELLOW}检查服务依赖...${NC}"
    local services=(
        "etcd:http://etcd:2379/health"
        "nats:http://nats:8222/"
        "xconnector:http://xconnector-service:8081/health"
    )

    for service_info in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_info"
        echo -n "  检查 $name..."
        local max_attempts=15
        local attempt=1
        while [ $attempt -le $max_attempts ]; do
            if curl -f -s "$url" > /dev/null 2>&1; then
                echo -e " ${GREEN}✓${NC}"
                break
            fi
            sleep 2
            ((attempt++))
        done
    done
}

# 显示可用配置选项
show_available_configs() {
    echo -e "\n${BLUE}=== 可用的配置选项 ===${NC}"
    echo -e "1. ${GREEN}agg${NC} - 聚合模式 (Aggregated)"
    echo -e "2. ${GREEN}agg_router${NC} - 聚合路由模式 (Aggregated with Router)"
    echo -e "3. ${GREEN}disagg${NC} - 分离模式 (Disaggregated)"
    echo -e "4. ${GREEN}disagg_router${NC} - 分离路由模式 (Disaggregated with Router)"
    echo -e "\n请选择配置模式 (1-4) 或按 Enter 使用默认 [disagg]:"
}

# 交互式选择配置
interactive_config_selection() {
    show_available_configs
    read -r choice

    case $choice in
        1)
            echo "agg_with_xconnector.yaml"
            ;;
        2)
            echo "agg_router_with_xconnector.yaml"
            ;;
        3|"")
            echo "disagg_with_xconnector_simple.yaml"
            ;;
        4)
            echo "disagg_router_with_xconnector.yaml"
            ;;
        *)
            echo -e "${YELLOW}无效选择，使用默认配置 disagg${NC}"
            echo "disagg_with_xconnector_simple.yaml"
            ;;
    esac
}

# 智能选择XConnector配置
smart_config_selection() {
    # 检查是否有命令行参数传入（从环境变量或参数）
    local requested_config="${DYNAMO_CONFIG_TYPE:-}"

    if [[ -n "$requested_config" ]]; then
        echo -e "${BLUE}检测到环境变量指定的配置类型: $requested_config${NC}"
        local xconnector_config="${CONFIG_MAPPING[$requested_config]}"
        if [[ -n "$xconnector_config" ]]; then
            echo "$xconnector_config"
            return 0
        fi
    fi

    # 如果没有环境变量，则进行交互式选择
    interactive_config_selection
}

# 检测图模块类型
detect_graph_module() {
    local config_file="$1"

    if [[ "$config_file" =~ agg.*router ]]; then
        echo "graphs.agg_router:Frontend"
    elif [[ "$config_file" =~ agg ]]; then
        echo "graphs.agg:Frontend"
    elif [[ "$config_file" =~ disagg.*router ]]; then
        echo "graphs.disagg_router:Frontend"
    elif [[ "$config_file" =~ disagg ]]; then
        echo "graphs.disagg:Frontend"
    else
        echo "graphs.disagg:Frontend"  # 默认
    fi
}

# 验证配置文件存在
verify_config_file() {
    local config_name="$1"
    local config_path="/workspace/xconnector-configs/$config_name"

    if [[ -f "$config_path" ]]; then
        echo "$config_path"
        return 0
    else
        echo -e "${RED}✗ 配置文件不存在: $config_path${NC}"
        echo -e "${YELLOW}可用的配置文件:${NC}"
        ls -la /workspace/xconnector-configs/ 2>/dev/null || echo "配置目录不存在"
        return 1
    fi
}

# 设置环境
setup_environment() {
    echo -e "${YELLOW}设置环境变量...${NC}"

    # 确保在正确的工作目录
    cd /workspace/examples/llm

    # 设置关键环境变量
    export ENABLE_XCONNECTOR=true
    export XCONNECTOR_SERVICE_URL="http://xconnector-service:8081"
    export XCONNECTOR_FAIL_ON_ERROR=false
    export ETCD_ENDPOINTS="http://etcd:2379"
    export ETCD_URL="http://etcd:2379"
    export NATS_URL="nats://nats:4222"
    export NATS_SERVER="nats://nats:4222"
    export PYTHONPATH=/workspace/examples/llm:/workspace/xconnector-integration:/workspace:$PYTHONPATH"

    # 设置XConnector集成
    mkdir -p /workspace/xconnector-integration
    cp /tmp/extension_loader.py /workspace/xconnector-integration/ 2>/dev/null || true
    cp /tmp/registry.py /workspace/xconnector-integration/ 2>/dev/null || true
    cp /tmp/__init__.py /workspace/xconnector-integration/ 2>/dev/null || true

    # 运行启动包装器
    if [[ -f "/workspace/startup-wrapper.py" ]]; then
        echo -e "${YELLOW}运行启动包装器...${NC}"
        python /workspace/startup-wrapper.py
    fi

    echo -e "${GREEN}✓ 环境设置完成${NC}"
}

# 显示配置信息
show_config_info() {
    local config_file="$1"
    local graph_module="$2"

    echo -e "\n${BLUE}=== 最终配置信息 ===${NC}"
    echo -e "工作目录: ${GREEN}$(pwd)${NC}"
    echo -e "配置文件: ${GREEN}$config_file${NC}"
    echo -e "图模块: ${GREEN}$graph_module${NC}"
    echo -e "Python路径: ${GREEN}$PYTHONPATH${NC}"

    echo -e "\n${BLUE}配置文件内容预览:${NC}"
    head -10 "$config_file" | sed 's/^/  /'

    # 验证graphs模块
    echo -e "\n${YELLOW}验证graphs模块...${NC}"
    if python -c "import graphs; print('  ✓ graphs模块导入成功')" 2>/dev/null; then
        echo -e "${GREEN}✓ graphs模块验证通过${NC}"
    else
        echo -e "${YELLOW}⚠ graphs模块导入测试失败，但继续启动${NC}"
    fi
}

# 主启动逻辑
main() {
    echo -e "${BLUE}当前工作目录: $(pwd)${NC}"
    echo -e "${BLUE}可用配置文件:${NC}"
    ls -la /workspace/xconnector-configs/ 2>/dev/null | sed 's/^/  /' || echo "  配置目录不存在"

    # 检查服务依赖
    check_service_dependencies

    # 设置基础环境
    setup_environment

    # 智能选择配置
    echo -e "\n${YELLOW}=== 配置选择 ===${NC}"
    local selected_config
    selected_config=$(smart_config_selection)

    echo -e "${GREEN}✓ 选择的配置: $selected_config${NC}"

    # 验证配置文件存在
    local config_file
    config_file=$(verify_config_file "$selected_config")
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # 检测图模块类型
    local graph_module
    graph_module=$(detect_graph_module "$config_file")

    # 显示配置信息
    show_config_info "$config_file" "$graph_module"

    # 构建启动命令
    local dynamo_cmd=(
        "/opt/dynamo/venv/bin/dynamo"
        "serve"
        "$graph_module"
        "-f"
        "$config_file"
    )

    echo -e "\n${GREEN}🚀 启动Dynamo服务器...${NC}"
    echo -e "${YELLOW}执行命令: ${dynamo_cmd[*]}${NC}"

    # 启动Dynamo
    exec "${dynamo_cmd[@]}"
}

# 错误处理
trap 'echo -e "\n${RED}启动过程中发生错误${NC}"; exit 1' ERR

# 执行主逻辑
main "$@"