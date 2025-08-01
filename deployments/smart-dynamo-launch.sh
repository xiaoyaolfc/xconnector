#!/bin/bash
# 智能 Dynamo 启动脚本 - 根据配置文件自动选择 XConnector 集成版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置映射表
declare -A CONFIG_MAPPING=(
    ["agg.yaml"]="agg_with_xconnector.yaml"
    ["agg_router.yaml"]="agg_router_with_xconnector.yaml"
    ["disagg.yaml"]="disagg_with_xconnector_simple.yaml"
    ["disagg_router.yaml"]="disagg_router_with_xconnector.yaml"

    # 支持完整路径匹配
    ["./configs/agg.yaml"]="agg_with_xconnector.yaml"
    ["./configs/agg_router.yaml"]="agg_router_with_xconnector.yaml"
    ["./configs/disagg.yaml"]="disagg_with_xconnector_simple.yaml"
    ["./configs/disagg_router.yaml"]="disagg_router_with_xconnector.yaml"

    # 支持绝对路径匹配（通过文件名提取）
)

# XConnector 配置文件目录
XCONNECTOR_CONFIG_DIR="/workspace/xconnector-integration/configs"

echo -e "${GREEN}=== 智能 Dynamo + XConnector 启动器 ===${NC}"

# 解析命令行参数
parse_arguments() {
    local args=("$@")
    local config_file=""
    local graph_module=""

    # 查找 -f 参数
    for i in "${!args[@]}"; do
        if [[ "${args[i]}" == "-f" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
            config_file="${args[$((i+1))]}"
            break
        fi
    done

    # 查找图模块参数
    for arg in "${args[@]}"; do
        if [[ "$arg" =~ graphs\.(.*):Frontend ]]; then
            graph_module="${BASH_REMATCH[1]}"
            break
        fi
    done

    echo "$config_file|$graph_module"
}

# 获取配置文件名（从路径中提取）
get_config_filename() {
    local config_path="$1"
    basename "$config_path"
}

# 选择对应的 XConnector 配置
select_xconnector_config() {
    local original_config="$1"
    local config_filename
    local xconnector_config

    # 首先尝试直接匹配路径
    if [[ -n "${CONFIG_MAPPING[$original_config]}" ]]; then
        xconnector_config="${CONFIG_MAPPING[$original_config]}"
    else
        # 提取文件名再匹配
        config_filename=$(get_config_filename "$original_config")
        xconnector_config="${CONFIG_MAPPING[$config_filename]}"
    fi

    if [[ -z "$xconnector_config" ]]; then
        echo -e "${YELLOW}⚠ 未找到对应的 XConnector 配置，使用默认配置${NC}"
        xconnector_config="disagg_with_xconnector.yaml"  # 默认配置
    fi

    echo "$xconnector_config"
}

# 检查 XConnector 服务状态
check_xconnector_service() {
    echo -e "${YELLOW}检查 XConnector 服务状态...${NC}"

    if curl -f -s http://xconnector-service:8081/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ XConnector 服务正常${NC}"
        return 0
    else
        echo -e "${RED}✗ XConnector 服务不可用${NC}"
        echo -e "${YELLOW}提示: 请确保 XConnector 服务已启动${NC}"
        return 1
    fi
}

# 检查服务依赖
check_service_dependencies() {
    echo -e "${YELLOW}检查服务依赖...${NC}"

    local services=(
        "etcd:http://etcd:2379/health"
        "nats:http://nats:8222/"
        "xconnector:http://xconnector-service:8081/health"
    )

    local all_ok=true

    for service_info in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_info"
        echo -n "  检查 $name..."

        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
        else
            echo -e " ${RED}✗${NC}"
            all_ok=false
        fi
    done

    if [[ "$all_ok" == "true" ]]; then
        echo -e "${GREEN}✓ 所有依赖服务正常${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ 部分服务不可用，但继续启动${NC}"
        return 0  # 不阻止启动，让用户决定
    fi
}

# 创建临时配置文件
create_temp_config() {
    local xconnector_config="$1"
    local temp_config="/tmp/dynamo_config_$(date +%s).yaml"

    # 配置文件路径（优先查找容器内路径，然后是本地路径）
    local config_paths=(
        "/workspace/xconnector-integration/configs/$xconnector_config"
        "./configs/$xconnector_config"
        "../configs/$xconnector_config"
        "../../integrations/dynamo/configs/$xconnector_config"
    )

    for config_path in "${config_paths[@]}"; do
        if [[ -f "$config_path" ]]; then
            cp "$config_path" "$temp_config"
            echo "$temp_config"
            return 0
        fi
    done

    echo -e "${RED}✗ 找不到 XConnector 配置文件: $xconnector_config${NC}"
    return 1
}

# 显示配置信息
show_config_info() {
    local original_config="$1"
    local xconnector_config="$2"
    local temp_config="$3"

    echo -e "\n${BLUE}=== 配置选择结果 ===${NC}"
    echo -e "原始配置: ${YELLOW}$original_config${NC}"
    echo -e "XConnector 配置: ${YELLOW}$xconnector_config${NC}"
    echo -e "临时配置路径: ${YELLOW}$temp_config${NC}"
    echo -e "包含服务地址: ${GREEN}etcd:2379, nats:4222${NC}"
}

# 清理临时文件
cleanup() {
    if [[ -n "$TEMP_CONFIG" ]] && [[ -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
        echo -e "\n${BLUE}清理临时配置文件${NC}"
    fi
}

# 设置清理陷阱
trap cleanup EXIT

# 主启动逻辑
main() {
    local args=("$@")

    # 检查参数
    if [[ $# -lt 3 ]]; then
        echo -e "${RED}用法: $0 serve graphs.<config>:Frontend -f <config_file> [其他参数]${NC}"
        echo -e "${YELLOW}示例: $0 serve graphs.agg:Frontend -f ./configs/agg.yaml${NC}"
        exit 1
    fi

    # 解析参数
    local parse_result
    parse_result=$(parse_arguments "${args[@]}")
    IFS='|' read -r original_config graph_module <<< "$parse_result"

    if [[ -z "$original_config" ]]; then
        echo -e "${RED}✗ 未找到配置文件参数 (-f)${NC}"
        exit 1
    fi

    echo -e "${BLUE}检测到配置: $original_config${NC}"
    echo -e "${BLUE}检测到图模块: $graph_module${NC}"

    # 检查服务依赖
    check_service_dependencies

    # 选择对应的 XConnector 配置
    local xconnector_config
    xconnector_config=$(select_xconnector_config "$original_config")
    echo -e "${GREEN}选择 XConnector 配置: $xconnector_config${NC}"

    # 创建临时配置文件
    local temp_config
    temp_config=$(create_temp_config "$xconnector_config")
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # 设置全局变量供清理使用
    TEMP_CONFIG="$temp_config"

    # 显示配置信息
    show_config_info "$original_config" "$xconnector_config" "$temp_config"

    # 替换原始参数中的配置文件路径
    local new_args=()
    for i in "${!args[@]}"; do
        if [[ "${args[i]}" == "-f" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
            new_args+=("${args[i]}")
            new_args+=("$temp_config")
            ((i++))  # 跳过下一个参数（原配置文件路径）
        else
            new_args+=("${args[i]}")
        fi
    done

    echo -e "\n${GREEN}启动 Dynamo with XConnector...${NC}"
    echo -e "${YELLOW}执行命令: dynamo ${new_args[*]}${NC}"

    # 启动 Dynamo
    exec dynamo "${new_args[@]}"
}

# 帮助信息
show_help() {
    echo "智能 Dynamo + XConnector 启动器"
    echo ""
    echo "用法: $0 serve graphs.<config>:Frontend -f <config_file> [其他参数]"
    echo ""
    echo "支持的配置映射:"
    echo "  agg.yaml          -> agg_with_xconnector.yaml"
    echo "  agg_router.yaml   -> agg_router_with_xconnector.yaml"
    echo "  disagg.yaml       -> disagg_with_xconnector_simple.yaml"
    echo "  disagg_router.yaml -> disagg_router_with_xconnector.yaml"
    echo ""
    echo "示例:"
    echo "  $0 serve graphs.agg:Frontend -f ./configs/agg.yaml"
    echo "  $0 serve graphs.disagg:Frontend -f ./configs/disagg.yaml"
    echo "  $0 serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml"
    echo ""
    echo "功能:"
    echo "  - 自动检测配置文件类型"
    echo "  - 选择对应的 XConnector 集成配置"
    echo "  - 验证服务依赖状态"
    echo "  - 创建临时配置并自动清理"
}

# 检查帮助参数
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# 执行主逻辑
main "$@"