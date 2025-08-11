#!/bin/bash
# XConnector自动化验证包装脚本
# 在Dynamo启动后自动运行验证

set -e

# 配置
DYNAMO_URL="http://localhost:8000"
VALIDATION_DIR="/workspace/xconnector-validation"
MAX_WAIT_TIME=300  # 最大等待时间（秒）
CHECK_INTERVAL=5   # 检查间隔（秒）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"
}

# 等待Dynamo服务启动
wait_for_dynamo() {
    log_info "等待Dynamo服务启动..."

    local elapsed=0
    while [ $elapsed -lt $MAX_WAIT_TIME ]; do
        if curl -s -f "$DYNAMO_URL/health" >/dev/null 2>&1; then
            log_success "Dynamo服务已启动 (等待时间: ${elapsed}秒)"
            return 0
        fi

        if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            log_info "仍在等待Dynamo服务启动... (${elapsed}/${MAX_WAIT_TIME}秒)"
        fi

        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
    done

    log_error "Dynamo服务启动超时"
    return 1
}

# 运行XConnector环境初始化
init_xconnector() {
    log_info "初始化XConnector环境..."

    if [ -f "/workspace/xconnector/init-xconnector.sh" ]; then
        source /workspace/xconnector/init-xconnector.sh
        log_success "XConnector环境初始化完成"
    else
        log_warning "XConnector初始化脚本未找到"
    fi
}

# 运行快速验证
run_quick_validation() {
    log_info "运行快速验证..."

    cd "$VALIDATION_DIR" || {
        log_error "验证目录不存在: $VALIDATION_DIR"
        return 1
    }

    if python3 quick_status_check.py; then
        log_success "快速验证通过"
        return 0
    else
        log_warning "快速验证发现问题"
        return 1
    fi
}

# 运行完整验证
run_full_validation() {
    log_info "运行完整验证..."

    cd "$VALIDATION_DIR" || {
        log_error "验证目录不存在: $VALIDATION_DIR"
        return 1
    }

    if python3 dynamo_xconnector_validator.py; then
        log_success "完整验证通过"
        return 0
    else
        log_error "完整验证失败"
        return 1
    fi
}

# 启动监控
start_monitoring() {
    log_info "启动后台监控..."

    cd "$VALIDATION_DIR" || {
        log_error "验证目录不存在: $VALIDATION_DIR"
        return 1
    }

    # 检查是否已有监控进程
    if pgrep -f "continuous_monitor.py" >/dev/null; then
        log_warning "监控进程已存在，跳过启动"
        return 0
    fi

    # 启动监控进程
    nohup python3 continuous_monitor.py -i 60 > logs/monitor.log 2>&1 &
    local monitor_pid=$!

    # 等待一下确认启动成功
    sleep 3
    if ps -p $monitor_pid >/dev/null; then
        log_success "后台监控已启动 (PID: $monitor_pid)"
        echo $monitor_pid > logs/monitor.pid
    else
        log_error "监控启动失败"
        return 1
    fi
}

# 生成验证报告
generate_report() {
    local report_file="$VALIDATION_DIR/logs/validation_report_$(date +%Y%m%d_%H%M%S).txt"

    log_info "生成验证报告: $report_file"

    {
        echo "XConnector验证报告"
        echo "=================="
        echo "验证时间: $(date)"
        echo "Dynamo URL: $DYNAMO_URL"
        echo ""

        echo "系统信息:"
        echo "--------"
        echo "操作系统: $(uname -a)"
        echo "Python版本: $(python3 --version)"
        echo ""

        echo "服务状态:"
        echo "--------"
        echo -n "Dynamo服务: "
        if curl -s -f "$DYNAMO_URL/health" >/dev/null 2>&1; then
            echo "✅ 运行中"
        else
            echo "❌ 未运行"
        fi

        echo -n "etcd服务: "
        if nc -z localhost 2379 2>/dev/null; then
            echo "✅ 可用"
        else
            echo "❌ 不可用"
        fi

        echo -n "NATS服务: "
        if nc -z localhost 4222 2>/dev/null; then
            echo "✅ 可用"
        else
            echo "❌ 不可用"
        fi

        echo ""
        echo "验证结果详情请查看上述日志输出"

    } > "$report_file"

    log_success "验证报告已生成: $report_file"
}

# 显示状态总览
show_status_overview() {
    echo ""
    echo "🔍 XConnector状态总览"
    echo "===================="

    # 服务状态
    echo "📡 服务状态:"
    services=("Dynamo:8000" "XConnector:8081" "etcd:2379" "NATS:4222")
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)

        echo -n "   $name: "
        if nc -z localhost $port 2>/dev/null; then
            echo "✅"
        else
            echo "❌"
        fi
    done

    # 监控进程状态
    echo ""
    echo "📊 监控状态:"
    if pgrep -f "continuous_monitor.py" >/dev/null; then
        echo "   后台监控: ✅ 运行中"
    else
        echo "   后台监控: ❌ 未运行"
    fi

    # 快捷操作提示
    echo ""
    echo "🎯 快捷操作:"
    echo "   cd $VALIDATION_DIR"
    echo "   ./check     # 快速检查"
    echo "   ./validate  # 完整验证"
    echo "   ./monitor   # 监控管理"
    echo "   ./status    # 状态总览"
}

# 主函数
main() {
    local mode="auto"
    local enable_monitoring=true
    local generate_report_flag=true

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick-only)
                mode="quick"
                shift
                ;;
            --no-monitoring)
                enable_monitoring=false
                shift
                ;;
            --no-report)
                generate_report_flag=false
                shift
                ;;
            --wait-timeout)
                MAX_WAIT_TIME="$2"
                shift 2
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --quick-only      只运行快速验证"
                echo "  --no-monitoring   不启动后台监控"
                echo "  --no-report       不生成验证报告"
                echo "  --wait-timeout N  设置等待超时时间（秒）"
                echo "  -h, --help        显示帮助"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done

    echo "🚀 XConnector自动化验证"
    echo "======================"
    echo "模式: $mode"
    echo "监控: $([ "$enable_monitoring" = true ] && echo "启用" || echo "禁用")"
    echo "报告: $([ "$generate_report_flag" = true ] && echo "生成" || echo "跳过")"
    echo ""

    # 执行验证流程
    local validation_success=true

    # 1. 等待Dynamo服务
    if ! wait_for_dynamo; then
        log_error "Dynamo服务不可用，终止验证"
        exit 1
    fi

    # 2. 初始化XConnector
    init_xconnector

    # 3. 运行验证
    if [ "$mode" = "quick" ]; then
        if ! run_quick_validation; then
            validation_success=false
        fi
    else
        # 先快速验证
        if ! run_quick_validation; then
            validation_success=false
            log_warning "快速验证失败，仍将继续完整验证"
        fi

        # 再完整验证
        if ! run_full_validation; then
            validation_success=false
        fi
    fi

    # 4. 启动监控
    if [ "$enable_monitoring" = true ]; then
        start_monitoring
    fi

    # 5. 生成报告
    if [ "$generate_report_flag" = true ]; then
        generate_report
    fi

    # 6. 显示状态总览
    show_status_overview

    # 7. 输出最终结果
    echo ""
    if [ "$validation_success" = true ]; then
        log_success "🎉 XConnector验证全部通过！"
        exit 0
    else
        log_warning "⚠️  验证过程中发现问题，请查看详细日志"
        exit 1
    fi
}

# 运行主函数
main "$@"