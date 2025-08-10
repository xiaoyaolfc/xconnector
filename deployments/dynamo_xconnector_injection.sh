#!/bin/bash
# dynamo_xconnector_injection.sh
# 在Dynamo VllmWorker进程中自动注入XConnector

set -e

echo "🔧 Dynamo XConnector 自动注入脚本"
echo "=================================="

# 配置参数
WORKSPACE_DIR="/workspace"
XCONNECTOR_DIR="${WORKSPACE_DIR}/xconnector"
INIT_SCRIPT="${WORKSPACE_DIR}/xconnector_worker_init.py"
DYNAMO_CONFIG="${WORKSPACE_DIR}/examples/llm/configs/agg_with_xconnector.yaml"

# 1. 确保初始化脚本存在
if [ ! -f "$INIT_SCRIPT" ]; then
    echo "❌ XConnector初始化脚本不存在: $INIT_SCRIPT"
    exit 1
fi

echo "✅ 找到初始化脚本: $INIT_SCRIPT"

# 2. 创建环境变量文件供VllmWorker使用
cat > /tmp/xconnector_worker_env.sh << 'EOF'
#!/bin/bash
# XConnector Worker 环境变量

export XCONNECTOR_CONFIG_FILE="/workspace/configs/dynamo-xconnector-offline.yaml"
export ENABLE_XCONNECTOR="true"
export XCONNECTOR_LOG_DIR="/workspace/xconnector/log"
export XCONNECTOR_MODE="worker"
export XCONNECTOR_WORKER_TYPE="vllm"
export PYTHONPATH="${PYTHONPATH}:/workspace/xconnector"
export XCONNECTOR_LOG_LEVEL="INFO"
export XCONNECTOR_DISABLE_ETCD="true"
export XCONNECTOR_OFFLINE_MODE="true"
EOF

echo "✅ 创建了环境变量文件: /tmp/xconnector_worker_env.sh"

# 3. 创建Python初始化钩子
cat > /tmp/xconnector_init_hook.py << 'EOF'
"""
XConnector初始化钩子 - 在Worker进程启动时自动执行
"""
import os
import sys

def initialize_xconnector():
    """自动初始化XConnector"""
    try:
        # 加载环境变量
        if os.path.exists('/tmp/xconnector_worker_env.sh'):
            import subprocess
            result = subprocess.run(['bash', '-c', 'source /tmp/xconnector_worker_env.sh && env'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '=' in line and (line.startswith('XCONNECTOR_') or line.startswith('PYTHONPATH=')):
                        key, value = line.split('=', 1)
                        os.environ[key] = value

        # 执行初始化脚本
        init_script = '/workspace/xconnector_worker_init.py'
        if os.path.exists(init_script):
            exec(open(init_script).read())
            print("🎉 XConnector初始化钩子执行成功")
        else:
            print(f"❌ 初始化脚本不存在: {init_script}")

    except Exception as e:
        print(f"⚠️  XConnector初始化钩子执行失败: {e}")

# 检查是否在VllmWorker进程中
if any('VllmWorker' in arg for arg in sys.argv):
    initialize_xconnector()
EOF

echo "✅ 创建了初始化钩子: /tmp/xconnector_init_hook.py"

# 4. 修改Dynamo配置，添加worker初始化环境
create_enhanced_dynamo_config() {
    # 创建配置文件到正确的位置
    mkdir -p "${DYNAMO_HOME}/examples/llm/configs"

    cat > "${DYNAMO_HOME}/examples/llm/configs/agg_with_xconnector_enhanced.yaml" << 'EOF'
Common:
  model: /data/model/DeepSeek-R1-Distill-Llama-8B
  block-size: 32
  max-model-len: 8192

Frontend:
  served_model_name: DeepSeek-R1-Distill-Llama-8B
  endpoint: dynamo.Processor.chat/completions
  port: 8000

Processor:
  router: round-robin
  router-num-threads: 4
  common-configs: [model, block-size, max-model-len]

VllmWorker:
  enforce-eager: true
  max-num-batched-tokens: 8192
  enable-prefix-caching: true
  gpu-memory-utilization: 0.85
  tensor-parallel-size: 8
  ServiceArgs:
    workers: 1
    resources:
      gpu: '8'
    # 添加XConnector环境变量到Worker进程
    worker-env:
      - XCONNECTOR_CONFIG_FILE=/workspace/configs/dynamo-xconnector-offline.yaml
      - ENABLE_XCONNECTOR=true
      - XCONNECTOR_LOG_DIR=/workspace/xconnector/log
      - XCONNECTOR_MODE=worker
      - XCONNECTOR_WORKER_TYPE=vllm
      - PYTHONPATH=${PYTHONPATH}:/workspace/xconnector
      - XCONNECTOR_LOG_LEVEL=INFO
      - XCONNECTOR_DISABLE_ETCD=true
      - XCONNECTOR_OFFLINE_MODE=true
  common-configs: [model, block-size, max-model-len]

Planner:
  environment: local
  no-operation: true
EOF

    echo "✅ 创建了增强版Dynamo配置: ${DYNAMO_HOME}/examples/llm/configs/agg_with_xconnector_enhanced.yaml"
}

# 5. 创建启动包装脚本
create_dynamo_wrapper() {
    cat > "${WORKSPACE_DIR}/start_dynamo_with_xconnector.sh" << 'EOF'
#!/bin/bash
# Dynamo with XConnector 启动包装脚本

set -e

echo "🚀 启动带XConnector集成的Dynamo服务"
echo "===================================="

# 设置基础环境
export PYTHONPATH="${PYTHONPATH}:/workspace/xconnector"

# 检查XConnector初始化脚本
if [ -f "/workspace/xconnector_worker_init.py" ]; then
    echo "✅ XConnector初始化脚本已就绪"
else
    echo "❌ XConnector初始化脚本缺失"
    exit 1
fi

# 检查配置文件 - 修正路径到正确的位置
CONFIG_FILE="./configs/agg_with_xconnector_enhanced.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Dynamo配置文件不存在: $CONFIG_FILE"
    echo "请先运行: ./dynamo_xconnector_injection.sh"
    exit 1
fi

echo "✅ 使用配置文件: $CONFIG_FILE"

# 在启动前运行XConnector初始化测试
echo "🔍 测试XConnector初始化..."
python3 /workspace/xconnector_worker_init.py
if [ $? -eq 0 ]; then
    echo "✅ XConnector初始化测试通过"
else
    echo "⚠️  XConnector初始化测试有问题，但继续启动"
fi

# 启动Dynamo - 使用正确的目录和路径
echo "🚀 启动Dynamo服务..."

# 切换到正确的目录 - 这很关键！
cd $DYNAMO_HOME/examples/llm

# 使用标准启动方式
dynamo serve graphs.agg:Frontend -f "$CONFIG_FILE"
EOF

    chmod +x "${WORKSPACE_DIR}/start_dynamo_with_xconnector.sh"
    echo "✅ 创建了Dynamo启动包装脚本"
}

# 6. 创建Worker进程监控脚本
create_worker_monitor() {
    cat > "${WORKSPACE_DIR}/monitor_worker_xconnector.sh" << 'EOF'
#!/bin/bash
# VllmWorker XConnector 监控脚本

echo "🔍 监控VllmWorker进程中的XConnector状态"
echo "========================================"

# 查找VllmWorker进程
WORKER_PID=$(ps aux | grep VllmWorker | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$WORKER_PID" ]; then
    echo "❌ 未找到VllmWorker进程"
    exit 1
fi

echo "✅ 找到VllmWorker进程: PID=$WORKER_PID"

# 检查进程环境变量
echo -e "\n🌍 Worker进程环境变量:"
cat /proc/$WORKER_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "(XCONNECTOR|PYTHONPATH)" || echo "   没有XConnector相关环境变量"

# 检查进程工作目录
echo -e "\n📂 Worker进程工作目录:"
readlink /proc/$WORKER_PID/cwd 2>/dev/null || echo "   无法读取工作目录"

# 检查进程命令行
echo -e "\n📋 Worker进程命令行:"
cat /proc/$WORKER_PID/cmdline 2>/dev/null | tr '\0' ' ' || echo "   无法读取命令行"

# 尝试在Worker进程中执行XConnector检查
echo -e "\n🔧 尝试在Worker进程中注入XConnector检查..."
python3 -c "
import os
import sys

# 模拟Worker环境
sys.path.insert(0, '/workspace/xconnector')

try:
    # 设置环境变量
    os.environ['XCONNECTOR_CONFIG_FILE'] = '/workspace/configs/dynamo-xconnector-offline.yaml'
    os.environ['ENABLE_XCONNECTOR'] = 'true'

    # 测试导入
    from integrations.dynamo.autopatch import get_integration_status
    status = get_integration_status()

    print('XConnector状态:')
    for key, value in status.items():
        icon = '✅' if value else '❌'
        print(f'   {icon} {key}: {value}')

except Exception as e:
    print(f'❌ 检查失败: {e}')
"
EOF

    chmod +x "${WORKSPACE_DIR}/monitor_worker_xconnector.sh"
    echo "✅ 创建了Worker监控脚本"
}

# 执行所有创建步骤
echo -e "\n📝 创建所有必要文件..."
create_enhanced_dynamo_config
create_dynamo_wrapper
create_worker_monitor

echo -e "\n🎉 Dynamo XConnector 注入脚本配置完成！"
echo ""
echo "📖 使用方法:"
echo "1. 启动服务: ./start_dynamo_with_xconnector.sh"
echo "2. 监控状态: ./monitor_worker_xconnector.sh"
echo ""
echo "📋 创建的文件:"
echo "   - /tmp/xconnector_worker_env.sh (环境变量)"
echo "   - /tmp/xconnector_init_hook.py (初始化钩子)"
echo "   - agg_with_xconnector_enhanced.yaml (增强配置)"
echo "   - start_dynamo_with_xconnector.sh (启动脚本)"
echo "   - monitor_worker_xconnector.sh (监控脚本)"