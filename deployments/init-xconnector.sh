#!/bin/bash
# init-xconnector.sh
# XConnector 容器内初始化脚本 (离线版本)
# 使用方法：在容器启动后执行 source /workspace/xconnector/init-xconnector.sh

set -e

echo "🚀 初始化 XConnector 集成 (离线模式)..."

# 检查是否启用
if [ "${XCONNECTOR_ENABLED:-false}" != "true" ]; then
    echo "⚠️  XConnector 未启用，跳过初始化"
    return 0 2>/dev/null || exit 0
fi

# 检查 XConnector 代码是否存在
if [ ! -d "/workspace/xconnector" ]; then
    echo "❌ 错误: XConnector 代码目录不存在: /workspace/xconnector"
    return 1 2>/dev/null || exit 1
fi

echo "📁 XConnector 代码路径: /workspace/xconnector"

# 设置 Python 路径
export PYTHONPATH="/workspace/xconnector:/workspace:${PYTHONPATH}"
echo "🐍 设置 PYTHONPATH: $PYTHONPATH"

# 检查依赖（仅检查，不安装）
echo "📦 检查 Python 依赖可用性..."

python3 -c "
import sys
import importlib

# 必需依赖列表
required_deps = [
    'yaml',
    'asyncio',
    'logging',
    'threading',
    'json',
    'os',
    'sys'
]

# 可选依赖列表（缺失时会优雅降级）
optional_deps = [
    'aiofiles',
    'aiohttp',
    'nats',
    'redis'
]

print('📋 检查必需依赖:')
missing_required = []
for dep in required_deps:
    try:
        importlib.import_module(dep)
        print(f'  ✅ {dep}')
    except ImportError:
        print(f'  ❌ {dep} (缺失)')
        missing_required.append(dep)

print('📋 检查可选依赖:')
missing_optional = []
for dep in optional_deps:
    try:
        importlib.import_module(dep)
        print(f'  ✅ {dep}')
    except ImportError:
        print(f'  ⚠️  {dep} (缺失，将优雅降级)')
        missing_optional.append(dep)

if missing_required:
    print(f'❌ 关键依赖缺失: {missing_required}')
    print('⚠️  XConnector 可能无法正常工作')
else:
    print('✅ 必需依赖检查通过')

if missing_optional:
    print(f'⚠️  可选依赖缺失: {missing_optional}')
    print('ℹ️  相关功能将被禁用，但核心功能仍可使用')
"

# 创建必要的目录
mkdir -p /workspace/logs
mkdir -p /workspace/configs

# 创建默认配置文件（如果不存在）
CONFIG_FILE="/workspace/configs/dynamo-xconnector.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚙️  创建默认配置文件: $CONFIG_FILE"
    cat > "$CONFIG_FILE" << 'EOF'
xconnector:
  enabled: true
  mode: embedded
  offline_mode: true

  etcd:
    endpoints: ['http://127.0.0.1:2379']
    timeout: 5
    enabled: true

  nats:
    url: 'nats://127.0.0.1:4222'
    timeout: 5
    enabled: true

  cache:
    provider: memory
    max_size: 1GB
    ttl: 3600

  kv_transfer:
    connector: DynamoNixlConnector
    batch_size: 1000
    compression: false

  logging:
    level: INFO
    file: /workspace/logs/xconnector.log
    console: true

  fault_tolerance:
    graceful_degradation: true
    retry_attempts: 3
    offline_fallback: true

sdk:
  mode: embedded
  enable_kv_cache: true
  enable_distributed: true

  adapters:
    - name: lmcache
      type: cache
      class_path: xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter
      config:
        storage_backend: memory
        max_cache_size: 1024
        enable_compression: true
        block_size: 16
      enabled: true
      priority: 1
EOF
fi

# 测试导入（带错误处理）
echo "🧪 测试 XConnector 导入..."
python3 -c "
import sys
sys.path.insert(0, '/workspace/xconnector')

try:
    print('📦 导入 XConnector 模块...')
    import integrations.dynamo.autopatch
    print('✅ XConnector autopatch 导入成功')

    from integrations.dynamo.autopatch import get_integration_status
    status = get_integration_status()
    print(f'📊 集成状态: {status}')

    # 检查是否初始化成功
    if status.get('initialized', False):
        print('🎉 XConnector 初始化成功')
    else:
        print('⚠️  XConnector 部分功能可能受限')

except ImportError as e:
    print(f'❌ 导入失败: {e}')
    print('💡 提示: 检查代码路径和依赖')
except Exception as e:
    print(f'⚠️  初始化警告: {e}')
    print('ℹ️  XConnector 将在降级模式下运行')
"

# 简化的服务连通性检查（不依赖外部包）
echo "🔍 检查服务连通性 (简化版本)..."
python3 -c "
import socket
import sys

def check_service(host, port, name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f'{name}: ✅')
            return True
        else:
            print(f'{name}: ❌ (连接失败)')
            return False
    except Exception as e:
        print(f'{name}: ❌ (异常: {str(e)[:50]})')
        return False

# 检查服务
etcd_ok = check_service('etcd-server', 2379, 'etcd')
nats_ok = check_service('nats-server', 4222, 'NATS')

if etcd_ok and nats_ok:
    print('🌐 所有服务连通正常')
elif etcd_ok or nats_ok:
    print('⚠️  部分服务可用，XConnector 将适应性运行')
else:
    print('❌ 外部服务不可用，XConnector 将使用本地模式')
"

echo ""
echo "🎉 XConnector 离线初始化完成！"
echo ""
echo "📋 接下来的步骤："
echo "1. cd \$DYNAMO_HOME/examples/llm"
echo "2. dynamo serve graphs.agg:Frontend -f /workspace/examples/llm/configs/agg_with_xconnector.yaml"
echo ""
echo "💡 离线模式提示："
echo "- 配置文件: /workspace/configs/"
echo "- 日志文件: /workspace/logs/"
echo "- XConnector 代码: /workspace/xconnector/"
echo "- 模式: 离线优雅降级"
echo ""
echo "⚠️  注意事项："
echo "- 缺失的可选依赖将自动禁用相关功能"
echo "- 核心 Worker patch 功能不受影响"
echo "- 建议检查日志确认功能状态"
echo ""