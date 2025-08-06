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
    'etcd3',
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
# XConnector 集成配置文件 - Dynamo 专用 (离线版本)
xconnector:
  enabled: true
  mode: "embedded"

  # 离线模式配置
  offline_mode: true

  # 服务发现配置
  etcd:
    endpoints: ["http://etcd-server:2379"]
    timeout: 5
    enabled: true  # 如果 etcd3 包不可用会自动禁用

  nats:
    url: "nats://nats-server:4222"
    timeout: 5
    enabled: true  # 如果 nats 包不可用会自动禁用

  # 缓存配置
  cache:
    provider: "memory"  # 离线模式只支持内存缓存
    max_size: "1GB"
    ttl: 3600

  # KV传输配置
  kv_transfer:
    connector: "DynamoNixlConnector"
    batch_size: 1000
    compression: false  # 离线模式禁用压缩以减少依赖

  # 日志配置
  logging:
    level: "INFO"
    file: "/workspace/logs/xconnector.log"
    console: true

  # 性能配置
  performance:
    async_workers: 2
    queue_size: 1000
    timeout: 30

  # 故障处理
  fault_tolerance:
    graceful_degradation: true
    retry_attempts: 3
    retry_delay: 1.0
    fail_on_error: false
    offline_fallback: true  # 离线回退模式

  # 监控配置
  monitoring:
    enable_metrics: false  # 离线模式禁用复杂监控
    health_check_interval: 30
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

# 修改服务连通性检查函数
def check_service_multiple_hosts(service_name, port, hosts):
    \"\"\"尝试多个主机名/IP\"\"\"
    for host in hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f'{service_name}: ✅ (via {host})')
                return True
        except Exception as e:
            continue

    print(f'{service_name}: ❌ (所有地址都不可用)')
    return False

# 检查服务 - 尝试多个地址
etcd_hosts = ['127.0.0.1', 'localhost', 'etcd', 'etcd-server']
nats_hosts = ['127.0.0.1', 'localhost', 'nats', 'nats-server']

etcd_ok = check_service_multiple_hosts('etcd', 2379, etcd_hosts)
nats_ok = check_service_multiple_hosts('NATS', 4222, nats_hosts)

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
echo "2. dynamo serve graphs.agg:Frontend -f ./configs/agg_with_xconnector.yaml"
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
echo ""#!/bin/bash
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
    'etcd3',
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
# XConnector 集成配置文件 - Dynamo 专用 (离线版本)
xconnector:
  enabled: true
  mode: "embedded"

  # 离线模式配置
  offline_mode: true

  # 服务发现配置
  etcd:
    endpoints: ["http://etcd-server:2379"]
    timeout: 5
    enabled: true  # 如果 etcd3 包不可用会自动禁用

  nats:
    url: "nats://nats-server:4222"
    timeout: 5
    enabled: true  # 如果 nats 包不可用会自动禁用

  # 缓存配置
  cache:
    provider: "memory"  # 离线模式只支持内存缓存
    max_size: "1GB"
    ttl: 3600

  # KV传输配置
  kv_transfer:
    connector: "DynamoNixlConnector"
    batch_size: 1000
    compression: false  # 离线模式禁用压缩以减少依赖

  # 日志配置
  logging:
    level: "INFO"
    file: "/workspace/logs/xconnector.log"
    console: true

  # 性能配置
  performance:
    async_workers: 2
    queue_size: 1000
    timeout: 30

  # 故障处理
  fault_tolerance:
    graceful_degradation: true
    retry_attempts: 3
    retry_delay: 1.0
    fail_on_error: false
    offline_fallback: true  # 离线回退模式

  # 监控配置
  monitoring:
    enable_metrics: false  # 离线模式禁用复杂监控
    health_check_interval: 30
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
echo "2. dynamo serve graphs.agg:Frontend -f ./configs/agg_with_xconnector.yaml"
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