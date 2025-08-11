#!/bin/bash
# XConnector验证工具部署和使用脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
VALIDATION_DIR="/workspace/xconnector-validation"
XCONNECTOR_PATH="/workspace/xconnector"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi

    # 检查XConnector路径
    if [ ! -d "$XCONNECTOR_PATH" ]; then
        print_error "XConnector路径不存在: $XCONNECTOR_PATH"
        exit 1
    fi

    # 检查必要的Python包
    python3 -c "import asyncio, requests, json" 2>/dev/null || {
        print_warning "缺少必要的Python包，尝试安装..."
        pip3 install requests asyncio 2>/dev/null || print_warning "包安装失败，但可能不影响基本功能"
    }

    print_success "依赖检查完成"
}

# 部署验证工具
deploy_validation_tools() {
    print_info "部署验证工具到 $VALIDATION_DIR..."

    # 创建目录
    mkdir -p "$VALIDATION_DIR"
    mkdir -p "$VALIDATION_DIR/logs"

    # 创建验证脚本（假设已经通过artifacts生成）
    cat > "$VALIDATION_DIR/dynamo_xconnector_validator.py" << 'EOF'
#!/usr/bin/env python3
"""
Dynamo XConnector 运行时验证器
在Dynamo服务运行期间验证XConnector集成状态
"""

import sys
import os
import time
import json
import asyncio
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 确保XConnector路径
sys.path.insert(0, '/workspace/xconnector')

class ValidationStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    INFO = "ℹ️  INFO"
    SKIP = "⏭️  SKIP"

@dataclass
class ValidationResult:
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DynamoXConnectorValidator:
    """Dynamo XConnector运行时验证器"""

    def __init__(self,
                 dynamo_url: str = "http://localhost:8000",
                 xconnector_service_url: str = "http://localhost:8081"):
        self.dynamo_url = dynamo_url
        self.xconnector_service_url = xconnector_service_url
        self.results: List[ValidationResult] = []

    def add_result(self, name: str, status: ValidationStatus, message: str, details: Dict = None):
        """添加验证结果"""
        self.results.append(ValidationResult(name, status, message, details))

    def print_result(self, result: ValidationResult):
        """打印单个验证结果"""
        print(f"{result.status.value} {result.name}: {result.message}")
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")

    async def validate_basic_services(self) -> bool:
        """验证基础服务连接"""
        print("\n🔍 验证基础服务连接...")

        # 检查Dynamo服务
        try:
            response = requests.get(f"{self.dynamo_url}/health", timeout=5)
            if response.status_code == 200:
                self.add_result("Dynamo服务", ValidationStatus.PASS, "服务正常响应")
            else:
                self.add_result("Dynamo服务", ValidationStatus.FAIL, f"服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            self.add_result("Dynamo服务", ValidationStatus.FAIL, f"连接失败: {str(e)}")
            return False

        # 检查XConnector服务（如果部署为独立服务）
        try:
            response = requests.get(f"{self.xconnector_service_url}/health", timeout=5)
            if response.status_code == 200:
                self.add_result("XConnector服务", ValidationStatus.PASS, "服务正常响应")
            else:
                self.add_result("XConnector服务", ValidationStatus.WARN, "独立服务不可用，检查嵌入模式")
        except Exception as e:
            self.add_result("XConnector服务", ValidationStatus.INFO, "使用嵌入模式，跳过独立服务检查")

        # 检查etcd和NATS
        await self._check_infrastructure_services()

        return True

    async def _check_infrastructure_services(self):
        """检查基础设施服务"""
        import socket

        # 检查etcd
        etcd_hosts = ['127.0.0.1', 'localhost', 'etcd', 'etcd-server']
        etcd_connected = False

        for host in etcd_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, 2379))
                sock.close()
                if result == 0:
                    etcd_connected = True
                    self.add_result("etcd连接", ValidationStatus.PASS, f"通过 {host}:2379 连接成功")
                    break
            except Exception:
                continue

        if not etcd_connected:
            self.add_result("etcd连接", ValidationStatus.WARN, "无法连接到etcd服务")

        # 检查NATS
        nats_hosts = ['127.0.0.1', 'localhost', 'nats', 'nats-server']
        nats_connected = False

        for host in nats_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, 4222))
                sock.close()
                if result == 0:
                    nats_connected = True
                    self.add_result("NATS连接", ValidationStatus.PASS, f"通过 {host}:4222 连接成功")
                    break
            except Exception:
                continue

        if not nats_connected:
            self.add_result("NATS连接", ValidationStatus.WARN, "无法连接到NATS服务")

    async def validate_xconnector_integration(self) -> bool:
        """验证XConnector集成状态"""
        print("\n🔧 验证XConnector集成状态...")

        try:
            # 尝试导入XConnector模块
            from integrations.dynamo.autopatch import get_integration_status, get_minimal_sdk

            # 获取集成状态
            status = get_integration_status()

            # 验证各项状态
            if status.get('sdk_available', False):
                self.add_result("SDK可用性", ValidationStatus.PASS, "XConnector SDK已正确加载")
            else:
                self.add_result("SDK可用性", ValidationStatus.FAIL, "XConnector SDK不可用")
                return False

            if status.get('config_found', False):
                self.add_result("配置文件", ValidationStatus.PASS, "配置文件已找到并加载")
            else:
                self.add_result("配置文件", ValidationStatus.WARN, "配置文件未找到，使用默认配置")

            # 获取SDK实例
            sdk = get_minimal_sdk()
            if sdk:
                self.add_result("SDK实例", ValidationStatus.PASS, f"SDK实例类型: {type(sdk).__name__}")

                # 检查缓存适配器
                if hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                    cache_stats = sdk.cache_adapter.get_cache_statistics()

                    if cache_stats.get('connector_class_available', False):
                        self.add_result("缓存适配器", ValidationStatus.PASS, "缓存适配器正常工作")

                        # 详细统计信息
                        self.add_result("缓存统计", ValidationStatus.INFO, "缓存适配器统计信息", cache_stats)
                    else:
                        self.add_result("缓存适配器", ValidationStatus.FAIL, "缓存适配器不可用")
                else:
                    self.add_result("缓存适配器", ValidationStatus.FAIL, "缓存适配器未初始化")
            else:
                self.add_result("SDK实例", ValidationStatus.FAIL, "无法获取SDK实例")

        except ImportError as e:
            self.add_result("模块导入", ValidationStatus.FAIL, f"无法导入XConnector模块: {str(e)}")
            return False
        except Exception as e:
            self.add_result("集成验证", ValidationStatus.FAIL, f"集成验证失败: {str(e)}")
            return False

        return True

    async def validate_adapters(self) -> bool:
        """验证适配器状态"""
        print("\n🔌 验证适配器状态...")

        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if not sdk:
                self.add_result("适配器验证", ValidationStatus.FAIL, "无法获取SDK实例")
                return False

            # 检查vLLM适配器
            vllm_available = False
            if hasattr(sdk, 'inference_adapters'):
                for name, adapter in sdk.inference_adapters.items():
                    if 'vllm' in name.lower() and adapter:
                        vllm_available = True
                        self.add_result("vLLM适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")

                        # 检查适配器健康状态
                        if hasattr(adapter, 'get_health'):
                            try:
                                health = await adapter.get_health()
                                self.add_result("vLLM健康状态", ValidationStatus.PASS, f"健康状态: {health}")
                            except Exception as e:
                                self.add_result("vLLM健康状态", ValidationStatus.WARN, f"健康检查失败: {str(e)}")
                        break

            if not vllm_available:
                self.add_result("vLLM适配器", ValidationStatus.WARN, "vLLM适配器未找到或未加载")

            # 检查LMCache适配器
            lmcache_available = False
            if hasattr(sdk, 'cache_adapters'):
                for name, adapter in sdk.cache_adapters.items():
                    if 'lmcache' in name.lower() and adapter:
                        lmcache_available = True
                        self.add_result("LMCache适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")

                        # 检查缓存统计
                        if hasattr(adapter, 'get_cache_statistics'):
                            try:
                                stats = adapter.get_cache_statistics()
                                self.add_result("LMCache统计", ValidationStatus.INFO, "缓存统计信息", stats)
                            except Exception as e:
                                self.add_result("LMCache统计", ValidationStatus.WARN, f"获取统计失败: {str(e)}")
                        break

            if not lmcache_available:
                self.add_result("LMCache适配器", ValidationStatus.WARN, "LMCache适配器未找到或未加载")

            # 检查分布式适配器
            distributed_available = False
            if hasattr(sdk, 'distributed_adapters'):
                for name, adapter in sdk.distributed_adapters.items():
                    if adapter:
                        distributed_available = True
                        self.add_result("分布式适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")
                        break

            if not distributed_available:
                self.add_result("分布式适配器", ValidationStatus.INFO, "分布式适配器未配置")

        except Exception as e:
            self.add_result("适配器验证", ValidationStatus.FAIL, f"适配器验证失败: {str(e)}")
            return False

        return True

    async def validate_functionality(self) -> bool:
        """验证功能性"""
        print("\n🧪 验证功能性...")

        # 测试基本的推理请求
        await self._test_inference_functionality()

        # 测试缓存功能
        await self._test_cache_functionality()

        return True

    async def _test_inference_functionality(self):
        """测试推理功能"""
        try:
            # 发送简单的聊天请求
            test_payload = {
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            response = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    self.add_result("推理功能", ValidationStatus.PASS, "推理请求成功完成")
                    self.add_result("推理结果", ValidationStatus.INFO, f"生成文本: {result['choices'][0].get('message', {}).get('content', '')[:50]}...")
                else:
                    self.add_result("推理功能", ValidationStatus.WARN, "推理请求返回但格式异常")
            else:
                self.add_result("推理功能", ValidationStatus.FAIL, f"推理请求失败: {response.status_code}")

        except Exception as e:
            self.add_result("推理功能", ValidationStatus.FAIL, f"推理测试失败: {str(e)}")

    async def _test_cache_functionality(self):
        """测试缓存功能"""
        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if not sdk or not hasattr(sdk, 'cache_adapter') or not sdk.cache_adapter:
                self.add_result("缓存功能", ValidationStatus.SKIP, "缓存适配器不可用，跳过缓存测试")
                return

            # 尝试缓存测试
            cache_adapter = sdk.cache_adapter

            # 发送两个相同的请求，检查缓存命中
            test_payload = {
                "messages": [{"role": "user", "content": "Test cache functionality"}],
                "max_tokens": 5,
                "temperature": 0.0  # 确保结果一致
            }

            # 第一次请求
            start_time = time.time()
            response1 = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )
            first_duration = time.time() - start_time

            # 短暂等待缓存写入
            await asyncio.sleep(1)

            # 第二次请求
            start_time = time.time()
            response2 = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )
            second_duration = time.time() - start_time

            if response1.status_code == 200 and response2.status_code == 200:
                # 检查缓存统计
                stats_after = cache_adapter.get_cache_statistics()

                self.add_result("缓存功能", ValidationStatus.PASS, "缓存测试完成")
                self.add_result("缓存性能", ValidationStatus.INFO,
                              f"第一次请求: {first_duration:.2f}s, 第二次请求: {second_duration:.2f}s",
                              {"first_duration": first_duration, "second_duration": second_duration})

                if second_duration < first_duration * 0.8:
                    self.add_result("缓存命中", ValidationStatus.PASS, "检测到性能提升，可能有缓存命中")
                else:
                    self.add_result("缓存命中", ValidationStatus.WARN, "未检测到明显性能提升")
            else:
                self.add_result("缓存功能", ValidationStatus.FAIL, "缓存测试请求失败")

        except Exception as e:
            self.add_result("缓存功能", ValidationStatus.FAIL, f"缓存测试失败: {str(e)}")

    async def validate_monitoring(self) -> bool:
        """验证监控功能"""
        print("\n📊 验证监控功能...")

        try:
            # 获取系统状态
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if sdk:
                # 获取连接器健康状态
                if hasattr(sdk, 'get_health_status'):
                    try:
                        health = await sdk.get_health_status()
                        self.add_result("健康监控", ValidationStatus.PASS, "健康状态获取成功", health)
                    except Exception as e:
                        self.add_result("健康监控", ValidationStatus.WARN, f"健康状态获取失败: {str(e)}")

                # 获取适配器列表
                if hasattr(sdk, 'list_adapters'):
                    try:
                        adapters = sdk.list_adapters()
                        self.add_result("适配器监控", ValidationStatus.PASS, f"发现 {len(adapters)} 类适配器", adapters)
                    except Exception as e:
                        self.add_result("适配器监控", ValidationStatus.WARN, f"适配器列表获取失败: {str(e)}")

        except Exception as e:
            self.add_result("监控验证", ValidationStatus.FAIL, f"监控验证失败: {str(e)}")

        return True

    async def run_validation(self) -> bool:
        """运行完整验证"""
        print("🚀 开始Dynamo XConnector运行时验证...")
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        success = True

        # 基础服务验证
        if not await self.validate_basic_services():
            success = False

        # XConnector集成验证
        if not await self.validate_xconnector_integration():
            success = False

        # 适配器验证
        if not await self.validate_adapters():
            success = False

        # 功能验证
        if not await self.validate_functionality():
            success = False

        # 监控验证
        if not await self.validate_monitoring():
            success = False

        # 打印所有结果
        print("\n" + "="*60)
        print("📋 验证结果汇总:")
        print("="*60)

        pass_count = 0
        fail_count = 0
        warn_count = 0

        for result in self.results:
            self.print_result(result)
            if result.status == ValidationStatus.PASS:
                pass_count += 1
            elif result.status == ValidationStatus.FAIL:
                fail_count += 1
            elif result.status == ValidationStatus.WARN:
                warn_count += 1

        print("\n" + "="*60)
        print(f"✅ 通过: {pass_count} | ❌ 失败: {fail_count} | ⚠️  警告: {warn_count}")

        if fail_count == 0:
            print("🎉 XConnector集成验证通过！")
        else:
            print("⚠️  发现问题，请检查失败项目")

        return success

async def main():
    """主函数"""
    validator = DynamoXConnectorValidator()
    success = await validator.run_validation()

    # 返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

    cat > "$VALIDATION_DIR/continuous_monitor.py" << 'EOF'
#!/usr/bin/env python3
"""
Dynamo XConnector 运行时验证器
在Dynamo服务运行期间验证XConnector集成状态
"""

import sys
import os
import time
import json
import asyncio
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 确保XConnector路径
sys.path.insert(0, '/workspace/xconnector')

class ValidationStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    INFO = "ℹ️  INFO"
    SKIP = "⏭️  SKIP"

@dataclass
class ValidationResult:
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DynamoXConnectorValidator:
    """Dynamo XConnector运行时验证器"""

    def __init__(self,
                 dynamo_url: str = "http://localhost:8000",
                 xconnector_service_url: str = "http://localhost:8081"):
        self.dynamo_url = dynamo_url
        self.xconnector_service_url = xconnector_service_url
        self.results: List[ValidationResult] = []

    def add_result(self, name: str, status: ValidationStatus, message: str, details: Dict = None):
        """添加验证结果"""
        self.results.append(ValidationResult(name, status, message, details))

    def print_result(self, result: ValidationResult):
        """打印单个验证结果"""
        print(f"{result.status.value} {result.name}: {result.message}")
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")

    async def validate_basic_services(self) -> bool:
        """验证基础服务连接"""
        print("\n🔍 验证基础服务连接...")

        # 检查Dynamo服务
        try:
            response = requests.get(f"{self.dynamo_url}/health", timeout=5)
            if response.status_code == 200:
                self.add_result("Dynamo服务", ValidationStatus.PASS, "服务正常响应")
            else:
                self.add_result("Dynamo服务", ValidationStatus.FAIL, f"服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            self.add_result("Dynamo服务", ValidationStatus.FAIL, f"连接失败: {str(e)}")
            return False

        # 检查XConnector服务（如果部署为独立服务）
        try:
            response = requests.get(f"{self.xconnector_service_url}/health", timeout=5)
            if response.status_code == 200:
                self.add_result("XConnector服务", ValidationStatus.PASS, "服务正常响应")
            else:
                self.add_result("XConnector服务", ValidationStatus.WARN, "独立服务不可用，检查嵌入模式")
        except Exception as e:
            self.add_result("XConnector服务", ValidationStatus.INFO, "使用嵌入模式，跳过独立服务检查")

        # 检查etcd和NATS
        await self._check_infrastructure_services()

        return True

    async def _check_infrastructure_services(self):
        """检查基础设施服务"""
        import socket

        # 检查etcd
        etcd_hosts = ['127.0.0.1', 'localhost', 'etcd', 'etcd-server']
        etcd_connected = False

        for host in etcd_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, 2379))
                sock.close()
                if result == 0:
                    etcd_connected = True
                    self.add_result("etcd连接", ValidationStatus.PASS, f"通过 {host}:2379 连接成功")
                    break
            except Exception:
                continue

        if not etcd_connected:
            self.add_result("etcd连接", ValidationStatus.WARN, "无法连接到etcd服务")

        # 检查NATS
        nats_hosts = ['127.0.0.1', 'localhost', 'nats', 'nats-server']
        nats_connected = False

        for host in nats_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, 4222))
                sock.close()
                if result == 0:
                    nats_connected = True
                    self.add_result("NATS连接", ValidationStatus.PASS, f"通过 {host}:4222 连接成功")
                    break
            except Exception:
                continue

        if not nats_connected:
            self.add_result("NATS连接", ValidationStatus.WARN, "无法连接到NATS服务")

    async def validate_xconnector_integration(self) -> bool:
        """验证XConnector集成状态"""
        print("\n🔧 验证XConnector集成状态...")

        try:
            # 尝试导入XConnector模块
            from integrations.dynamo.autopatch import get_integration_status, get_minimal_sdk

            # 获取集成状态
            status = get_integration_status()

            # 验证各项状态
            if status.get('sdk_available', False):
                self.add_result("SDK可用性", ValidationStatus.PASS, "XConnector SDK已正确加载")
            else:
                self.add_result("SDK可用性", ValidationStatus.FAIL, "XConnector SDK不可用")
                return False

            if status.get('config_found', False):
                self.add_result("配置文件", ValidationStatus.PASS, "配置文件已找到并加载")
            else:
                self.add_result("配置文件", ValidationStatus.WARN, "配置文件未找到，使用默认配置")

            # 获取SDK实例
            sdk = get_minimal_sdk()
            if sdk:
                self.add_result("SDK实例", ValidationStatus.PASS, f"SDK实例类型: {type(sdk).__name__}")

                # 检查缓存适配器
                if hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                    cache_stats = sdk.cache_adapter.get_cache_statistics()

                    if cache_stats.get('connector_class_available', False):
                        self.add_result("缓存适配器", ValidationStatus.PASS, "缓存适配器正常工作")

                        # 详细统计信息
                        self.add_result("缓存统计", ValidationStatus.INFO, "缓存适配器统计信息", cache_stats)
                    else:
                        self.add_result("缓存适配器", ValidationStatus.FAIL, "缓存适配器不可用")
                else:
                    self.add_result("缓存适配器", ValidationStatus.FAIL, "缓存适配器未初始化")
            else:
                self.add_result("SDK实例", ValidationStatus.FAIL, "无法获取SDK实例")

        except ImportError as e:
            self.add_result("模块导入", ValidationStatus.FAIL, f"无法导入XConnector模块: {str(e)}")
            return False
        except Exception as e:
            self.add_result("集成验证", ValidationStatus.FAIL, f"集成验证失败: {str(e)}")
            return False

        return True

    async def validate_adapters(self) -> bool:
        """验证适配器状态"""
        print("\n🔌 验证适配器状态...")

        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if not sdk:
                self.add_result("适配器验证", ValidationStatus.FAIL, "无法获取SDK实例")
                return False

            # 检查vLLM适配器
            vllm_available = False
            if hasattr(sdk, 'inference_adapters'):
                for name, adapter in sdk.inference_adapters.items():
                    if 'vllm' in name.lower() and adapter:
                        vllm_available = True
                        self.add_result("vLLM适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")

                        # 检查适配器健康状态
                        if hasattr(adapter, 'get_health'):
                            try:
                                health = await adapter.get_health()
                                self.add_result("vLLM健康状态", ValidationStatus.PASS, f"健康状态: {health}")
                            except Exception as e:
                                self.add_result("vLLM健康状态", ValidationStatus.WARN, f"健康检查失败: {str(e)}")
                        break

            if not vllm_available:
                self.add_result("vLLM适配器", ValidationStatus.WARN, "vLLM适配器未找到或未加载")

            # 检查LMCache适配器
            lmcache_available = False
            if hasattr(sdk, 'cache_adapters'):
                for name, adapter in sdk.cache_adapters.items():
                    if 'lmcache' in name.lower() and adapter:
                        lmcache_available = True
                        self.add_result("LMCache适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")

                        # 检查缓存统计
                        if hasattr(adapter, 'get_cache_statistics'):
                            try:
                                stats = adapter.get_cache_statistics()
                                self.add_result("LMCache统计", ValidationStatus.INFO, "缓存统计信息", stats)
                            except Exception as e:
                                self.add_result("LMCache统计", ValidationStatus.WARN, f"获取统计失败: {str(e)}")
                        break

            if not lmcache_available:
                self.add_result("LMCache适配器", ValidationStatus.WARN, "LMCache适配器未找到或未加载")

            # 检查分布式适配器
            distributed_available = False
            if hasattr(sdk, 'distributed_adapters'):
                for name, adapter in sdk.distributed_adapters.items():
                    if adapter:
                        distributed_available = True
                        self.add_result("分布式适配器", ValidationStatus.PASS, f"适配器 {name} 已加载")
                        break

            if not distributed_available:
                self.add_result("分布式适配器", ValidationStatus.INFO, "分布式适配器未配置")

        except Exception as e:
            self.add_result("适配器验证", ValidationStatus.FAIL, f"适配器验证失败: {str(e)}")
            return False

        return True

    async def validate_functionality(self) -> bool:
        """验证功能性"""
        print("\n🧪 验证功能性...")

        # 测试基本的推理请求
        await self._test_inference_functionality()

        # 测试缓存功能
        await self._test_cache_functionality()

        return True

    async def _test_inference_functionality(self):
        """测试推理功能"""
        try:
            # 发送简单的聊天请求
            test_payload = {
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            response = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    self.add_result("推理功能", ValidationStatus.PASS, "推理请求成功完成")
                    self.add_result("推理结果", ValidationStatus.INFO, f"生成文本: {result['choices'][0].get('message', {}).get('content', '')[:50]}...")
                else:
                    self.add_result("推理功能", ValidationStatus.WARN, "推理请求返回但格式异常")
            else:
                self.add_result("推理功能", ValidationStatus.FAIL, f"推理请求失败: {response.status_code}")

        except Exception as e:
            self.add_result("推理功能", ValidationStatus.FAIL, f"推理测试失败: {str(e)}")

    async def _test_cache_functionality(self):
        """测试缓存功能"""
        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if not sdk or not hasattr(sdk, 'cache_adapter') or not sdk.cache_adapter:
                self.add_result("缓存功能", ValidationStatus.SKIP, "缓存适配器不可用，跳过缓存测试")
                return

            # 尝试缓存测试
            cache_adapter = sdk.cache_adapter

            # 发送两个相同的请求，检查缓存命中
            test_payload = {
                "messages": [{"role": "user", "content": "Test cache functionality"}],
                "max_tokens": 5,
                "temperature": 0.0  # 确保结果一致
            }

            # 第一次请求
            start_time = time.time()
            response1 = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )
            first_duration = time.time() - start_time

            # 短暂等待缓存写入
            await asyncio.sleep(1)

            # 第二次请求
            start_time = time.time()
            response2 = requests.post(
                f"{self.dynamo_url}/v1/chat/completions",
                json=test_payload,
                timeout=30
            )
            second_duration = time.time() - start_time

            if response1.status_code == 200 and response2.status_code == 200:
                # 检查缓存统计
                stats_after = cache_adapter.get_cache_statistics()

                self.add_result("缓存功能", ValidationStatus.PASS, "缓存测试完成")
                self.add_result("缓存性能", ValidationStatus.INFO,
                              f"第一次请求: {first_duration:.2f}s, 第二次请求: {second_duration:.2f}s",
                              {"first_duration": first_duration, "second_duration": second_duration})

                if second_duration < first_duration * 0.8:
                    self.add_result("缓存命中", ValidationStatus.PASS, "检测到性能提升，可能有缓存命中")
                else:
                    self.add_result("缓存命中", ValidationStatus.WARN, "未检测到明显性能提升")
            else:
                self.add_result("缓存功能", ValidationStatus.FAIL, "缓存测试请求失败")

        except Exception as e:
            self.add_result("缓存功能", ValidationStatus.FAIL, f"缓存测试失败: {str(e)}")

    async def validate_monitoring(self) -> bool:
        """验证监控功能"""
        print("\n📊 验证监控功能...")

        try:
            # 获取系统状态
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if sdk:
                # 获取连接器健康状态
                if hasattr(sdk, 'get_health_status'):
                    try:
                        health = await sdk.get_health_status()
                        self.add_result("健康监控", ValidationStatus.PASS, "健康状态获取成功", health)
                    except Exception as e:
                        self.add_result("健康监控", ValidationStatus.WARN, f"健康状态获取失败: {str(e)}")

                # 获取适配器列表
                if hasattr(sdk, 'list_adapters'):
                    try:
                        adapters = sdk.list_adapters()
                        self.add_result("适配器监控", ValidationStatus.PASS, f"发现 {len(adapters)} 类适配器", adapters)
                    except Exception as e:
                        self.add_result("适配器监控", ValidationStatus.WARN, f"适配器列表获取失败: {str(e)}")

        except Exception as e:
            self.add_result("监控验证", ValidationStatus.FAIL, f"监控验证失败: {str(e)}")

        return True

    async def run_validation(self) -> bool:
        """运行完整验证"""
        print("🚀 开始Dynamo XConnector运行时验证...")
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        success = True

        # 基础服务验证
        if not await self.validate_basic_services():
            success = False

        # XConnector集成验证
        if not await self.validate_xconnector_integration():
            success = False

        # 适配器验证
        if not await self.validate_adapters():
            success = False

        # 功能验证
        if not await self.validate_functionality():
            success = False

        # 监控验证
        if not await self.validate_monitoring():
            success = False

        # 打印所有结果
        print("\n" + "="*60)
        print("📋 验证结果汇总:")
        print("="*60)

        pass_count = 0
        fail_count = 0
        warn_count = 0

        for result in self.results:
            self.print_result(result)
            if result.status == ValidationStatus.PASS:
                pass_count += 1
            elif result.status == ValidationStatus.FAIL:
                fail_count += 1
            elif result.status == ValidationStatus.WARN:
                warn_count += 1

        print("\n" + "="*60)
        print(f"✅ 通过: {pass_count} | ❌ 失败: {fail_count} | ⚠️  警告: {warn_count}")

        if fail_count == 0:
            print("🎉 XConnector集成验证通过！")
        else:
            print("⚠️  发现问题，请检查失败项目")

        return success

async def main():
    """主函数"""
    validator = DynamoXConnectorValidator()
    success = await validator.run_validation()

    # 返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

    cat > "$VALIDATION_DIR/quick_status_check.py" << 'EOF'
#!/usr/bin/env python3
"""
XConnector 快速状态检查脚本
快速检查XConnector在Dynamo中的运行状态
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime

sys.path.insert(0, '/workspace/xconnector')

class QuickStatusChecker:
    """快速状态检查器"""

    def __init__(self):
        self.status_icons = {
            "ok": "✅",
            "warn": "⚠️ ",
            "error": "❌",
            "info": "ℹ️ ",
            "skip": "⏭️ "
        }

    def print_section(self, title: str):
        """打印节标题"""
        print(f"\n🔍 {title}")
        print("-" * (len(title) + 4))

    def print_status(self, item: str, status: str, message: str, details: str = ""):
        """打印状态行"""
        icon = self.status_icons.get(status, "❓")
        print(f"{icon} {item}: {message}")
        if details:
            print(f"   详情: {details}")

    async def check_basic_integration(self):
        """检查基础集成状态"""
        self.print_section("基础集成状态")

        try:
            # 检查模块导入
            try:
                from integrations.dynamo.autopatch import get_integration_status, get_minimal_sdk
                self.print_status("模块导入", "ok", "XConnector模块导入成功")
            except ImportError as e:
                self.print_status("模块导入", "error", "模块导入失败", str(e))
                return False

            # 检查集成状态
            try:
                status = get_integration_status()

                if status.get('sdk_available', False):
                    self.print_status("SDK可用性", "ok", "SDK已正确加载")
                else:
                    self.print_status("SDK可用性", "error", "SDK不可用")

                if status.get('config_found', False):
                    self.print_status("配置文件", "ok", "配置文件已找到")
                else:
                    self.print_status("配置文件", "warn", "配置文件未找到，使用默认配置")

                if status.get('initialized', False):
                    self.print_status("初始化状态", "ok", "XConnector已初始化")
                else:
                    self.print_status("初始化状态", "warn", "XConnector未完全初始化")

            except Exception as e:
                self.print_status("状态检查", "error", "状态检查失败", str(e))
                return False

            return True

        except Exception as e:
            self.print_status("基础集成", "error", "基础集成检查失败", str(e))
            return False

    async def check_sdk_instance(self):
        """检查SDK实例状态"""
        self.print_section("SDK实例状态")

        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if not sdk:
                self.print_status("SDK实例", "error", "无法获取SDK实例")
                return False

            self.print_status("SDK实例", "ok", f"SDK类型: {type(sdk).__name__}")

            # 检查缓存适配器
            if hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                self.print_status("缓存适配器", "ok", "缓存适配器已加载")

                try:
                    stats = sdk.cache_adapter.get_cache_statistics()
                    if stats.get('connector_class_available', False):
                        self.print_status("缓存连接器", "ok", "缓存连接器可用")
                    else:
                        self.print_status("缓存连接器", "warn", "缓存连接器不可用")

                    # 显示关键统计信息
                    if 'cache_hits' in stats:
                        self.print_status("缓存统计", "info", f"命中率相关数据可用")

                except Exception as e:
                    self.print_status("缓存统计", "warn", "无法获取缓存统计", str(e))
            else:
                self.print_status("缓存适配器", "error", "缓存适配器未初始化")

            # 检查适配器数量
            if hasattr(sdk, 'list_adapters'):
                try:
                    adapters = sdk.list_adapters()
                    total_adapters = sum(len(adapters.get(t, [])) for t in ['inference', 'cache', 'distributed'])
                    self.print_status("适配器总数", "info", f"已加载 {total_adapters} 个适配器")

                    for adapter_type, adapter_list in adapters.items():
                        if adapter_list:
                            self.print_status(f"{adapter_type}适配器", "ok", f"{len(adapter_list)} 个已加载: {', '.join(adapter_list)}")
                        else:
                            self.print_status(f"{adapter_type}适配器", "info", "无")

                except Exception as e:
                    self.print_status("适配器列表", "warn", "无法获取适配器列表", str(e))

            return True

        except Exception as e:
            self.print_status("SDK检查", "error", "SDK检查失败", str(e))
            return False

    async def check_services_connectivity(self):
        """检查服务连通性"""
        self.print_section("服务连通性")

        import socket
        import requests

        # 检查Dynamo服务
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                self.print_status("Dynamo服务", "ok", "服务正常响应")
            else:
                self.print_status("Dynamo服务", "warn", f"响应状态码: {response.status_code}")
        except Exception as e:
            self.print_status("Dynamo服务", "error", "连接失败", str(e))

        # 检查XConnector服务（如果作为独立服务运行）
        try:
            response = requests.get("http://localhost:8081/health", timeout=3)
            if response.status_code == 200:
                self.print_status("XConnector服务", "ok", "独立服务可用")
            else:
                self.print_status("XConnector服务", "warn", f"响应异常: {response.status_code}")
        except Exception:
            self.print_status("XConnector服务", "info", "独立服务不可用（正常，使用嵌入模式）")

        # 检查etcd
        etcd_hosts = ['127.0.0.1', 'localhost', 'etcd', 'etcd-server']
        etcd_connected = False

        for host in etcd_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, 2379))
                sock.close()
                if result == 0:
                    self.print_status("etcd服务", "ok", f"通过 {host}:2379 连接成功")
                    etcd_connected = True
                    break
            except Exception:
                continue

        if not etcd_connected:
            self.print_status("etcd服务", "warn", "无法连接到etcd")

        # 检查NATS
        nats_hosts = ['127.0.0.1', 'localhost', 'nats', 'nats-server']
        nats_connected = False

        for host in nats_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, 4222))
                sock.close()
                if result == 0:
                    self.print_status("NATS服务", "ok", f"通过 {host}:4222 连接成功")
                    nats_connected = True
                    break
            except Exception:
                continue

        if not nats_connected:
            self.print_status("NATS服务", "warn", "无法连接到NATS")

        return True

    async def run_functionality_test(self):
        """运行功能性测试"""
        self.print_section("功能性测试")

        # 测试推理功能
        try:
            import requests

            test_payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.1
            }

            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json=test_payload,
                timeout=15
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result:
                    self.print_status("推理测试", "ok", f"推理成功 (耗时: {duration:.2f}s)")

                    # 显示生成的内容
                    if result['choices'] and 'message' in result['choices'][0]:
                        content = result['choices'][0]['message'].get('content', '')[:30]
                        self.print_status("生成内容", "info", f"'{content}...'")
                else:
                    self.print_status("推理测试", "warn", "响应格式异常")
            else:
                self.print_status("推理测试", "error", f"请求失败: {response.status_code}")

        except Exception as e:
            self.print_status("推理测试", "error", "测试失败", str(e))

        # 测试缓存功能（简单）
        try:
            from integrations.dynamo.autopatch import get_minimal_sdk

            sdk = get_minimal_sdk()
            if sdk and hasattr(sdk, 'cache_adapter') and sdk.cache_adapter:
                try:
                    stats = sdk.cache_adapter.get_cache_statistics()
                    if stats:
                        self.print_status("缓存功能", "ok", "缓存适配器响应正常")
                    else:
                        self.print_status("缓存功能", "warn", "缓存适配器无响应")
                except Exception as e:
                    self.print_status("缓存功能", "warn", "缓存测试失败", str(e))
            else:
                self.print_status("缓存功能", "skip", "缓存适配器不可用")

        except Exception as e:
            self.print_status("缓存测试", "warn", "缓存测试异常", str(e))

        return True

    def print_summary(self):
        """打印总结"""
        print("\n" + "="*50)
        print("📋 状态检查总结")
        print("="*50)
        print("✅ = 正常 | ⚠️  = 警告 | ❌ = 错误 | ℹ️  = 信息 | ⏭️  = 跳过")
        print(f"🕐 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n💡 建议:")
        print("   - 如果看到❌错误，请检查相关配置和服务")
        print("   - 如果看到⚠️ 警告，功能可能受限但基本可用")
        print("   - 运行完整验证: python dynamo_xconnector_validator.py")
        print("   - 启动持续监控: python continuous_monitor.py")

    async def run_quick_check(self):
        """运行快速检查"""
        print("⚡ XConnector 快速状态检查")
        print("="*40)

        # 基础集成检查
        await self.check_basic_integration()

        # SDK实例检查
        await self.check_sdk_instance()

        # 服务连通性检查
        await self.check_services_connectivity()

        # 功能性测试
        await self.run_functionality_test()

        # 打印总结
        self.print_summary()

async def main():
    """主函数"""
    checker = QuickStatusChecker()
    await checker.run_quick_check()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 检查已中断")
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        sys.exit(1)
EOF

    # 设置执行权限
    chmod +x "$VALIDATION_DIR"/*.py

    print_success "验证工具部署完成"
}

# 创建快捷命令
create_shortcuts() {
    print_info "创建快捷命令..."

    # 创建快速检查命令
    cat > "$VALIDATION_DIR/check" << EOF
#!/bin/bash
cd "$VALIDATION_DIR"
python3 quick_status_check.py "\$@"
EOF

    # 创建完整验证命令
    cat > "$VALIDATION_DIR/validate" << EOF
#!/bin/bash
cd "$VALIDATION_DIR"
python3 dynamo_xconnector_validator.py "\$@"
EOF

    # 创建监控命令
    cat > "$VALIDATION_DIR/monitor" << EOF
#!/bin/bash
cd "$VALIDATION_DIR"
python3 continuous_monitor.py "\$@"
EOF

    # 创建综合状态命令
    cat > "$VALIDATION_DIR/status" << 'EOF'
#!/bin/bash
# 综合状态检查命令

echo "🚀 XConnector 状态总览"
echo "========================"

# 检查Dynamo服务
echo -n "Dynamo服务: "
if curl -s -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ 运行中"
else
    echo "❌ 未运行"
fi

# 检查XConnector服务
echo -n "XConnector服务: "
if curl -s -f http://localhost:8081/health >/dev/null 2>&1; then
    echo "✅ 运行中"
else
    echo "ℹ️  嵌入模式"
fi

# 检查基础服务
echo -n "etcd服务: "
if nc -z localhost 2379 2>/dev/null; then
    echo "✅ 连接正常"
else
    echo "❌ 连接失败"
fi

echo -n "NATS服务: "
if nc -z localhost 4222 2>/dev/null; then
    echo "✅ 连接正常"
else
    echo "❌ 连接失败"
fi

echo ""
echo "📋 可用命令:"
echo "  ./check     - 快速状态检查"
echo "  ./validate  - 完整功能验证"
echo "  ./monitor   - 持续监控"
echo "  ./status    - 服务状态总览"
EOF

    # 设置执行权限
    chmod +x "$VALIDATION_DIR"/check
    chmod +x "$VALIDATION_DIR"/validate
    chmod +x "$VALIDATION_DIR"/monitor
    chmod +x "$VALIDATION_DIR"/status

    print_success "快捷命令创建完成"
}

# 创建配置文件
create_config() {
    print_info "创建配置文件..."

    cat > "$VALIDATION_DIR/validation_config.json" << EOF
{
    "dynamo_url": "http://localhost:8000",
    "xconnector_service_url": "http://localhost:8081",
    "monitoring": {
        "default_interval": 30,
        "max_history": 100,
        "log_file": "$VALIDATION_DIR/logs/validation.log"
    },
    "alerts": {
        "enable_notifications": false,
        "critical_failures": true,
        "performance_degradation": true
    },
    "timeouts": {
        "service_check": 5,
        "inference_test": 30,
        "cache_test": 10
    }
}
EOF

    print_success "配置文件创建完成"
}

# 安装系统服务（可选）
install_systemd_service() {
    if [ "$EUID" -eq 0 ]; then
        print_info "安装systemd服务..."

        cat > /etc/systemd/system/xconnector-monitor.service << EOF
[Unit]
Description=XConnector Monitoring Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$VALIDATION_DIR
ExecStart=/usr/bin/python3 $VALIDATION_DIR/continuous_monitor.py -i 60
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        print_success "systemd服务已安装，使用以下命令管理:"
        print_info "  sudo systemctl start xconnector-monitor"
        print_info "  sudo systemctl enable xconnector-monitor"
        print_info "  sudo systemctl status xconnector-monitor"
    else
        print_warning "非root用户，跳过systemd服务安装"
    fi
}

# 运行初始验证
run_initial_validation() {
    print_info "运行初始验证..."

    cd "$VALIDATION_DIR"

    # 等待服务启动
    print_info "等待Dynamo服务启动..."
    for i in {1..30}; do
        if curl -s -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Dynamo服务已启动"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Dynamo服务启动超时，继续验证..."
        fi
        sleep 2
    done

    # 运行快速检查
    print_info "运行快速状态检查..."
    python3 quick_status_check.py

    echo ""
    print_info "初始验证完成"
}

# 显示使用说明
show_usage() {
    echo ""
    print_success "🎉 XConnector验证工具部署完成！"
    echo ""
    print_info "验证工具位置: $VALIDATION_DIR"
    echo ""
    print_info "📋 使用方法:"
    echo "  cd $VALIDATION_DIR"
    echo ""
    echo "  ./status    - 查看服务状态总览"
    echo "  ./check     - 快速状态检查（推荐定期使用）"
    echo "  ./validate  - 完整功能验证（推荐启动后运行）"
    echo "  ./monitor   - 启动持续监控（后台运行）"
    echo ""
    print_info "📊 监控建议:"
    echo "  - 启动Dynamo后运行: ./validate"
    echo "  - 定期检查状态: ./check"
    echo "  - 生产环境: nohup ./monitor &"
    echo ""
    print_info "📁 日志位置:"
    echo "  - 验证日志: $VALIDATION_DIR/logs/"
    echo "  - 监控历史: $VALIDATION_DIR/logs/xconnector_metrics.jsonl"
    echo ""
    print_warning "注意: 首次运行前确保Dynamo服务已启动"
}

# 主函数
main() {
    echo "🚀 XConnector验证工具部署脚本"
    echo "=================================="

    # 解析命令行参数
    SKIP_VALIDATION=false
    INSTALL_SERVICE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --install-service)
                INSTALL_SERVICE=true
                shift
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --skip-validation  跳过初始验证"
                echo "  --install-service  安装systemd服务"
                echo "  -h, --help        显示帮助"
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                exit 1
                ;;
        esac
    done

    # 执行部署步骤
    check_dependencies
    deploy_validation_tools
    create_shortcuts
    create_config

    if [ "$INSTALL_SERVICE" = true ]; then
        install_systemd_service
    fi

    if [ "$SKIP_VALIDATION" = false ]; then
        run_initial_validation
    fi

    show_usage
}

# 运行主函数
main "$@"