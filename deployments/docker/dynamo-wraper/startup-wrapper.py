#!/usr/bin/env python3
"""
XConnector-Dynamo Integration Startup Wrapper
用于在 Dynamo 启动时注入 XConnector 扩展
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_xconnector_service():
    """等待 XConnector 服务准备就绪"""
    service_url = os.getenv("XCONNECTOR_SERVICE_URL", "http://xconnector-service:8081")
    max_retries = 30
    retry_delay = 2

    logger.info(f"Waiting for XConnector service at {service_url}")

    for i in range(max_retries):
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ XConnector service is ready")
                return True
        except Exception as e:
            logger.info(f"Waiting for XConnector service... ({i + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)

    logger.warning("⚠ XConnector service not ready, continuing anyway")
    return False


def create_xconnector_extension():
    """创建 XConnector 扩展文件"""
    extension_content = '''
import os
import sys
import logging
import requests
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)

class RemoteXConnectorExtension:
    """远程 XConnector 扩展"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.enabled = True

    async def route_message(self, source: str, target: str, method: str, **kwargs) -> Any:
        """通过远程 XConnector 服务路由消息"""
        try:
            response = requests.post(
                f"{self.service_url}/route",
                json={
                    "source": source,
                    "target": target,
                    "method": method,
                    "params": kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                return result.get("result")
            else:
                raise Exception(result.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Remote routing failed: {e}")
            # 返回空结果而不是抛出异常，避免中断 Dynamo
            return {"found": False}

class ExtensionLoader:
    """扩展加载器"""

    _extensions = {}

    @classmethod
    def load_extensions(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """加载扩展"""
        if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
            service_url = os.getenv("XCONNECTOR_SERVICE_URL", "http://xconnector-service:8081")
            extension = RemoteXConnectorExtension(service_url)
            cls._extensions["xconnector"] = extension
            logger.info(f"Loaded XConnector extension (remote mode): {service_url}")

        return cls._extensions

    @classmethod
    def get_extension(cls, name: str) -> Optional[Any]:
        """获取扩展"""
        return cls._extensions.get(name)

    @classmethod
    def inject_into_worker(cls, worker_instance: Any) -> None:
        """注入扩展到 worker"""
        xconnector_ext = cls.get_extension("xconnector")
        if xconnector_ext:
            worker_instance.xconnector = xconnector_ext
            worker_instance.xconnector_enabled = True
            worker_instance.xconnector_mode = "remote"

            logger.info("✓ XConnector injected into worker (remote mode)")
        else:
            logger.warning("⚠ XConnector extension not available")

# 测试连接
def test_xconnector_connection():
    """测试 XConnector 连接"""
    try:
        service_url = os.getenv("XCONNECTOR_SERVICE_URL", "http://xconnector-service:8081")
        response = requests.get(f"{service_url}/status", timeout=10)

        if response.status_code == 200:
            status = response.json()
            logger.info(f"✓ XConnector service status: {status}")
            return True
        else:
            logger.error(f"✗ XConnector service error: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ XConnector connection test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing XConnector connection...")
    test_xconnector_connection()
'''

    # 写入扩展文件
    extension_dir = Path("/workspace/xconnector-integration")
    extension_dir.mkdir(exist_ok=True)

    extension_file = extension_dir / "xconnector_extension.py"
    with open(extension_file, 'w') as f:
        f.write(extension_content)

    # 创建 __init__.py
    init_file = extension_dir / "__init__.py"
    with open(init_file, 'w') as f:
        f.write("# XConnector Integration Package\n")

    logger.info(f"✓ Created XConnector extension at {extension_file}")


def setup_environment():
    """设置环境"""
    # 添加集成包到 Python 路径
    sys.path.insert(0, "/workspace/xconnector-integration")

    # 设置环境变量
    os.environ["PYTHONPATH"] = "/workspace:/workspace/xconnector-integration:" + os.environ.get("PYTHONPATH", "")

    logger.info("✓ Environment setup completed")


def main():
    """主函数"""
    logger.info("=== XConnector-Dynamo Integration Startup ===")

    # 1. 设置环境
    setup_environment()

    # 2. 等待 XConnector 服务
    if os.getenv("ENABLE_XCONNECTOR", "").lower() == "true":
        wait_for_xconnector_service()

    # 3. 创建集成扩展
    create_xconnector_extension()

    logger.info("✓ XConnector integration setup completed")
    logger.info("Ready to start Dynamo with XConnector support")


if __name__ == "__main__":
    main()