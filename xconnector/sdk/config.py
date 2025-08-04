# xconnector/sdk/config.py
"""
XConnector SDK 配置模块

复用现有配置结构，提供SDK专用的配置管理
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

# 复用现有的配置结构
from xconnector.config import (
    AdapterType, AdapterConfig, RouterConfig,
    HealthCheckConfig, MonitoringConfig, SDKMode, SDKConfig
)
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class SDKMode(Enum):
    """SDK运行模式"""
    EMBEDDED = "embedded"  # 嵌入模式：完全集成到宿主进程
    HYBRID = "hybrid"  # 混合模式：部分功能独立


@dataclass
class SDKConfig:
    """SDK配置类 - 复用并扩展现有配置"""

    # 运行模式
    mode: SDKMode = SDKMode.EMBEDDED

    # 功能开关
    enable_kv_cache: bool = True
    enable_distributed: bool = True
    enable_monitoring: bool = True

    # 复用现有适配器配置
    adapters: List[Dict[str, Any]] = field(default_factory=list)

    # 复用现有路由配置
    router: RouterConfig = field(default_factory=RouterConfig)

    # 复用现有健康检查配置
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)

    # 复用现有监控配置
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # SDK特有配置
    integration: Dict[str, Any] = field(default_factory=lambda: {
        "kv_cache": {
            "default_adapter": "lmcache",
            "enable_fallback": True,
            "enable_batching": True
        },
        "distributed": {
            "auto_discovery": True,
            "health_monitoring": True
        }
    })

    # 性能配置
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "async_mode": True,
        "batch_processing": True,
        "memory_optimization": True,
        "max_concurrent_requests": 1000
    })

    # 错误处理配置
    error_handling: Dict[str, Any] = field(default_factory=lambda: {
        "graceful_degradation": True,
        "error_isolation": True,
        "fallback_enabled": True,
        "max_retry_attempts": 3
    })

    def __post_init__(self):
        """后处理：转换字符串枚举"""
        if isinstance(self.mode, str):
            self.mode = SDKMode(self.mode.lower())


# === Dynamo配置适配器 ===

class DynamoConfigAdapter:
    """
    Dynamo配置适配器

    将Dynamo配置转换为XConnector SDK配置
    """

    @staticmethod
    def from_dynamo_config(dynamo_config: Dict[str, Any]) -> SDKConfig:
        """从Dynamo配置创建SDK配置"""

        # 提取XConnector相关配置
        xconnector_config = dynamo_config.get("xconnector", {})

        # 创建适配器配置
        adapters = []

        # 自动检测并添加VLLM适配器
        if "VllmWorker" in dynamo_config:
            vllm_config = DynamoConfigAdapter._extract_vllm_config(dynamo_config)
            adapters.append(vllm_config)

        # 自动检测并添加LMCache适配器
        if dynamo_config.get("enable-prefix-caching") or "kv-transfer-config" in dynamo_config:
            lmcache_config = DynamoConfigAdapter._extract_lmcache_config(dynamo_config)
            adapters.append(lmcache_config)

        # 添加Dynamo分布式适配器
        if any(key in dynamo_config for key in ["etcd-endpoints", "nats-url"]):
            dynamo_adapter_config = DynamoConfigAdapter._extract_dynamo_config(dynamo_config)
            adapters.append(dynamo_adapter_config)

        # 合并用户自定义适配器
        custom_adapters = xconnector_config.get("adapters", {})
        for name, config in custom_adapters.items():
            adapters.append({
                "name": name,
                "type": config.get("type", "cache"),
                "class_path": config.get("class_path", ""),
                "config": config.get("config", {}),
                "enabled": config.get("enabled", True)
            })

        # 创建SDK配置
        return SDKConfig(
            mode=SDKMode.EMBEDDED,
            enable_kv_cache=True,
            enable_distributed=True,
            adapters=adapters,
            integration={
                "kv_cache": {
                    "default_adapter": "lmcache",
                    "enable_fallback": True
                },
                "distributed": {
                    "namespace": "dynamo",
                    "etcd_endpoints": dynamo_config.get("etcd-endpoints", []),
                    "nats_url": dynamo_config.get("nats-url", "")
                }
            }
        )

    @staticmethod
    def _extract_vllm_config(dynamo_config: Dict[str, Any]) -> Dict[str, Any]:
        """提取VLLM适配器配置"""
        vllm_worker_config = dynamo_config.get("VllmWorker", {})

        return {
            "name": "vllm",
            "type": "inference",
            "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
            "config": {
                "model_name": dynamo_config.get("model", ""),
                "tensor_parallel_size": vllm_worker_config.get("tensor-parallel-size", 1),
                "enable_prefix_caching": dynamo_config.get("enable-prefix-caching", True),
                "max_batch_size": vllm_worker_config.get("max-num-batched-tokens", 256),
                "block_size": dynamo_config.get("block-size", 16),
                "max_model_len": dynamo_config.get("max-model-len", 16384)
            },
            "enabled": True,
            "priority": 0
        }

    @staticmethod
    def _extract_lmcache_config(dynamo_config: Dict[str, Any]) -> Dict[str, Any]:
        """提取LMCache适配器配置"""
        kv_transfer_config = dynamo_config.get("kv-transfer-config", "{}")

        # 解析KV transfer配置（通常是JSON字符串）
        import json
        try:
            if isinstance(kv_transfer_config, str):
                kv_config = json.loads(kv_transfer_config)
            else:
                kv_config = kv_transfer_config
        except:
            kv_config = {}

        return {
            "name": "lmcache",
            "type": "cache",
            "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
            "config": {
                "storage_backend": kv_config.get("storage_backend", "memory"),
                "max_cache_size": 1024,
                "enable_compression": True,
                "block_size": dynamo_config.get("block-size", 16),
                "kv_connector": kv_config.get("kv_connector", "LMCacheConnector")
            },
            "enabled": True,
            "priority": 1
        }

    @staticmethod
    def _extract_dynamo_config(dynamo_config: Dict[str, Any]) -> Dict[str, Any]:
        """提取Dynamo分布式适配器配置"""
        return {
            "name": "dynamo",
            "type": "distributed",
            "class_path": "xconnector.adapters.distributed.dynamo_adapter.DynamoAdapter",
            "config": {
                "namespace": "dynamo",
                "component_name": "xconnector",
                "etcd_endpoints": dynamo_config.get("etcd-endpoints", ["http://localhost:2379"]),
                "nats_url": dynamo_config.get("nats-url", "nats://localhost:4222"),
                "routing_policy": {
                    "strategy": "least_loaded",
                    "health_check_interval": 30
                }
            },
            "enabled": True,
            "priority": 2
        }


# === vLLM配置适配器 ===

class VLLMConfigAdapter:
    """
    vLLM配置适配器

    将vLLM配置转换为XConnector SDK配置
    """

    @staticmethod
    def from_vllm_config(vllm_config: Any) -> SDKConfig:
        """从vLLM VllmConfig对象创建SDK配置"""

        adapters = []

        # 添加VLLM适配器
        vllm_adapter = {
            "name": "vllm",
            "type": "inference",
            "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
            "config": {
                "model_name": getattr(vllm_config.model_config, 'model', '') if hasattr(vllm_config,
                                                                                        'model_config') else '',
                "tensor_parallel_size": getattr(vllm_config.parallel_config, 'tensor_parallel_size', 1) if hasattr(
                    vllm_config, 'parallel_config') else 1,
                "enable_prefix_caching": True,
                "block_size": getattr(vllm_config.cache_config, 'block_size', 16) if hasattr(vllm_config,
                                                                                             'cache_config') else 16
            },
            "enabled": True
        }
        adapters.append(vllm_adapter)

        # 如果启用了KV transfer，添加LMCache适配器
        if hasattr(vllm_config, 'kv_transfer_config') and vllm_config.kv_transfer_config:
            lmcache_adapter = {
                "name": "lmcache",
                "type": "cache",
                "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                "config": {
                    "storage_backend": "memory",
                    "max_cache_size": 1024
                },
                "enabled": True
            }
            adapters.append(lmcache_adapter)

        return SDKConfig(
            mode=SDKMode.EMBEDDED,
            enable_kv_cache=True,
            adapters=adapters
        )


# === 环境变量配置 ===

class EnvironmentConfigLoader:
    """从环境变量加载配置"""

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}

        # SDK模式
        if os.getenv("XCONNECTOR_SDK_MODE"):
            config["mode"] = os.getenv("XCONNECTOR_SDK_MODE", "embedded")

        # 功能开关
        if os.getenv("ENABLE_KV_CACHE"):
            config["enable_kv_cache"] = os.getenv("ENABLE_KV_CACHE", "true").lower() == "true"

        if os.getenv("ENABLE_DISTRIBUTED"):
            config["enable_distributed"] = os.getenv("ENABLE_DISTRIBUTED", "true").lower() == "true"

        # 错误处理
        if os.getenv("GRACEFUL_DEGRADATION"):
            config["error_handling"] = {
                "graceful_degradation": os.getenv("GRACEFUL_DEGRADATION", "true").lower() == "true"
            }

        return config


# === 配置验证器 ===

class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_sdk_config(config: SDKConfig) -> List[str]:
        """验证SDK配置"""
        errors = []

        # 验证适配器配置
        adapter_names = set()
        for adapter in config.adapters:
            name = adapter.get("name")
            if not name:
                errors.append("Adapter missing name")
                continue

            if name in adapter_names:
                errors.append(f"Duplicate adapter name: {name}")
            adapter_names.add(name)

            if not adapter.get("class_path"):
                errors.append(f"Adapter {name} missing class_path")

        # 验证模式
        if config.mode not in SDKMode:
            errors.append(f"Invalid SDK mode: {config.mode}")

        return errors


# === 便捷函数 ===

def create_sdk_config_from_dynamo(dynamo_config: Dict[str, Any]) -> SDKConfig:
    """从Dynamo配置创建SDK配置的便捷函数"""
    return DynamoConfigAdapter.from_dynamo_config(dynamo_config)


def create_sdk_config_from_vllm(vllm_config: Any) -> SDKConfig:
    """从vLLM配置创建SDK配置的便捷函数"""
    return VLLMConfigAdapter.from_vllm_config(vllm_config)


def load_sdk_config_from_env() -> SDKConfig:
    """从环境变量加载SDK配置的便捷函数"""
    env_config = EnvironmentConfigLoader.load_from_env()
    return SDKConfig(**env_config)


# === 默认配置 ===

def get_default_sdk_config() -> SDKConfig:
    """获取默认SDK配置"""
    return SDKConfig(
        mode=SDKMode.EMBEDDED,
        enable_kv_cache=True,
        enable_distributed=False,
        adapters=[
            {
                "name": "lmcache",
                "type": "cache",
                "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                "config": {"storage_backend": "memory", "max_cache_size": 512},
                "enabled": True
            }
        ]
    )


# === 使用示例 ===

if __name__ == "__main__":
    # 示例：从Dynamo配置创建SDK配置
    dynamo_config = {
        "model": "/data/model/DeepSeek-R1",
        "block-size": 64,
        "enable-prefix-caching": True,
        "VllmWorker": {
            "tensor-parallel-size": 1,
            "max-num-batched-tokens": 256
        },
        "etcd-endpoints": ["http://etcd:2379"],
        "xconnector": {
            "enabled": True
        }
    }

    sdk_config = create_sdk_config_from_dynamo(dynamo_config)
    print(f"Generated SDK config with {len(sdk_config.adapters)} adapters")

    # 验证配置
    errors = ConfigValidator.validate_sdk_config(sdk_config)
    if errors:
        print(f"Config validation errors: {errors}")
    else:
        print("✓ SDK config is valid")