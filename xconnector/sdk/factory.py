# xconnector/sdk/factory.py
"""
XConnector SDK 工厂函数

提供便捷的SDK创建和配置方法，支持不同的使用场景和快速集成。
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from xconnector.sdk import XConnectorSDK, SDKConfig, SDKMode
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


def create_sdk(
        mode: Union[str, SDKMode] = SDKMode.EMBEDDED,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
) -> XConnectorSDK:
    """
    创建XConnector SDK实例

    Args:
        mode: SDK运行模式 (embedded/hybrid)
        config: 配置字典
        **kwargs: 额外配置参数

    Returns:
        XConnectorSDK: SDK实例

    Example:
        >>> sdk = create_sdk(
        ...     mode="embedded",
        ...     enable_kv_cache=True,
        ...     adapters=[{
        ...         "name": "lmcache",
        ...         "type": "cache",
        ...         "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter"
        ...     }]
        ... )
    """
    # 处理模式参数
    if isinstance(mode, str):
        mode = SDKMode(mode.lower())

    # 合并配置
    final_config = config or {}
    final_config.update(kwargs)
    final_config["mode"] = mode

    # 创建SDK配置
    sdk_config = SDKConfig(**final_config)

    # 创建并返回SDK实例
    return XConnectorSDK(sdk_config)


def create_dynamo_sdk(
        dynamo_config: Dict[str, Any],
        **kwargs
) -> XConnectorSDK:
    """
    专门为Dynamo创建的SDK实例

    Args:
        dynamo_config: Dynamo配置字典
        **kwargs: 额外SDK配置

    Returns:
        XConnectorSDK: 配置好的SDK实例

    Example:
        >>> dynamo_config = {
        ...     "model": "/data/model/DeepSeek-R1",
        ...     "xconnector": {
        ...         "enabled": True,
        ...         "adapters": {...}
        ...     }
        ... }
        >>> sdk = create_dynamo_sdk(dynamo_config)
    """
    # 从Dynamo配置中提取XConnector配置
    xconnector_config = dynamo_config.get("xconnector", {})

    # 构建SDK配置
    sdk_config = {
        "mode": SDKMode.EMBEDDED,
        "enable_kv_cache": True,
        "enable_distributed": True,
        "enable_monitoring": True,
    }

    # 更新配置
    sdk_config.update(xconnector_config)
    sdk_config.update(kwargs)

    # 如果没有显式配置适配器，使用默认适配器
    if "adapters" not in sdk_config:
        sdk_config["adapters"] = _get_default_dynamo_adapters(dynamo_config)

    return create_sdk(**sdk_config)


def create_kv_cache_sdk(
        cache_backend: str = "lmcache",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
) -> XConnectorSDK:
    """
    创建专用于KV缓存的SDK实例

    Args:
        cache_backend: 缓存后端类型 (lmcache/redis/memory)
        config: 缓存配置
        **kwargs: 额外配置

    Returns:
        XConnectorSDK: KV缓存专用SDK实例

    Example:
        >>> sdk = create_kv_cache_sdk(
        ...     cache_backend="lmcache",
        ...     config={"storage_backend": "memory", "max_cache_size": 1024}
        ... )
    """
    # 构建适配器配置
    adapter_config = _get_cache_adapter_config(cache_backend, config or {})

    # 创建SDK配置
    sdk_config = {
        "mode": SDKMode.EMBEDDED,
        "enable_kv_cache": True,
        "enable_distributed": False,
        "enable_monitoring": True,
        "adapters": [adapter_config]
    }

    sdk_config.update(kwargs)

    return create_sdk(**sdk_config)


def create_distributed_sdk(
        framework: str = "dynamo",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
) -> XConnectorSDK:
    """
    创建专用于分布式处理的SDK实例

    Args:
        framework: 分布式框架类型 (dynamo/ray/horovod)
        config: 分布式配置
        **kwargs: 额外配置

    Returns:
        XConnectorSDK: 分布式专用SDK实例
    """
    # 构建适配器配置
    adapter_config = _get_distributed_adapter_config(framework, config or {})

    # 创建SDK配置
    sdk_config = {
        "mode": SDKMode.EMBEDDED,
        "enable_kv_cache": True,
        "enable_distributed": True,
        "enable_monitoring": True,
        "adapters": [adapter_config]
    }

    sdk_config.update(kwargs)

    return create_sdk(**sdk_config)


def create_sdk_from_config_file(
        config_path: Union[str, Path],
        **kwargs
) -> XConnectorSDK:
    """
    从配置文件创建SDK实例

    Args:
        config_path: 配置文件路径 (支持 .yaml/.json)
        **kwargs: 额外配置覆盖

    Returns:
        XConnectorSDK: SDK实例
    """
    import yaml
    import json

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # 合并额外配置
    config.update(kwargs)

    return create_sdk(**config)


def auto_detect_and_create_sdk(
        context: Optional[Dict[str, Any]] = None,
        **kwargs
) -> XConnectorSDK:
    """
    自动检测环境并创建合适的SDK实例

    Args:
        context: 环境上下文信息
        **kwargs: 额外配置

    Returns:
        XConnectorSDK: 自动配置的SDK实例
    """
    context = context or {}

    # 检测运行环境
    detected_config = _detect_environment(context)

    # 合并配置
    detected_config.update(kwargs)

    logger.info(f"Auto-detected SDK configuration: {detected_config}")

    return create_sdk(**detected_config)


# === 内部辅助函数 ===

def _get_default_dynamo_adapters(dynamo_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """获取Dynamo的默认适配器配置"""
    adapters = []

    # 添加VLLM推理适配器
    adapters.append({
        "name": "vllm",
        "type": "inference",
        "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
        "config": {
            "model_name": dynamo_config.get("model", ""),
            "enable_prefix_caching": dynamo_config.get("enable-prefix-caching", True),
            "tensor_parallel_size": _extract_tensor_parallel_size(dynamo_config),
            "max_batch_size": dynamo_config.get("max-num-batched-tokens", 256)
        },
        "enabled": True,
        "priority": 0
    })

    # 添加LMCache缓存适配器
    cache_config = dynamo_config.get("cache_config", {})
    adapters.append({
        "name": "lmcache",
        "type": "cache",
        "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
        "config": {
            "storage_backend": cache_config.get("storage_backend", "memory"),
            "max_cache_size": cache_config.get("max_cache_size", 1024),
            "enable_compression": cache_config.get("enable_compression", True),
            "block_size": dynamo_config.get("block-size", 16)
        },
        "enabled": True,
        "priority": 1
    })

    # 添加Dynamo分布式适配器
    adapters.append({
        "name": "dynamo",
        "type": "distributed",
        "class_path": "xconnector.adapters.distributed.dynamo_adapter.DynamoAdapter",
        "config": {
            "namespace": "dynamo",
            "component_name": "xconnector",
            "etcd_endpoints": dynamo_config.get("etcd-endpoints", ["http://localhost:2379"]),
            "nats_url": dynamo_config.get("nats-url", "nats://localhost:4222")
        },
        "enabled": True,
        "priority": 2
    })

    return adapters


def _extract_tensor_parallel_size(dynamo_config: Dict[str, Any]) -> int:
    """从Dynamo配置中提取tensor parallel size"""
    # 从VllmWorker配置中提取
    vllm_config = dynamo_config.get("VllmWorker", {})

    if "tensor-parallel-size" in vllm_config:
        return vllm_config["tensor-parallel-size"]

    # 从ServiceArgs.resources.gpu中提取
    service_args = vllm_config.get("ServiceArgs", {})
    resources = service_args.get("resources", {})
    gpu_count = resources.get("gpu", "1")

    try:
        return int(gpu_count)
    except (ValueError, TypeError):
        return 1


def _get_cache_adapter_config(cache_backend: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """获取缓存适配器配置"""
    if cache_backend == "lmcache":
        return {
            "name": "lmcache",
            "type": "cache",
            "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
            "config": {
                "storage_backend": config.get("storage_backend", "memory"),
                "max_cache_size": config.get("max_cache_size", 1024),
                "enable_compression": config.get("enable_compression", True),
                **config
            },
            "enabled": True
        }
    elif cache_backend == "redis":
        return {
            "name": "redis",
            "type": "cache",
            "class_path": "xconnector.adapters.cache.redis_adapter.RedisAdapter",
            "config": {
                "host": config.get("host", "localhost"),
                "port": config.get("port", 6379),
                "db": config.get("db", 0),
                **config
            },
            "enabled": True
        }
    else:
        raise ValueError(f"Unsupported cache backend: {cache_backend}")


def _get_distributed_adapter_config(framework: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """获取分布式适配器配置"""
    if framework == "dynamo":
        return {
            "name": "dynamo",
            "type": "distributed",
            "class_path": "xconnector.adapters.distributed.dynamo_adapter.DynamoAdapter",
            "config": {
                "namespace": config.get("namespace", "dynamo"),
                "component_name": config.get("component_name", "xconnector"),
                **config
            },
            "enabled": True
        }
    else:
        raise ValueError(f"Unsupported distributed framework: {framework}")


def _detect_environment(context: Dict[str, Any]) -> Dict[str, Any]:
    """检测运行环境并返回合适的配置"""
    config = {
        "mode": SDKMode.EMBEDDED,
        "enable_kv_cache": True,
        "enable_distributed": False,
        "enable_monitoring": True,
        "adapters": []
    }

    # 检测是否在Dynamo环境中
    if _is_dynamo_environment(context):
        config["enable_distributed"] = True
        config["adapters"] = _get_default_dynamo_adapters(context)
        logger.info("Detected Dynamo environment")

    # 检测是否有GPU
    if _has_gpu_available():
        config["performance"] = {
            "async_mode": True,
            "batch_processing": True,
            "gpu_acceleration": True
        }
        logger.info("Detected GPU environment")

    return config


def _is_dynamo_environment(context: Dict[str, Any]) -> bool:
    """检测是否在Dynamo环境中运行"""
    # 检查环境变量
    import os
    if "DYNAMO_WORKER" in os.environ:
        return True

    # 检查上下文中的Dynamo配置
    if any(key in context for key in ["VllmWorker", "PrefillWorker", "Processor"]):
        return True

    # 检查是否可以导入Dynamo模块
    try:
        import dynamo
        return True
    except ImportError:
        return False


def _has_gpu_available() -> bool:
    """检测是否有GPU可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# === 便捷别名 ===

# 为常用场景提供便捷别名
create_for_dynamo = create_dynamo_sdk
create_for_kv_cache = create_kv_cache_sdk
create_for_distributed = create_distributed_sdk
auto_create = auto_detect_and_create_sdk

# === 导出 ===

__all__ = [
    'create_sdk',
    'create_dynamo_sdk',
    'create_kv_cache_sdk',
    'create_distributed_sdk',
    'create_sdk_from_config_file',
    'auto_detect_and_create_sdk',
    # 别名
    'create_for_dynamo',
    'create_for_kv_cache',
    'create_for_distributed',
    'auto_create'
]