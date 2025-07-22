# xconnector/utils/config.py
import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)

class AdapterType(Enum):
    """适配器类型枚举"""
    INFERENCE = "inference"
    CACHE = "cache"
    DISTRIBUTED = "distributed"

@dataclass
class AdapterConfig:
    """适配器配置"""
    name: str
    type: AdapterType
    class_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小优先级越高
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        # 如果 type 是字符串，转换为 AdapterType
        if isinstance(self.type, str):
            self.type = AdapterType(self.type)

@dataclass
class RouterConfig:
    """路由器配置"""
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, cache_aware, weighted
    request_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    enable_fallback: bool = True
    fallback_strategy: str = "default"
    # 路由规则
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    # 扩展配置
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    enabled: bool = True
    interval: int = 30  # 检查间隔（秒）
    timeout: int = 10  # 单次检查超时时间（秒）
    failure_threshold: int = 3  # 失败阈值
    success_threshold: int = 2  # 成功阈值
    initial_delay: int = 0  # 初始延迟（秒）
    log_results: bool = True
    enable_auto_recovery: bool = True
    recovery_delay: int = 60  # 自动恢复延迟（秒）
    # 自定义健康检查端点
    custom_endpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_enabled: bool = True
    metrics_interval: int = 60  # 指标收集间隔（秒）
    metrics_retention: int = 86400  # 指标保留时间（秒）
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_grafana: bool = False
    grafana_dashboard_path: Optional[str] = None
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_max_size: int = 100  # MB
    log_backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # 警报配置
    enable_alerts: bool = False
    alert_endpoints: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectorConfig:
    """XConnector主配置类"""
    # 核心配置
    adapters: List[AdapterConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    # 系统配置
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # 兼容性属性
    @property
    def host(self) -> str:
        return self.network.host

    @property
    def port(self) -> int:
        return self.network.port

    @property
    def workers(self) -> int:
        return self.network.workers

    @property
    def log_level(self) -> str:
        return self.monitoring.log_level

    @property
    def log_file(self) -> Optional[str]:
        return self.monitoring.log_file

    @property
    def queue_size(self) -> int:
        return self.performance.queue_size

    @property
    def batch_size(self) -> int:
        return self.performance.batch_size

    @property
    def max_concurrent_requests(self) -> int:
        return self.performance.max_concurrent_requests

    @property
    def enable_health_check(self) -> bool:
        return self.health_check.enabled

    @property
    def health_check_interval(self) -> int:
        return self.health_check.interval

    @property
    def log_health_check(self) -> bool:
        return self.health_check.log_results


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[ConnectorConfig] = None
        self._config_cache: Dict[str, Any] = {}
        self._watchers: List[Any] = []

    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 查找顺序：环境变量 -> 当前目录 -> 用户目录 -> 系统目录
        paths = [
            os.environ.get('XCONNECTOR_CONFIG'),
            './xconnector.yaml',
            './xconnector.json',
            '~/.xconnector/config.yaml',
            '~/.xconnector/config.json',
            '/etc/xconnector/config.yaml',
            '/etc/xconnector/config.json'
        ]

        for path in paths:
            if path and Path(path).expanduser().exists():
                return str(Path(path).expanduser())

        return './xconnector.yaml'

    def load_config(self) -> ConnectorConfig:
        """加载配置"""
        if self._config is not None:
            return self._config

        if not Path(self.config_path).exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = ConnectorConfig()
            return self._config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            self._config = self._dict_to_config(data)
            logger.info(f"Config loaded from: {self.config_path}")
            return self._config

        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self._config = ConnectorConfig()
            return self._config

    def save_config(self, config: ConnectorConfig) -> None:
        """保存配置"""
        try:
            config_dict = self._config_to_dict(config)

            # 确保目录存在
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Config saved to: {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def reload_config(self) -> ConnectorConfig:
        """重新加载配置"""
        self._config = None
        self._config_cache.clear()
        return self.load_config()

    def _dict_to_config(self, data: Dict[str, Any]) -> ConnectorConfig:
        """字典转配置对象"""
        # 解析适配器配置
        adapters = []
        for adapter_data in data.get('adapters', []):
            adapter_config = AdapterConfig(
                name=adapter_data['name'],
                type=AdapterType(adapter_data['type']),
                class_path=adapter_data['class_path'],
                config=adapter_data.get('config', {}),
                enabled=adapter_data.get('enabled', True),
                priority=adapter_data.get('priority', 0),
                dependencies=adapter_data.get('dependencies', [])
            )
            adapters.append(adapter_config)

        # 解析各个配置段
        router_config = RouterConfig(**data.get('router', {}))
        health_check_config = HealthCheckConfig(**data.get('health_check', {}))
        monitoring_config = MonitoringConfig(**data.get('monitoring', {}))

        return ConnectorConfig(
            adapters=adapters,
            router=router_config,
            performance=performance_config,
            health_check=health_check_config,
            monitoring=monitoring_config
        )

    def _config_to_dict(self, config: ConnectorConfig) -> Dict[str, Any]:
        """配置对象转字典（使用 dataclasses.asdict）"""
        from dataclasses import asdict

        result = asdict(config)

        # 处理枚举类型
        for adapter in result.get('adapters', []):
            if 'type' in adapter and hasattr(adapter['type'], 'value'):
                adapter['type'] = adapter['type'].value

        return result

    def get_adapter_config(self, adapter_name: str) -> Optional[AdapterConfig]:
        """获取特定适配器的配置"""
        config = self.load_config()
        for adapter in config.adapters:
            if adapter.name == adapter_name:
                return adapter
        return None

    def add_adapter_config(self, adapter_config: AdapterConfig) -> None:
        """添加适配器配置"""
        config = self.load_config()

        # 检查是否已存在同名适配器
        for i, existing_adapter in enumerate(config.adapters):
            if existing_adapter.name == adapter_config.name:
                config.adapters[i] = adapter_config
                break
        else:
            config.adapters.append(adapter_config)

        self.save_config(config)

    def remove_adapter_config(self, adapter_name: str) -> bool:
        """移除适配器配置"""
        config = self.load_config()

        for i, adapter in enumerate(config.adapters):
            if adapter.name == adapter_name:
                config.adapters.pop(i)
                self.save_config(config)
                return True

        return False

    def validate_config(self, config: ConnectorConfig) -> Tuple[bool, List[str]]:
        """验证配置的有效性"""
        errors = []

        # 验证适配器配置
        adapter_names = set()
        for adapter in config.adapters:
            if adapter.name in adapter_names:
                errors.append(f"Duplicate adapter name: {adapter.name}")
            adapter_names.add(adapter.name)

            if not adapter.class_path:
                errors.append(f"Missing class_path for adapter: {adapter.name}")

        # 验证网络配置
        if config.network.port < 1 or config.network.port > 65535:
            errors.append(f"Invalid port number: {config.network.port}")

        # 验证性能配置
        if config.performance.queue_size < 1:
            errors.append(f"Invalid queue_size: {config.performance.queue_size}")

        if config.performance.batch_size < 1:
            errors.append(f"Invalid batch_size: {config.performance.batch_size}")

        # 验证健康检查配置
        if config.health_check.enabled and config.health_check.interval < 1:
            errors.append(f"Invalid health_check interval: {config.health_check.interval}")

        return len(errors) == 0, errors

    def get_example_config(self) -> Dict[str, Any]:
        """获取示例配置"""
        return {
            "adapters": [
                {
                    "name": "vllm",
                    "type": "inference",
                    "class_path": "xconnector.adapters.inference.vllm_adapter.VLLMAdapter",
                    "config": {
                        "model_path": "facebook/opt-125m",
                        "tensor_parallel_size": 1
                    },
                    "enabled": True,
                    "priority": 0,
                    "dependencies": ["torch", "vllm"]
                },
                {
                    "name": "lmcache",
                    "type": "cache",
                    "class_path": "xconnector.adapters.cache.lmcache_adapter.LMCacheAdapter",
                    "config": {
                        "cache_size": "100MB",
                        "cache_type": "memory"
                    },
                    "enabled": True,
                    "priority": 1,
                    "dependencies": ["lmcache"]
                }
            ]
        }
