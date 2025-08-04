# xconnector/config.py
"""
XConnector 统一配置定义

所有配置类的统一定义位置，避免循环导入
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# === 基础枚举定义 ===

class AdapterType(Enum):
    """适配器类型枚举"""
    INFERENCE = "inference"
    CACHE = "cache"
    DISTRIBUTED = "distributed"


class SDKMode(Enum):
    """SDK运行模式"""
    EMBEDDED = "embedded"
    HYBRID = "hybrid"


# === 基础配置类 ===

@dataclass
class AdapterConfig:
    """适配器配置"""
    name: str
    type: AdapterType
    class_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = AdapterType(self.type)


@dataclass
class RouterConfig:
    """路由器配置"""
    load_balance_strategy: str = "round_robin"
    request_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    enable_fallback: bool = True
    fallback_strategy: str = "default"
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    enabled: bool = True
    interval: int = 30
    timeout: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    initial_delay: int = 0
    log_results: bool = True
    enable_auto_recovery: bool = True
    recovery_delay: int = 60
    custom_endpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_enabled: bool = True
    metrics_interval: int = 60
    metrics_retention: int = 86400
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_grafana: bool = False
    grafana_dashboard_path: Optional[str] = None
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_max_size: int = 100
    log_backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_alerts: bool = False
    alert_endpoints: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)


# === 主配置类 ===

@dataclass
class ConnectorConfig:
    """XConnector主配置类"""
    adapters: List[AdapterConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def enable_health_check(self) -> bool:
        return self.health_check.enabled

    @property
    def health_check_interval(self) -> int:
        return self.health_check.interval

    @property
    def log_health_check(self) -> bool:
        return self.health_check.log_results

    @property
    def log_level(self) -> str:
        return self.monitoring.log_level

    @property
    def log_file(self) -> Optional[str]:
        return self.monitoring.log_file


@dataclass
class SDKConfig:
    """SDK配置类"""
    mode: SDKMode = SDKMode.EMBEDDED
    enable_kv_cache: bool = True
    enable_distributed: bool = True
    enable_monitoring: bool = True
    adapters: List[Dict[str, Any]] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

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

    performance: Dict[str, Any] = field(default_factory=lambda: {
        "async_mode": True,
        "batch_processing": True,
        "memory_optimization": True,
        "max_concurrent_requests": 1000
    })

    error_handling: Dict[str, Any] = field(default_factory=lambda: {
        "graceful_degradation": True,
        "error_isolation": True,
        "fallback_enabled": True,
        "max_retry_attempts": 3
    })

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = SDKMode(self.mode.lower())