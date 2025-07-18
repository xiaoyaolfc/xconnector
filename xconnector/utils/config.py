# xconnector/utils/config.py
import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from xconnector.utils.logging import get_logger

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
class VLLMConfig:
    """VLLM推理引擎配置"""
    model_path: str = "facebook/opt-125m"
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    trust_remote_code: bool = False
    revision: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    max_paddings: int = 256
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: float = 0
    device: str = "auto"
    ray_workers_use_nsight: bool = False
    pipeline_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    tokenizer_pool_type: str = "ray"
    tokenizer_pool_extra_config: Optional[Dict[str, Any]] = None
    enable_prefix_caching: bool = False
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = False
    enable_chunked_prefill: bool = False
    max_num_on_the_fly_seq_groups: int = 64
    guided_decoding_backend: str = "outlines"
    # 扩展配置
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LMCacheConfig:
    """LMCache缓存配置"""
    cache_size: int = 1000
    cache_type: str = "local"  # local, redis, distributed
    redis_url: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_max_connections: int = 10
    ttl: int = 3600  # 缓存过期时间（秒）
    max_memory_usage: float = 0.8  # 最大内存使用率
    eviction_policy: str = "lru"  # lru, lfu, fifo
    compression: bool = True
    compression_algorithm: str = "gzip"  # gzip, lz4, zstd
    persistence: bool = False
    persistence_path: Optional[str] = None
    batch_size: int = 32
    prefetch_size: int = 64
    enable_metrics: bool = True
    metrics_interval: int = 60
    # 分布式缓存配置
    distributed_nodes: List[str] = field(default_factory=list)
    replication_factor: int = 1
    consistency_level: str = "eventual"  # strong, eventual
    # 扩展配置
    custom_config: Dict[str, Any] = field(default_factory=dict)


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
class SecurityConfig:
    """安全配置"""
    enable_auth: bool = False
    auth_method: str = "api_key"  # api_key, jwt, oauth
    api_keys: List[str] = field(default_factory=list)
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    # TLS配置
    enable_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_ca_path: Optional[str] = None
    # 访问控制
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    # 速率限制
    enable_rate_limiting: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 秒
    rate_limit_storage: str = "memory"  # memory, redis


@dataclass
class NetworkConfig:
    """网络配置"""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    max_connections: int = 1000
    keepalive_timeout: int = 30
    client_timeout: int = 60
    # 代理配置
    proxy_enabled: bool = False
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    # 网络优化
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True
    socket_reuse_port: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    queue_size: int = 1000
    batch_size: int = 32
    max_concurrent_requests: int = 100
    request_timeout: int = 300
    # 资源限制
    max_memory_usage: float = 0.8  # 最大内存使用率
    max_cpu_usage: float = 0.8  # 最大CPU使用率
    max_disk_usage: float = 0.8  # 最大磁盘使用率
    # 异步配置
    async_pool_size: int = 100
    async_timeout: int = 300
    # 缓存配置
    enable_request_cache: bool = True
    request_cache_size: int = 1000
    request_cache_ttl: int = 300


@dataclass
class ConnectorConfig:
    """XConnector主配置类"""
    # 核心配置
    adapters: List[AdapterConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)

    # 内置适配器配置
    vllm_config: VLLMConfig = field(default_factory=VLLMConfig)
    lmcache_config: LMCacheConfig = field(default_factory=LMCacheConfig)

    # 系统配置
    network: NetworkConfig = field(default_factory=NetworkConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

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
        vllm_config = VLLMConfig(**data.get('vllm', {}))
        lmcache_config = LMCacheConfig(**data.get('lmcache', {}))
        router_config = RouterConfig(**data.get('router', {}))
        network_config = NetworkConfig(**data.get('network', {}))
        performance_config = PerformanceConfig(**data.get('performance', {}))
        health_check_config = HealthCheckConfig(**data.get('health_check', {}))
        monitoring_config = MonitoringConfig(**data.get('monitoring', {}))
        security_config = SecurityConfig(**data.get('security', {}))

        return ConnectorConfig(
            adapters=adapters,
            vllm_config=vllm_config,
            lmcache_config=lmcache_config,
            router=router_config,
            network=network_config,
            performance=performance_config,
            health_check=health_check_config,
            monitoring=monitoring_config,
            security=security_config
        )

    def _config_to_dict(self, config: ConnectorConfig) -> Dict[str, Any]:
        """配置对象转字典"""
        return {
            'adapters': [
                {
                    'name': adapter.name,
                    'type': adapter.type.value,
                    'class_path': adapter.class_path,
                    'config': adapter.config,
                    'enabled': adapter.enabled,
                    'priority': adapter.priority,
                    'dependencies': adapter.dependencies
                }
                for adapter in config.adapters
            ],
            'vllm': {
                'model_path': config.vllm_config.model_path,
                'tensor_parallel_size': config.vllm_config.tensor_parallel_size,
                'max_model_len': config.vllm_config.max_model_len,
                'gpu_memory_utilization': config.vllm_config.gpu_memory_utilization,
                'enforce_eager': config.vllm_config.enforce_eager,
                'trust_remote_code': config.vllm_config.trust_remote_code,
                'revision': config.vllm_config.revision,
                'tokenizer_mode': config.vllm_config.tokenizer_mode,
                'skip_tokenizer_init': config.vllm_config.skip_tokenizer_init,
                'load_format': config.vllm_config.load_format,
                'dtype': config.vllm_config.dtype,
                'seed': config.vllm_config.seed,
                'max_num_seqs': config.vllm_config.max_num_seqs,
                'max_num_batched_tokens': config.vllm_config.max_num_batched_tokens,
                'max_paddings': config.vllm_config.max_paddings,
                'block_size': config.vllm_config.block_size,
                'swap_space': config.vllm_config.swap_space,
                'cpu_offload_gb': config.vllm_config.cpu_offload_gb,
                'device': config.vllm_config.device,
                'ray_workers_use_nsight': config.vllm_config.ray_workers_use_nsight,
                'pipeline_parallel_size': config.vllm_config.pipeline_parallel_size,
                'max_parallel_loading_workers': config.vllm_config.max_parallel_loading_workers,
                'disable_custom_all_reduce': config.vllm_config.disable_custom_all_reduce,
                'tokenizer_pool_size': config.vllm_config.tokenizer_pool_size,
                'tokenizer_pool_type': config.vllm_config.tokenizer_pool_type,
                'tokenizer_pool_extra_config': config.vllm_config.tokenizer_pool_extra_config,
                'enable_prefix_caching': config.vllm_config.enable_prefix_caching,
                'disable_sliding_window': config.vllm_config.disable_sliding_window,
                'use_v2_block_manager': config.vllm_config.use_v2_block_manager,
                'enable_chunked_prefill': config.vllm_config.enable_chunked_prefill,
                'max_num_on_the_fly_seq_groups': config.vllm_config.max_num_on_the_fly_seq_groups,
                'guided_decoding_backend': config.vllm_config.guided_decoding_backend,
                'custom_config': config.vllm_config.custom_config,
            },
            'lmcache': {
                'cache_size': config.lmcache_config.cache_size,
                'cache_type': config.lmcache_config.cache_type,
                'redis_url': config.lmcache_config.redis_url,
                'redis_password': config.lmcache_config.redis_password,
                'redis_db': config.lmcache_config.redis_db,
                'redis_max_connections': config.lmcache_config.redis_max_connections,
                'ttl': config.lmcache_config.ttl,
                'max_memory_usage': config.lmcache_config.max_memory_usage,
                'eviction_policy': config.lmcache_config.eviction_policy,
                'compression': config.lmcache_config.compression,
                'compression_algorithm': config.lmcache_config.compression_algorithm,
                'persistence': config.lmcache_config.persistence,
                'persistence_path': config.lmcache_config.persistence_path,
                'batch_size': config.lmcache_config.batch_size,
                'prefetch_size': config.lmcache_config.prefetch_size,
                'enable_metrics': config.lmcache_config.enable_metrics,
                'metrics_interval': config.lmcache_config.metrics_interval,
                'distributed_nodes': config.lmcache_config.distributed_nodes,
                'replication_factor': config.lmcache_config.replication_factor,
                'consistency_level': config.lmcache_config.consistency_level,
                'custom_config': config.lmcache_config.custom_config,
            },
            'router': {
                'load_balance_strategy': config.router.load_balance_strategy,
                'request_timeout': config.router.request_timeout,
                'max_retries': config.router.max_retries,
                'retry_delay': config.router.retry_delay,
                'circuit_breaker_threshold': config.router.circuit_breaker_threshold,
                'circuit_breaker_timeout': config.router.circuit_breaker_timeout,
                'enable_fallback': config.router.enable_fallback,
                'fallback_strategy': config.router.fallback_strategy,
                'routing_rules': config.router.routing_rules,
                'custom_config': config.router.custom_config,
            },
            'network': {
                'host': config.network.host,
                'port': config.network.port,
                'workers': config.network.workers,
                'max_connections': config.network.max_connections,
                'keepalive_timeout': config.network.keepalive_timeout,
                'client_timeout': config.network.client_timeout,
                'proxy_enabled': config.network.proxy_enabled,
                'proxy_host': config.network.proxy_host,
                'proxy_port': config.network.proxy_port,
                'proxy_username': config.network.proxy_username,
                'proxy_password': config.network.proxy_password,
                'tcp_nodelay': config.network.tcp_nodelay,
                'tcp_keepalive': config.network.tcp_keepalive,
                'socket_reuse_port': config.network.socket_reuse_port,
            },
            'performance': {
                'queue_size': config.performance.queue_size,
                'batch_size': config.performance.batch_size,
                'max_concurrent_requests': config.performance.max_concurrent_requests,
                'request_timeout': config.performance.request_timeout,
                'max_memory_usage': config.performance.max_memory_usage,
                'max_cpu_usage': config.performance.max_cpu_usage,
                'max_disk_usage': config.performance.max_disk_usage,
                'async_pool_size': config.performance.async_pool_size,
                'async_timeout': config.performance.async_timeout,
                'enable_request_cache': config.performance.enable_request_cache,
                'request_cache_size': config.performance.request_cache_size,
                'request_cache_ttl': config.performance.request_cache_ttl,
            },
            'health_check': {
                'enabled': config.health_check.enabled,
                'interval': config.health_check.interval,
                'timeout': config.health_check.timeout,
                'failure_threshold': config.health_check.failure_threshold,
                'success_threshold': config.health_check.success_threshold,
                'initial_delay': config.health_check.initial_delay,
                'log_results': config.health_check.log_results,
                'enable_auto_recovery': config.health_check.enable_auto_recovery,
                'recovery_delay': config.health_check.recovery_delay,
                'custom_endpoints': config.health_check.custom_endpoints,
            },
            'monitoring': {
                'enabled': config.monitoring.enabled,
                'metrics_enabled': config.monitoring.metrics_enabled,
                'metrics_interval': config.monitoring.metrics_interval,
                'metrics_retention': config.monitoring.metrics_retention,
                'enable_prometheus': config.monitoring.enable_prometheus,
                'prometheus_port': config.monitoring.prometheus_port,
                'enable_grafana': config.monitoring.enable_grafana,
                'grafana_dashboard_path': config.monitoring.grafana_dashboard_path,
                'log_level': config.monitoring.log_level,
                'log_file': config.monitoring.log_file,
                'log_max_size': config.monitoring.log_max_size,
                'log_backup_count': config.monitoring.log_backup_count,
                'log_format': config.monitoring.log_format,
                'enable_alerts': config.monitoring.enable_alerts,
                'alert_endpoints': config.monitoring.alert_endpoints,
                'alert_thresholds': config.monitoring.alert_thresholds,
            },
            'security': {
                'enable_auth': config.security.enable_auth,
                'auth_method': config.security.auth_method,
                'api_keys': config.security.api_keys,
                'jwt_secret': config.security.jwt_secret,
                'jwt_algorithm': config.security.jwt_algorithm,
                'jwt_expiration': config.security.jwt_expiration,
                'enable_tls': config.security.enable_tls,
                'tls_cert_path': config.security.tls_cert_path,
                'tls_key_path': config.security.tls_key_path,
                'tls_ca_path': config.security.tls_ca_path,
                'enable_cors': config.security.enable_cors,
                'cors_origins': config.security.cors_origins,
                'cors_methods': config.security.cors_methods,
                'cors_headers': config.security.cors_headers,
                'enable_rate_limiting': config.security.enable_rate_limiting,
                'rate_limit_requests': config.security.rate_limit_requests,
                'rate_limit_window': config.security.rate_limit_window,
                'rate_limit_storage': config.security.rate_limit_storage,
            }
        }

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