# xconnector/config.py
import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class VLLMConfig:
    """VLLM配置"""
    model_path: str = "facebook/opt-125m"
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    
@dataclass
class LMCacheConfig:
    """LMCache配置"""
    cache_size: int = 1000
    cache_type: str = "local"  # local, redis, distributed
    redis_url: Optional[str] = None
    ttl: int = 3600  # 缓存过期时间
    
@dataclass
class DynamoConfig:
    """Dynamo配置"""
    max_workers: int = 4
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, cache_aware
    request_timeout: int = 300
    health_check_interval: int = 30
    
@dataclass
class XConnectorConfig:
    """XConnector总配置"""
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    lmcache: LMCacheConfig = field(default_factory=LMCacheConfig)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    
    # 网络配置
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 性能配置
    queue_size: int = 1000
    batch_size: int = 32
    max_concurrent_requests: int = 100

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[XConnectorConfig] = None
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 查找顺序：环境变量 -> 当前目录 -> 用户目录 -> 系统目录
        paths = [
            os.environ.get('XCONNECTOR_CONFIG'),
            './xconnector.yaml',
            '~/.xconnector/config.yaml',
            '/etc/xconnector/config.yaml'
        ]
        
        for path in paths:
            if path and Path(path).expanduser().exists():
                return str(Path(path).expanduser())
        
        return './xconnector.yaml'
    
    def load_config(self) -> XConnectorConfig:
        """加载配置"""
        if self._config is not None:
            return self._config
        
        if not Path(self.config_path).exists():
            print(f"Config file not found: {self.config_path}, using defaults")
            self._config = XConnectorConfig()
            return self._config
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._config = self._dict_to_config(data)
            return self._config
            
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            self._config = XConnectorConfig()
            return self._config
    
    def save_config(self, config: XConnectorConfig) -> None:
        """保存配置"""
        config_dict = self._config_to_dict(config)
        
        # 确保目录存在
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> XConnectorConfig:
        """字典转配置对象"""
        vllm_config = VLLMConfig(**data.get('vllm', {}))
        lmcache_config = LMCacheConfig(**data.get('lmcache', {}))
        dynamo_config = DynamoConfig(**data.get('dynamo', {}))
        
        # 移除嵌套配置
        main_config = {k: v for k, v in data.items() 
                      if k not in ['vllm', 'lmcache', 'dynamo']}
        
        return XConnectorConfig(
            vllm=vllm_config,
            lmcache=lmcache_config,
            dynamo=dynamo_config,
            **main_config
        )
    
    def _config_to_dict(self, config: XConnectorConfig) -> Dict[str, Any]:
        """配置对象转字典"""
        return {
            'vllm': {
                'model_path': config.vllm.model_path,
                'tensor_parallel_size': config.vllm.tensor_parallel_size,
                'max_model_len': config.vllm.max_model_len,
                'gpu_memory_utilization': config.vllm.gpu_memory_utilization,
                'enforce_eager': config.vllm.enforce_eager,
            },
            'lmcache': {
                'cache_size': config.lmcache.cache_size,
                'cache_type': config.lmcache.cache_type,
                'redis_url': config.lmcache.redis_url,
                'ttl': config.lmcache.ttl,
            },
            'dynamo': {
                'max_workers': config.dynamo.max_workers,
                'load_balance_strategy': config.dynamo.load_balance_strategy,
                'request_timeout': config.dynamo.request_timeout,
                'health_check_interval': config.dynamo.health_check_interval,
            },
            'host': config.host,
            'port': config.port,
            'workers': config.workers,
            'log_level': config.log_level,
            'log_file': config.log_file,
            'queue_size': config.queue_size,
            'batch_size': config.batch_size,
            'max_concurrent_requests': config.max_concurrent_requests,
        }

# 全局配置实例
config_manager = ConfigManager()

def get_config() -> XConnectorConfig:
    """获取全局配置"""
    return config_manager.load_config()