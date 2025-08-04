# XConnector SDK 集成设计方案

## 1. 架构变更概述

### 当前架构（服务模式）
```
┌─────────────────┐    HTTP API     ┌─────────────────┐
│   Dynamo Pod    │ ──────────────► │ XConnector Pod  │
│                 │                 │                 │
│ - Worker        │                 │ - Service       │
│ - Extension     │                 │ - Adapters      │
└─────────────────┘                 └─────────────────┘
```

### 目标架构（SDK模式）
```
┌─────────────────────────────────────┐
│           Dynamo Pod                │
│                                     │
│ ┌─────────────┐  ┌─────────────────┐│
│ │   Worker    │  │ XConnector SDK  ││
│ │             │  │                 ││
│ │ - Model     │◄─┤ - Core          ││
│ │ - KV Cache  │  │ - Adapters      ││
│ └─────────────┘  │ - Router        ││
│                  └─────────────────┘│
└─────────────────────────────────────┘
```

## 2. 核心修改方案

### 2.1 SDK 入口设计

#### 主要入口类
```python
# xconnector/sdk/__init__.py
class XConnectorSDK:
    """XConnector SDK 主入口"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.connector = XConnector(config)
        self.initialized = False
    
    async def initialize(self) -> bool:
        """初始化SDK"""
        pass
    
    async def load_adapters(self, adapters_config: List[AdapterConfig]) -> bool:
        """加载适配器"""
        pass
    
    def get_kv_handler(self) -> 'KVCacheHandler':
        """获取KV缓存处理器"""
        pass
    
    def get_distributed_handler(self) -> 'DistributedHandler':
        """获取分布式处理器"""
        pass
```

#### 便捷工厂函数
```python
# xconnector/sdk/factory.py
def create_xconnector_sdk(
    mode: str = "embedded",
    config: Dict[str, Any] = None
) -> XConnectorSDK:
    """创建XConnector SDK实例"""
    pass

def create_kv_cache_sdk(
    cache_backend: str = "lmcache",
    config: Dict[str, Any] = None
) -> 'KVCacheSDK':
    """创建专用的KV缓存SDK"""
    pass
```

### 2.2 Dynamo集成层设计

#### 集成接口
```python
# xconnector/integrations/dynamo/sdk_integration.py
class DynamoXConnectorIntegration:
    """Dynamo与XConnector的SDK集成"""
    
    def __init__(self, dynamo_config: Dict[str, Any]):
        self.sdk = None
        self.config = dynamo_config
    
    async def setup(self) -> bool:
        """设置集成"""
        pass
    
    def inject_into_worker(self, worker_instance) -> None:
        """注入到Worker实例"""
        pass
    
    def wrap_kv_methods(self, worker_instance) -> None:
        """包装KV缓存方法"""
        pass
```

#### Worker方法包装器
```python
# xconnector/integrations/dynamo/worker_wrapper.py
class VLLMWorkerWrapper:
    """VLLM Worker包装器"""
    
    def __init__(self, original_worker, xconnector_sdk):
        self.worker = original_worker
        self.sdk = xconnector_sdk
    
    async def recv_kv_caches(self, *args, **kwargs):
        """包装的KV接收方法"""
        pass
    
    async def send_kv_caches(self, *args, **kwargs):
        """包装的KV发送方法"""
        pass
```

### 2.3 配置管理重构

#### 统一配置格式
```python
# xconnector/sdk/config.py
@dataclass
class SDKConfig:
    """SDK配置"""
    
    # 核心配置
    mode: str = "embedded"  # embedded, hybrid
    enable_distributed: bool = True
    enable_caching: bool = True
    
    # 适配器配置
    adapters: List[AdapterConfig] = field(default_factory=list)
    
    # 集成配置
    integration: Dict[str, Any] = field(default_factory=dict)
    
    # 性能配置
    performance: Dict[str, Any] = field(default_factory=dict)

def load_config_from_dynamo(dynamo_config: Dict[str, Any]) -> SDKConfig:
    """从Dynamo配置加载SDK配置"""
    pass
```

## 3. 具体实现计划

### 3.1 第一阶段：SDK核心重构

#### 文件结构调整
```
xconnector/
├── sdk/
│   ├── __init__.py          # SDK主入口
│   ├── factory.py           # 工厂函数
│   ├── config.py            # 配置管理
│   ├── handlers/            # 功能处理器
│   │   ├── kv_cache.py      # KV缓存处理
│   │   ├── distributed.py   # 分布式处理
│   │   └── monitoring.py    # 监控处理
│   └── exceptions.py        # SDK异常
├── integrations/
│   └── dynamo/
│       ├── autopatch.py          # 自动monkey patch
│       ├── config_detector.py    # 配置文件检测
│       ├── worker_injector.py    # Worker方法注入
│       └── minimal_sdk.py        # 最小SDK封装
└── core/
    ├── connector.py         # 保持兼容，重构内部实现
    └── ...                  # 其他核心组件
```

#### 核心组件重构
1. **XConnector类改造**
   - 保持现有API兼容性
   - 内部支持SDK模式
   - 简化生命周期管理

2. **适配器加载机制**
   - 支持动态加载
   - 简化依赖管理
   - 增强错误处理

3. **路由器简化**
   - 移除HTTP相关代码
   - 专注进程内路由
   - 优化性能

### 3.2 第二阶段：Dynamo集成

#### 集成点设计
1. **初始化集成**
   ```python
   # 在Dynamo Worker初始化时
   # worker.py
    import xconnector.integrations.dynamo.autopatch  # 添加这一行
    
    # 其他原有代码保持不变
    import os
    import sys
    from vllm_worker import VLLMWorker
    
    def main():
        # 原有逻辑不变
        config = load_config()
        worker = VLLMWorker(config)
        worker.start()
    
    if __name__ == "__main__":
        main()
   ```

2. **修改配置文件**
   ```yaml
   # 自动包装关键方法
   Common:
      model: /your/model/path
      # 其他原有配置...
      
      # 新增XConnector配置
      xconnector:
        enabled: true
        adapters:
          - name: lmcache
            type: cache
    
    VllmWorker:
      # 原有配置保持不变
      enable-prefix-caching: true
   ```


### 3.3 第三阶段：优化与完善

#### 性能优化
1. **内存共享**
   - 适配器间共享内存
   - 减少数据拷贝
   - 优化缓存访问

2. **异步优化**
   - 非阻塞操作
   - 并发处理
   - 批量操作

#### 监控集成
1. **指标收集**
   - 集成到Dynamo监控
   - 统一指标格式
   - 实时性能监控

2. **错误处理**
   - 优雅降级
   - 错误隔离
   - 详细日志

## 4. 配置示例

### 4.1 Dynamo配置示例
```yaml
# dynamo_config.yaml
Common:
  model: /data/model/DeepSeek-R1-Distill-Llama-70B
  
  # XConnector SDK配置
  xconnector:
    enabled: true
    mode: embedded
    adapters:
      - name: lmcache
        type: cache
        config:
          storage_backend: memory
          max_cache_size: 2048
      - name: vllm
        type: inference
        config:
          enable_prefix_caching: true

VllmWorker:
  enable-prefix-caching: true
  # 其他Worker配置...
```

### 4.2 SDK直接使用示例
```python
# 在Dynamo代码中直接使用
from xconnector.sdk import create_xconnector_sdk

# 初始化SDK
sdk = create_xconnector_sdk(
    mode="embedded",
    config={
        'adapters': [
            {
                'name': 'lmcache',
                'type': 'cache',
                'config': {'storage_backend': 'memory'}
            }
        ]
    }
)

# 在Worker中使用
class VLLMWorkerWithXConnector(VLLMWorker):
    def __init__(self, config):
        super().__init__(config)
        self.xconnector_sdk = sdk
        
    async def recv_kv_caches(self, *args, **kwargs):
        # 通过SDK处理KV缓存
        result = await self.xconnector_sdk.handle_kv_receive(*args, **kwargs)
        if result.cache_hit:
            return result.data
        
        # 回退到原始逻辑
        return await super().recv_kv_caches(*args, **kwargs)
```

## 5. 迁移计划

### 5.1 兼容性保证
- 保持现有Docker部署方式可用
- 支持混合模式（部分SDK，部分服务）
- 提供迁移工具和指南

### 5.2 分步迁移
1. **Phase 1**: SDK框架搭建，基础功能迁移
2. **Phase 2**: Dynamo集成，功能验证
3. **Phase 3**: 性能优化，完整替换
4. **Phase 4**: 清理旧代码，文档完善

### 5.3 测试策略
- 单元测试：SDK各组件
- 集成测试：与Dynamo的集成
- 性能测试：对比服务模式
- 兼容测试：确保向后兼容

## 5.4 pip包生成与发布

### 1. 安装构建工具
```bash
pip install build twine
```

### 2. 构建包
```bash
python -m build
```

### 3. 会生成dist目录
```bash
dist/
├── xconnector-1.0.0-py3-none-any.whl
└── xconnector-1.0.0.tar.gz
```

### 4. 本地安装测试
```bash
pip install dist/xconnector-1.0.0-py3-none-any.whl
```
