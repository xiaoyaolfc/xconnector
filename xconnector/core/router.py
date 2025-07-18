# xconnector/core/router.py
import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from xconnector.interfaces.base_interface import BaseInterface
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"


@dataclass
class RouteRule:
    """路由规则配置"""
    source_type: str
    target_type: str
    handler: Callable
    priority: int = 0
    enabled: bool = True
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    timeout: Optional[float] = None
    retry_count: int = 0
    circuit_breaker_enabled: bool = False


@dataclass
class Message:
    """消息对象"""
    source: str
    target: str
    method: str
    args: Tuple
    kwargs: Dict[str, Any]
    message_id: Optional[str] = None
    priority: int = 0
    timeout: Optional[float] = None


class LoadBalancer(ABC):
    """负载均衡器基类"""

    def __init__(self, targets: List[str]):
        self.targets = targets
        self.connection_counts = {target: 0 for target in targets}
        self.weights = {target: 1 for target in targets}
        self.health_status = {target: True for target in targets}

    @abstractmethod
    def select_target(self) -> Optional[str]:
        """选择目标"""
        pass

    def update_connection_count(self, target: str, delta: int):
        """更新连接数"""
        if target in self.connection_counts:
            self.connection_counts[target] += delta

    def set_weight(self, target: str, weight: int):
        """设置权重"""
        if target in self.weights:
            self.weights[target] = weight

    def set_health_status(self, target: str, is_healthy: bool):
        """设置健康状态"""
        if target in self.health_status:
            self.health_status[target] = is_healthy

    def get_healthy_targets(self) -> List[str]:
        """获取健康的目标"""
        return [target for target in self.targets if self.health_status[target]]


class RoundRobinLoadBalancer(LoadBalancer):
    """轮询负载均衡器"""

    def __init__(self, targets: List[str]):
        super().__init__(targets)
        self.current_index = 0

    def select_target(self) -> Optional[str]:
        healthy_targets = self.get_healthy_targets()
        if not healthy_targets:
            return None

        target = healthy_targets[self.current_index % len(healthy_targets)]
        self.current_index = (self.current_index + 1) % len(healthy_targets)
        return target


class RandomLoadBalancer(LoadBalancer):
    """随机负载均衡器"""

    def select_target(self) -> Optional[str]:
        healthy_targets = self.get_healthy_targets()
        if not healthy_targets:
            return None
        return random.choice(healthy_targets)


class LeastConnectionsLoadBalancer(LoadBalancer):
    """最少连接负载均衡器"""

    def select_target(self) -> Optional[str]:
        healthy_targets = self.get_healthy_targets()
        if not healthy_targets:
            return None

        # 选择连接数最少的目标
        min_connections = min(self.connection_counts[target] for target in healthy_targets)
        candidates = [target for target in healthy_targets
                      if self.connection_counts[target] == min_connections]
        return random.choice(candidates)


class WeightedRoundRobinLoadBalancer(LoadBalancer):
    """加权轮询负载均衡器"""

    def __init__(self, targets: List[str]):
        super().__init__(targets)
        self.current_weights = {target: 0 for target in targets}

    def select_target(self) -> Optional[str]:
        healthy_targets = self.get_healthy_targets()
        if not healthy_targets:
            return None

        # 更新当前权重
        for target in healthy_targets:
            self.current_weights[target] += self.weights[target]

        # 选择权重最高的目标
        max_weight = max(self.current_weights[target] for target in healthy_targets)
        selected_target = next(target for target in healthy_targets
                               if self.current_weights[target] == max_weight)

        # 减少选中目标的权重
        total_weight = sum(self.weights[target] for target in healthy_targets)
        self.current_weights[selected_target] -= total_weight

        return selected_target


class HealthBasedLoadBalancer(LoadBalancer):
    """基于健康状态的负载均衡器"""

    def __init__(self, targets: List[str]):
        super().__init__(targets)
        self.response_times = {target: 0.0 for target in targets}

    def select_target(self) -> Optional[str]:
        healthy_targets = self.get_healthy_targets()
        if not healthy_targets:
            return None

        # 选择响应时间最短的目标
        min_response_time = min(self.response_times[target] for target in healthy_targets)
        candidates = [target for target in healthy_targets
                      if self.response_times[target] == min_response_time]
        return random.choice(candidates)

    def update_response_time(self, target: str, response_time: float):
        """更新响应时间"""
        if target in self.response_times:
            # 使用简单的指数移动平均
            self.response_times[target] = (self.response_times[target] * 0.8 +
                                           response_time * 0.2)


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open

    def call_allowed(self) -> bool:
        """判断是否允许调用"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if asyncio.get_event_loop().time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True

    def record_success(self):
        """记录成功调用"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """记录失败调用"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class Router:
    """消息路由器"""

    def __init__(self):
        self.routes: Dict[str, RouteRule] = {}
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.adapters: Dict[str, BaseInterface] = {}
        self.route_stats: Dict[str, Dict[str, Any]] = {}

    def add_route(self, source_type: str, target_type: str, handler: Callable,
                  priority: int = 0, **kwargs) -> None:
        """
        添加路由规则

        Args:
            source_type: 源适配器类型
            target_type: 目标适配器类型
            handler: 路由处理函数
            priority: 优先级
            **kwargs: 其他路由配置
        """
        route_key = f"{source_type}->{target_type}"

        route_rule = RouteRule(
            source_type=source_type,
            target_type=target_type,
            handler=handler,
            priority=priority,
            **kwargs
        )

        self.routes[route_key] = route_rule

        # 初始化统计信息
        self.route_stats[route_key] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "last_request_time": 0
        }

        logger.info(f"Added route: {route_key}")

    def remove_route(self, source_type: str, target_type: str) -> None:
        """删除路由规则"""
        route_key = f"{source_type}->{target_type}"

        if route_key in self.routes:
            del self.routes[route_key]
            if route_key in self.route_stats:
                del self.route_stats[route_key]
            logger.info(f"Removed route: {route_key}")
        else:
            logger.warning(f"Route not found: {route_key}")

    def register_route(self, source: str, target: str, handler: Callable) -> None:
        """
        注册路由规则 (兼容性方法)

        Args:
            source: 源适配器名称
            target: 目标适配器名称
            handler: 路由处理函数
        """
        self.add_route(source, target, handler)

    def register_adapter(self, name: str, adapter: BaseInterface) -> None:
        """
        注册适配器

        Args:
            name: 适配器名称
            adapter: 适配器实例
        """
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")

    def unregister_adapter(self, name: str) -> None:
        """注销适配器"""
        if name in self.adapters:
            del self.adapters[name]
            logger.info(f"Unregistered adapter: {name}")

    def add_load_balancer(self, endpoint: str, strategy: LoadBalanceStrategy,
                          targets: List[str]) -> None:
        """
        添加负载均衡策略

        Args:
            endpoint: 端点名称
            strategy: 负载均衡策略
            targets: 目标列表
        """
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            self.load_balancers[endpoint] = RoundRobinLoadBalancer(targets)
        elif strategy == LoadBalanceStrategy.RANDOM:
            self.load_balancers[endpoint] = RandomLoadBalancer(targets)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            self.load_balancers[endpoint] = LeastConnectionsLoadBalancer(targets)
        elif strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            self.load_balancers[endpoint] = WeightedRoundRobinLoadBalancer(targets)
        elif strategy == LoadBalanceStrategy.HEALTH_BASED:
            self.load_balancers[endpoint] = HealthBasedLoadBalancer(targets)
        else:
            raise ValueError(f"Unsupported load balance strategy: {strategy}")

        logger.info(f"Added load balancer for {endpoint}: {strategy.value}")

    def get_load_balancer(self, endpoint: str) -> Optional[LoadBalancer]:
        """获取负载均衡器"""
        return self.load_balancers.get(endpoint)

    def enable_circuit_breaker(self, endpoint: str, failure_threshold: int = 5,
                               timeout: float = 60.0) -> None:
        """
        启用熔断器

        Args:
            endpoint: 端点名称
            failure_threshold: 失败阈值
            timeout: 超时时间
        """
        self.circuit_breakers[endpoint] = CircuitBreaker(failure_threshold, timeout)
        logger.info(f"Enabled circuit breaker for {endpoint}")

    async def route(self, source: str, target: str, method: str, *args, **kwargs) -> Any:
        """
        路由消息到目标适配器

        Args:
            source: 源适配器名称
            target: 目标适配器名称
            method: 方法名称
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 方法执行结果
        """
        route_key = f"{source}->{target}"

        # 检查路由是否存在
        if route_key not in self.routes:
            raise ValueError(f"Route not found: {route_key}")

        route_rule = self.routes[route_key]

        # 检查路由是否启用
        if not route_rule.enabled:
            raise ValueError(f"Route disabled: {route_key}")

        # 检查熔断器
        if route_rule.circuit_breaker_enabled and route_key in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[route_key]
            if not circuit_breaker.call_allowed():
                raise Exception(f"Circuit breaker open for route: {route_key}")

        # 获取适配器
        source_adapter = self.adapters.get(source)
        target_adapter = self.adapters.get(target)

        if not source_adapter:
            raise ValueError(f"Source adapter not found: {source}")
        if not target_adapter:
            raise ValueError(f"Target adapter not found: {target}")

        # 更新统计信息
        stats = self.route_stats[route_key]
        stats["total_requests"] += 1
        stats["last_request_time"] = asyncio.get_event_loop().time()

        # 负载均衡
        actual_target = target
        if route_key in self.load_balancers:
            load_balancer = self.load_balancers[route_key]
            selected_target = load_balancer.select_target()
            if selected_target:
                actual_target = selected_target
                load_balancer.update_connection_count(selected_target, 1)

        start_time = asyncio.get_event_loop().time()

        try:
            # 应用超时
            timeout = route_rule.timeout or kwargs.pop('timeout', None)

            if timeout:
                result = await asyncio.wait_for(
                    route_rule.handler(source_adapter, target_adapter, method, *args, **kwargs),
                    timeout=timeout
                )
            else:
                result = await route_rule.handler(source_adapter, target_adapter, method, *args, **kwargs)

            # 记录成功
            stats["successful_requests"] += 1

            if route_rule.circuit_breaker_enabled and route_key in self.circuit_breakers:
                self.circuit_breakers[route_key].record_success()

            return result

        except Exception as e:
            # 记录失败
            stats["failed_requests"] += 1

            if route_rule.circuit_breaker_enabled and route_key in self.circuit_breakers:
                self.circuit_breakers[route_key].record_failure()

            # 重试逻辑
            if route_rule.retry_count > 0:
                for attempt in range(route_rule.retry_count):
                    try:
                        logger.warning(f"Retrying route {route_key}, attempt {attempt + 1}")

                        if timeout:
                            result = await asyncio.wait_for(
                                route_rule.handler(source_adapter, target_adapter, method, *args, **kwargs),
                                timeout=timeout
                            )
                        else:
                            result = await route_rule.handler(source_adapter, target_adapter, method, *args, **kwargs)

                        stats["successful_requests"] += 1
                        return result

                    except Exception as retry_error:
                        if attempt == route_rule.retry_count - 1:
                            raise retry_error
                        await asyncio.sleep(0.1 * (2 ** attempt))  # 指数退避

            raise e

        finally:
            # 更新负载均衡器连接数
            if route_key in self.load_balancers:
                load_balancer = self.load_balancers[route_key]
                load_balancer.update_connection_count(actual_target, -1)

                # 更新响应时间（如果是健康状态负载均衡器）
                if isinstance(load_balancer, HealthBasedLoadBalancer):
                    response_time = asyncio.get_event_loop().time() - start_time
                    load_balancer.update_response_time(actual_target, response_time)

            # 更新平均响应时间
            response_time = asyncio.get_event_loop().time() - start_time
            stats["avg_response_time"] = (stats["avg_response_time"] * (stats["total_requests"] - 1) +
                                          response_time) / stats["total_requests"]

    async def route_message(self, message: Message) -> Any:
        """
        路由消息对象

        Args:
            message: 消息对象

        Returns:
            Any: 执行结果
        """
        return await self.route(
            message.source,
            message.target,
            message.method,
            *message.args,
            **message.kwargs
        )

    def get_route_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取路由统计信息"""
        return self.route_stats.copy()

    def get_route_info(self, source_type: str, target_type: str) -> Optional[RouteRule]:
        """获取路由信息"""
        route_key = f"{source_type}->{target_type}"
        return self.routes.get(route_key)

    def list_routes(self) -> List[str]:
        """列出所有路由"""
        return list(self.routes.keys())

    def get_health_status(self) -> Dict[str, Any]:
        """获取路由器健康状态"""
        return {
            "total_routes": len(self.routes),
            "active_routes": len([r for r in self.routes.values() if r.enabled]),
            "load_balancers": len(self.load_balancers),
            "circuit_breakers": len(self.circuit_breakers),
            "registered_adapters": len(self.adapters),
            "route_stats": self.route_stats
        }

    def clear_stats(self) -> None:
        """清空统计信息"""
        for stats in self.route_stats.values():
            stats.update({
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0.0,
                "last_request_time": 0
            })
        logger.info("Route statistics cleared")

    def update_adapter_health(self, adapter_name: str, is_healthy: bool) -> None:
        """更新适配器健康状态"""
        for endpoint, load_balancer in self.load_balancers.items():
            if adapter_name in load_balancer.targets:
                load_balancer.set_health_status(adapter_name, is_healthy)

        logger.info(f"Updated adapter health: {adapter_name} -> {'healthy' if is_healthy else 'unhealthy'}")