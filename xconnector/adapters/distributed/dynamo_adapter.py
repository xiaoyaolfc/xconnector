# xconnector/adapters/distributed/dynamo_adapter.py
"""
Dynamo Adapter for XConnector

This adapter provides integration between XConnector and AI-Dynamo framework,
enabling distributed inference coordination and resource management.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from xconnector.adapters.base_adapter import BaseAdapter
from xconnector.interfaces.base_interface import (
    AdapterStatus,
    HealthStatus,
    HealthCheckResult,
    Capability,
)
from xconnector.utils.xconnector_logging import get_logger

logger = get_logger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    TERMINATING = "terminating"


@dataclass
class WorkerInfo:
    """Worker information"""
    worker_id: str
    model_name: str
    gpu_memory: int
    status: WorkerStatus
    endpoint: str
    registered_at: datetime
    last_heartbeat: datetime
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingPolicy:
    """Routing policy configuration"""
    strategy: str = "least_loaded"  # least_loaded, round_robin, affinity
    affinity_key: Optional[str] = None
    max_requests_per_worker: int = 100
    health_check_interval: int = 30
    unhealthy_threshold: int = 3


class DynamoAdapter(BaseAdapter):
    """
    Dynamo Adapter for XConnector

    Provides integration with AI-Dynamo for distributed inference,
    worker management, and resource coordination.
    """

    __version__ = "1.0.0"
    __author__ = "xiaoyaolfc"
    __dependencies__ = ["etcd3", "nats-py"]

    def __init__(self, core_instance=None, config: Dict[str, Any] = None):
        super().__init__(core_instance, config)

        # Dynamo configuration
        self.namespace = config.get("namespace", "dynamo")
        self.component_name = config.get("component_name", "xconnector")

        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_affinity: Dict[str, str] = {}  # request_prefix -> worker_id

        # Routing configuration
        self.routing_policy = RoutingPolicy(**config.get("routing_policy", {}))

        # Service discovery
        self.etcd_client = None
        self.etcd_prefix = f"/{self.namespace}/xconnector"

        # Health monitoring
        self.health_check_task = None
        self.last_health_check_time = datetime.now()

        # Metrics
        self.total_routed_requests = 0
        self.routing_errors = 0
        self.cache_coordination_count = 0

        logger.info(f"DynamoAdapter initialized for namespace: {self.namespace}")

    # === Required BaseInterface Methods ===

    async def _initialize_impl(self) -> bool:
        """Initialize Dynamo adapter"""
        try:
            # Try to get Dynamo context
            try:
                from dynamo.sdk import dynamo_context

                if "runtime" in dynamo_context:
                    runtime = dynamo_context["runtime"]
                    self.etcd_client = runtime.etcd_client()

                    # Initialize configuration cache
                    await self._init_config_cache()

                    # Start service discovery
                    await self._start_service_discovery()

                    logger.info("Connected to Dynamo runtime")
                else:
                    logger.warning("Dynamo runtime not available, running in standalone mode")

            except ImportError:
                logger.warning("Dynamo SDK not available, running in standalone mode")

            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor_loop())

            return True

        except Exception as e:
            self.log_error(e, {"operation": "initialize"})
            return False

    async def _start_impl(self) -> bool:
        """Start Dynamo adapter services"""
        try:
            # Register with core router if available
            if self.core:
                self.core.router.register_adapter("dynamo", self)

            # Start accepting worker registrations
            if self.etcd_client:
                await self._start_worker_discovery()

            logger.info("DynamoAdapter started successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "start"})
            return False

    async def _stop_impl(self) -> bool:
        """Stop Dynamo adapter services"""
        try:
            # Cancel health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass

            # Cleanup worker registrations
            if self.etcd_client:
                await self._cleanup_registrations()

            # Clear local state
            self.workers.clear()
            self.worker_affinity.clear()

            logger.info("DynamoAdapter stopped successfully")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "stop"})
            return False

    def get_capabilities(self) -> Dict[str, Capability]:
        """Return Dynamo adapter capabilities"""
        return {
            "worker_management": Capability(
                name="worker_management",
                description="Manage and monitor inference workers",
                version="1.0.0",
                supported=True,
                parameters={
                    "max_workers": 1000,
                    "health_monitoring": True,
                    "auto_scaling": False
                }
            ),
            "distributed_routing": Capability(
                name="distributed_routing",
                description="Route requests across distributed workers",
                version="1.0.0",
                supported=True,
                parameters={
                    "strategies": ["least_loaded", "round_robin", "affinity"],
                    "circuit_breaker": True
                }
            ),
            "cache_coordination": Capability(
                name="cache_coordination",
                description="Coordinate cache operations across workers",
                version="1.0.0",
                supported=True,
                parameters={
                    "cache_affinity": True,
                    "cache_migration": False
                }
            ),
            "service_discovery": Capability(
                name="service_discovery",
                description="Automatic service discovery via etcd",
                version="1.0.0",
                supported=self.etcd_client is not None,
                parameters={
                    "backend": "etcd",
                    "auto_register": True
                }
            )
        }

    async def _health_check_impl(self) -> Optional[HealthCheckResult]:
        """Dynamo specific health check"""
        try:
            # Check etcd connectivity
            etcd_healthy = await self._check_etcd_health() if self.etcd_client else True

            # Check worker health
            healthy_workers = sum(1 for w in self.workers.values()
                                  if w.status in [WorkerStatus.READY, WorkerStatus.BUSY])
            total_workers = len(self.workers)

            # Determine overall health
            if not etcd_healthy:
                status = HealthStatus.DEGRADED
                message = "etcd connection issues"
            elif total_workers == 0:
                status = HealthStatus.DEGRADED
                message = "No workers registered"
            elif healthy_workers < total_workers * 0.5:
                status = HealthStatus.UNHEALTHY
                message = f"Only {healthy_workers}/{total_workers} workers healthy"
            else:
                status = HealthStatus.HEALTHY
                message = "All systems operational"

            return HealthCheckResult(
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    "total_workers": total_workers,
                    "healthy_workers": healthy_workers,
                    "etcd_connected": etcd_healthy,
                    "routed_requests": self.total_routed_requests,
                    "routing_errors": self.routing_errors,
                    "uptime_minutes": (datetime.now() - self.last_health_check_time).seconds / 60
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.now()
            )

    # === Worker Management ===

    async def register_worker(self, worker_id: str, worker_info: Dict[str, Any]) -> bool:
        """
        Register a new worker

        Args:
            worker_id: Unique worker identifier
            worker_info: Worker information including model, resources, endpoint

        Returns:
            bool: Registration success
        """
        try:
            # Create worker info
            worker = WorkerInfo(
                worker_id=worker_id,
                model_name=worker_info.get("model", "unknown"),
                gpu_memory=worker_info.get("gpu_memory", 0),
                status=WorkerStatus.READY,
                endpoint=worker_info.get("endpoint", ""),
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                metadata=worker_info.get("metadata", {})
            )

            # Store locally
            self.workers[worker_id] = worker

            # Register in etcd if available
            if self.etcd_client:
                await self._register_worker_etcd(worker_id, worker)

            # Emit event
            if self.core:
                await self.emit_event("worker_registered",
                                      worker_id=worker_id,
                                      worker_info=worker_info)

            logger.info(f"Registered worker: {worker_id} with model {worker.model_name}")
            return True

        except Exception as e:
            self.log_error(e, {"operation": "register_worker", "worker_id": worker_id})
            return False

    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker"""
        try:
            if worker_id in self.workers:
                del self.workers[worker_id]

                # Remove from etcd
                if self.etcd_client:
                    await self._unregister_worker_etcd(worker_id)

                # Clean up affinity mappings
                self.worker_affinity = {k: v for k, v in self.worker_affinity.items()
                                        if v != worker_id}

                logger.info(f"Unregistered worker: {worker_id}")
                return True

            return False

        except Exception as e:
            self.log_error(e, {"operation": "unregister_worker", "worker_id": worker_id})
            return False

    async def update_worker_status(self, worker_id: str, status: WorkerStatus,
                                   metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Update worker status and metrics"""
        try:
            if worker_id not in self.workers:
                return False

            worker = self.workers[worker_id]
            worker.status = status
            worker.last_heartbeat = datetime.now()

            if metrics:
                worker.active_requests = metrics.get("active_requests", 0)
                worker.total_requests = metrics.get("total_requests", 0)
                worker.error_count = metrics.get("error_count", 0)

            # Update in etcd
            if self.etcd_client:
                await self._update_worker_etcd(worker_id, worker)

            return True

        except Exception as e:
            self.log_error(e, {"operation": "update_worker_status", "worker_id": worker_id})
            return False

    # === Request Routing ===

    async def route_request(self, request: Dict[str, Any]) -> Optional[str]:
        """
        Route a request to the best available worker

        Args:
            request: Request information including model, tokens, etc.

        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker
        """
        try:
            self.total_routed_requests += 1

            # Get eligible workers
            model_name = request.get("model", "")
            eligible_workers = self._get_eligible_workers(model_name)

            if not eligible_workers:
                logger.warning(f"No eligible workers for model: {model_name}")
                self.routing_errors += 1
                return None

            # Apply routing strategy
            selected_worker = await self._apply_routing_strategy(request, eligible_workers)

            if selected_worker:
                # Update worker metrics
                self.workers[selected_worker].active_requests += 1

                # Update affinity if needed
                if self.routing_policy.strategy == "affinity":
                    affinity_key = request.get(self.routing_policy.affinity_key, "")
                    if affinity_key:
                        self.worker_affinity[affinity_key] = selected_worker

                logger.debug(f"Routed request to worker: {selected_worker}")
                return selected_worker

            self.routing_errors += 1
            return None

        except Exception as e:
            self.log_error(e, {"operation": "route_request"})
            self.routing_errors += 1
            return None

    async def coordinate_cache_operation(self, operation: str, cache_key: str,
                                         worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Coordinate cache operations across workers

        Args:
            operation: Cache operation (get, put, delete)
            cache_key: Cache key
            worker_id: Specific worker to target (optional)

        Returns:
            Dict containing operation result
        """
        try:
            self.cache_coordination_count += 1

            # Determine target worker(s)
            if worker_id:
                target_workers = [worker_id] if worker_id in self.workers else []
            else:
                # Find workers that might have this cache
                target_workers = await self._find_cache_workers(cache_key)

            if not target_workers:
                return {"status": "no_workers", "result": None}

            # Execute cache operation
            results = []
            for worker in target_workers:
                result = await self._execute_cache_op_on_worker(
                    worker, operation, cache_key
                )
                if result.get("found"):
                    return {"status": "success", "result": result, "worker": worker}
                results.append(result)

            return {"status": "not_found", "results": results}

        except Exception as e:
            self.log_error(e, {"operation": "coordinate_cache_operation"})
            return {"status": "error", "error": str(e)}

    # === Helper Methods ===

    def _get_eligible_workers(self, model_name: str) -> List[str]:
        """Get workers eligible for a model"""
        eligible = []

        for worker_id, worker in self.workers.items():
            # Check model compatibility
            if model_name and worker.model_name != model_name:
                continue

            # Check health status
            if worker.status not in [WorkerStatus.READY, WorkerStatus.BUSY]:
                continue

            # Check load
            if worker.active_requests >= self.routing_policy.max_requests_per_worker:
                continue

            eligible.append(worker_id)

        return eligible

    async def _apply_routing_strategy(self, request: Dict[str, Any],
                                      workers: List[str]) -> Optional[str]:
        """Apply routing strategy to select worker"""
        strategy = self.routing_policy.strategy

        if strategy == "least_loaded":
            # Sort by active requests
            workers_by_load = sorted(
                workers,
                key=lambda w: self.workers[w].active_requests
            )
            return workers_by_load[0] if workers_by_load else None

        elif strategy == "round_robin":
            # Simple round-robin
            if not hasattr(self, '_rr_index'):
                self._rr_index = 0

            if workers:
                selected = workers[self._rr_index % len(workers)]
                self._rr_index += 1
                return selected

        elif strategy == "affinity":
            # Check affinity mapping
            affinity_key = request.get(self.routing_policy.affinity_key, "")
            if affinity_key in self.worker_affinity:
                preferred = self.worker_affinity[affinity_key]
                if preferred in workers:
                    return preferred

            # Fall back to least loaded
            return await self._apply_routing_strategy(
                request, workers
            )

        return None

    async def _health_monitor_loop(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.routing_policy.health_check_interval)

                # Check worker health
                now = datetime.now()
                for worker_id, worker in list(self.workers.items()):
                    # Check heartbeat timeout
                    time_since_heartbeat = (now - worker.last_heartbeat).seconds

                    if time_since_heartbeat > self.routing_policy.health_check_interval * 2:
                        # Mark as unhealthy
                        worker.status = WorkerStatus.UNHEALTHY
                        worker.error_count += 1

                        # Remove if threshold exceeded
                        if worker.error_count > self.routing_policy.unhealthy_threshold:
                            await self.unregister_worker(worker_id)
                            logger.warning(f"Removed unhealthy worker: {worker_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    # === etcd Integration ===

    async def _init_config_cache(self):
        """Initialize configuration cache in etcd"""
        try:
            from dynamo.runtime import EtcdKvCache

            self.config_cache = await EtcdKvCache.create(
                self.etcd_client,
                f"{self.etcd_prefix}/config/",
                {
                    "routing_policy": self.routing_policy.__dict__,
                    "namespace": self.namespace,
                    "component": self.component_name
                }
            )

        except Exception as e:
            logger.error(f"Failed to init config cache: {e}")

    async def _register_worker_etcd(self, worker_id: str, worker: WorkerInfo):
        """Register worker in etcd"""
        try:
            key = f"{self.etcd_prefix}/workers/{worker_id}"
            value = {
                "model_name": worker.model_name,
                "endpoint": worker.endpoint,
                "status": worker.status.value,
                "registered_at": worker.registered_at.isoformat(),
                "metadata": worker.metadata
            }

            await self.etcd_client.put(key, json.dumps(value))

        except Exception as e:
            logger.error(f"Failed to register worker in etcd: {e}")

    async def _check_etcd_health(self) -> bool:
        """Check etcd connectivity"""
        try:
            # Simple health check - try to read a key
            await self.etcd_client.get(f"{self.etcd_prefix}/health")
            return True
        except:
            return False

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """Get Dynamo adapter specific metrics"""
        return {
            "total_workers": len(self.workers),
            "healthy_workers": sum(1 for w in self.workers.values()
                                   if w.status == WorkerStatus.READY),
            "total_routed_requests": self.total_routed_requests,
            "routing_errors": self.routing_errors,
            "cache_coordinations": self.cache_coordination_count,
            "routing_strategy": self.routing_policy.strategy,
            "namespace": self.namespace
        }

    # === Public API Methods ===

    def list_workers(self) -> List[Dict[str, Any]]:
        """List all registered workers"""
        return [
            {
                "worker_id": w.worker_id,
                "model": w.model_name,
                "status": w.status.value,
                "endpoint": w.endpoint,
                "active_requests": w.active_requests,
                "total_requests": w.total_requests,
                "last_heartbeat": w.last_heartbeat.isoformat()
            }
            for w in self.workers.values()
        ]

    def get_worker_info(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get specific worker information"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            return {
                "worker_id": worker.worker_id,
                "model": worker.model_name,
                "status": worker.status.value,
                "endpoint": worker.endpoint,
                "gpu_memory": worker.gpu_memory,
                "active_requests": worker.active_requests,
                "total_requests": worker.total_requests,
                "error_count": worker.error_count,
                "registered_at": worker.registered_at.isoformat(),
                "last_heartbeat": worker.last_heartbeat.isoformat(),
                "metadata": worker.metadata
            }
        return None

    def update_routing_policy(self, policy: Dict[str, Any]) -> bool:
        """Update routing policy"""
        try:
            self.routing_policy = RoutingPolicy(**policy)
            logger.info(f"Updated routing policy: {policy}")
            return True
        except Exception as e:
            logger.error(f"Failed to update routing policy: {e}")
            return False