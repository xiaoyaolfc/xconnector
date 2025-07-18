# xconnector/core/plugin_manager.py
import os
import sys
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, List, Any, Optional, Type, Union
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from xconnector.interfaces.base_interface import BaseInterface
from xconnector.utils.logging import get_logger

logger = get_logger(__name__)


class PluginStatus(Enum):
    """插件状态"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    class_path: str
    dependencies: List[str]
    status: PluginStatus = PluginStatus.UNLOADED
    error_message: Optional[str] = None


class PluginManager:
    """
    XConnector 插件管理器

    负责插件的发现、加载、卸载和生命周期管理
    """

    def __init__(self):
        self.inference_adapters: Dict[str, BaseInterface] = {}
        self.cache_adapters: Dict[str, BaseInterface] = {}
        self.distributed_adapters: Dict[str, BaseInterface] = {}

        # 适配器配置存储
        self.adapter_configs: Dict[str, Any] = {}

        # 插件信息存储
        self.plugin_info: Dict[str, PluginInfo] = {}

        # 线程安全锁
        self._lock = threading.RLock()

        # 插件搜索路径
        self.plugin_paths: List[Path] = [
            Path("xconnector/adapters"),
            Path("plugins"),
            Path.cwd() / "plugins"
        ]

        # 已加载的模块缓存
        self._loaded_modules: Dict[str, Any] = {}

        logger.info("PluginManager initialized")

    def add_plugin_path(self, path: Union[str, Path]):
        """
        添加插件搜索路径

        Args:
            path: 插件路径
        """
        path = Path(path)
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")

    def register_adapter(self, adapter_config) -> None:
        """
        注册适配器配置

        Args:
            adapter_config: 适配器配置对象
        """
        with self._lock:
            self.adapter_configs[adapter_config.name] = adapter_config
            logger.info(f"Registered adapter config: {adapter_config.name}")

    def register_inference_adapter(self, name: str, adapter_class: Type[BaseInterface]):
        """
        注册推理引擎适配器类

        Args:
            name: 适配器名称
            adapter_class: 适配器类
        """
        with self._lock:
            if not issubclass(adapter_class, BaseInterface):
                raise TypeError(f"Adapter {name} must inherit from BaseInterface")

            # 存储类引用而不是实例
            self.inference_adapters[name] = adapter_class
            logger.info(f"Registered inference adapter: {name}")

    def register_cache_adapter(self, name: str, adapter_class: Type[BaseInterface]):
        """
        注册缓存管理适配器类

        Args:
            name: 适配器名称
            adapter_class: 适配器类
        """
        with self._lock:
            if not issubclass(adapter_class, BaseInterface):
                raise TypeError(f"Adapter {name} must inherit from BaseInterface")

            self.cache_adapters[name] = adapter_class
            logger.info(f"Registered cache adapter: {name}")

    def register_distributed_adapter(self, name: str, adapter_class: Type[BaseInterface]):
        """
        注册分布式适配器类

        Args:
            name: 适配器名称
            adapter_class: 适配器类
        """
        with self._lock:
            if not issubclass(adapter_class, BaseInterface):
                raise TypeError(f"Adapter {name} must inherit from BaseInterface")

            self.distributed_adapters[name] = adapter_class
            logger.info(f"Registered distributed adapter: {name}")

    def discover_adapters(self) -> Dict[str, List[str]]:
        """
        自动发现可用的适配器

        Returns:
            Dict[str, List[str]]: 按类型分组的适配器列表
        """
        discovered = {
            "inference": [],
            "cache": [],
            "distributed": []
        }

        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                continue

            logger.info(f"Discovering adapters in: {plugin_path}")

            # 搜索 Python 文件
            for py_file in plugin_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    # 构建模块名
                    relative_path = py_file.relative_to(plugin_path)
                    module_name = str(relative_path.with_suffix("")).replace(os.sep, ".")

                    # 加载模块
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # 检查模块中的类
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, BaseInterface) and
                                    obj != BaseInterface and
                                    not inspect.isabstract(obj)):

                                # 根据类名或路径推断适配器类型
                                adapter_type = self._infer_adapter_type(py_file, name)
                                if adapter_type:
                                    discovered[adapter_type].append(f"{module_name}.{name}")

                except Exception as e:
                    logger.warning(f"Failed to discover adapters in {py_file}: {e}")

        logger.info(f"Discovered adapters: {discovered}")
        return discovered

    def _infer_adapter_type(self, file_path: Path, class_name: str) -> Optional[str]:
        """
        根据文件路径和类名推断适配器类型

        Args:
            file_path: 文件路径
            class_name: 类名

        Returns:
            Optional[str]: 适配器类型
        """
        path_str = str(file_path).lower()
        class_name_lower = class_name.lower()

        # 基于路径的推断
        if "inference" in path_str:
            return "inference"
        elif "cache" in path_str:
            return "cache"
        elif "distributed" in path_str:
            return "distributed"

        # 基于类名的推断
        if any(keyword in class_name_lower for keyword in ["vllm", "tgi", "inference", "llm"]):
            return "inference"
        elif any(keyword in class_name_lower for keyword in ["cache", "redis", "memcache"]):
            return "cache"
        elif any(keyword in class_name_lower for keyword in ["distributed", "cluster", "shard"]):
            return "distributed"

        return None

    def load_adapter(self, adapter_config, core_instance=None) -> BaseInterface:
        """
        动态加载适配器

        Args:
            adapter_config: 适配器配置
            core_instance: 核心实例

        Returns:
            BaseInterface: 加载的适配器实例
        """
        with self._lock:
            adapter_name = adapter_config.name

            try:
                logger.info(f"Loading adapter: {adapter_name}")

                # 检查是否已经加载
                if adapter_name in self._loaded_modules:
                    logger.warning(f"Adapter {adapter_name} is already loaded")

                # 根据类路径加载类
                adapter_class = self._load_class_from_path(adapter_config.class_path)

                # 创建适配器实例
                if core_instance:
                    adapter_instance = adapter_class(core_instance, adapter_config.config)
                else:
                    adapter_instance = adapter_class(adapter_config.config)

                # 验证适配器接口
                if not isinstance(adapter_instance, BaseInterface):
                    raise TypeError(f"Adapter {adapter_name} must implement BaseInterface")

                # 缓存实例
                self._loaded_modules[adapter_name] = adapter_instance

                # 更新插件信息
                self.plugin_info[adapter_name] = PluginInfo(
                    name=adapter_name,
                    version=getattr(adapter_instance, "__version__", "unknown"),
                    description=getattr(adapter_instance, "__doc__", ""),
                    author=getattr(adapter_instance, "__author__", "unknown"),
                    class_path=adapter_config.class_path,
                    dependencies=getattr(adapter_instance, "__dependencies__", []),
                    status=PluginStatus.LOADED
                )

                logger.info(f"Successfully loaded adapter: {adapter_name}")
                return adapter_instance

            except Exception as e:
                error_msg = f"Failed to load adapter {adapter_name}: {e}"
                logger.error(error_msg)

                # 更新插件信息
                self.plugin_info[adapter_name] = PluginInfo(
                    name=adapter_name,
                    version="unknown",
                    description="",
                    author="unknown",
                    class_path=adapter_config.class_path,
                    dependencies=[],
                    status=PluginStatus.ERROR,
                    error_message=str(e)
                )

                raise RuntimeError(error_msg) from e

    def _load_class_from_path(self, class_path: str) -> Type[BaseInterface]:
        """
        从类路径加载类

        Args:
            class_path: 类路径 (例如: "xconnector.adapters.vllm.VLLMAdapter")

        Returns:
            Type[BaseInterface]: 加载的类
        """
        try:
            # 分离模块路径和类名
            module_path, class_name = class_path.rsplit(".", 1)

            # 导入模块
            module = importlib.import_module(module_path)

            # 获取类
            adapter_class = getattr(module, class_name)

            if not inspect.isclass(adapter_class):
                raise TypeError(f"{class_path} is not a class")

            return adapter_class

        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Cannot import {class_path}: {e}") from e

    def unload_adapter(self, adapter_name: str) -> None:
        """
        卸载适配器

        Args:
            adapter_name: 适配器名称
        """
        with self._lock:
            try:
                if adapter_name in self._loaded_modules:
                    adapter_instance = self._loaded_modules[adapter_name]

                    # 调用清理方法
                    if hasattr(adapter_instance, 'cleanup'):
                        adapter_instance.cleanup()

                    # 从缓存中移除
                    del self._loaded_modules[adapter_name]

                    # 更新插件信息
                    if adapter_name in self.plugin_info:
                        self.plugin_info[adapter_name].status = PluginStatus.UNLOADED

                    logger.info(f"Successfully unloaded adapter: {adapter_name}")
                else:
                    logger.warning(f"Adapter {adapter_name} is not loaded")

            except Exception as e:
                logger.error(f"Failed to unload adapter {adapter_name}: {e}")
                raise

    def get_adapter_info(self, adapter_name: str) -> Optional[PluginInfo]:
        """
        获取适配器信息

        Args:
            adapter_name: 适配器名称

        Returns:
            Optional[PluginInfo]: 适配器信息
        """
        return self.plugin_info.get(adapter_name)

    def list_loaded_adapters(self) -> List[str]:
        """
        列出已加载的适配器

        Returns:
            List[str]: 已加载的适配器名称列表
        """
        with self._lock:
            return list(self._loaded_modules.keys())

    def list_registered_adapters(self) -> Dict[str, List[str]]:
        """
        列出已注册的适配器类

        Returns:
            Dict[str, List[str]]: 按类型分组的适配器列表
        """
        with self._lock:
            return {
                "inference": list(self.inference_adapters.keys()),
                "cache": list(self.cache_adapters.keys()),
                "distributed": list(self.distributed_adapters.keys())
            }

    def get_adapter_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有适配器状态

        Returns:
            Dict[str, Dict[str, Any]]: 适配器状态信息
        """
        status = {}

        with self._lock:
            for name, info in self.plugin_info.items():
                status[name] = {
                    "status": info.status.value,
                    "version": info.version,
                    "description": info.description,
                    "author": info.author,
                    "dependencies": info.dependencies,
                    "error_message": info.error_message
                }

        return status

    def reload_adapter(self, adapter_name: str) -> BaseInterface:
        """
        重新加载适配器

        Args:
            adapter_name: 适配器名称

        Returns:
            BaseInterface: 重新加载的适配器实例
        """
        with self._lock:
            # 先卸载
            self.unload_adapter(adapter_name)

            # 重新加载
            if adapter_name in self.adapter_configs:
                adapter_config = self.adapter_configs[adapter_name]
                return self.load_adapter(adapter_config)
            else:
                raise ValueError(f"No configuration found for adapter: {adapter_name}")

    def validate_dependencies(self, adapter_name: str) -> bool:
        """
        验证适配器依赖

        Args:
            adapter_name: 适配器名称

        Returns:
            bool: 依赖是否满足
        """
        if adapter_name not in self.plugin_info:
            return False

        info = self.plugin_info[adapter_name]

        for dependency in info.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                logger.warning(f"Missing dependency for {adapter_name}: {dependency}")
                return False

        return True

    def cleanup(self):
        """清理插件管理器"""
        with self._lock:
            # 卸载所有适配器
            for adapter_name in list(self._loaded_modules.keys()):
                try:
                    self.unload_adapter(adapter_name)
                except Exception as e:
                    logger.error(f"Error during cleanup of {adapter_name}: {e}")

            # 清理缓存
            self._loaded_modules.clear()
            self.plugin_info.clear()

            logger.info("PluginManager cleaned up")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 忽略析构函数中的错误