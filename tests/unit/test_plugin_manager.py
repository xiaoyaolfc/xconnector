import pytest
import tempfile
import shutil
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Type, Any, Union

# Import the class to test
from xconnector.core.plugin_manager import PluginManager, BaseInterface, PluginInfo, PluginStatus


# Mock classes for testing
class MockBaseInterface(BaseInterface):
    pass


class MockAdapterConfig:
    def __init__(self, name: str, class_path: str, config: Dict = None):
        self.name = name
        self.class_path = class_path
        self.config = config or {}


# Test cases
class TestPluginManager:
    @pytest.fixture
    def manager(self):
        """Fixture providing a clean PluginManager instance for each test"""
        return PluginManager()

    @pytest.fixture
    def temp_plugin_dir(self):
        """Fixture creating a temporary directory with test plugins"""
        temp_dir = tempfile.mkdtemp()
        # Create directory structure
        (Path(temp_dir) / "inference_adapters").mkdir()
        (Path(temp_dir) / "cache_adapters").mkdir()
        (Path(temp_dir) / "distributed_adapters").mkdir()

        # Create test files
        with open(Path(temp_dir) / "inference_adapters" / "test_inference.py", "w") as f:
            f.write("""
from plugin_manager import BaseInterface

class TestInferenceAdapter(BaseInterface):
    pass
""")

        with open(Path(temp_dir) / "cache_adapters" / "test_cache.py", "w") as f:
            f.write("""
from plugin_manager import BaseInterface

class TestCacheAdapter(BaseInterface):
    pass
""")

        with open(Path(temp_dir) / "distributed_adapters" / "test_distributed.py", "w") as f:
            f.write("""
from plugin_manager import BaseInterface

class TestDistributedAdapter(BaseInterface):
    pass
""")

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, manager):
        """Test PluginManager initialization"""
        assert len(manager.plugin_paths) == 3
        assert isinstance(manager._lock, type(manager._lock))
        assert manager.inference_adapters == {}
        assert manager.cache_adapters == {}
        assert manager.distributed_adapters == {}
        assert manager.adapter_configs == {}
        assert manager.plugin_info == {}
        assert manager._loaded_modules == {}

    def test_add_plugin_path(self, manager):
        """Test adding plugin paths"""
        new_path = "/test/path"
        manager.add_plugin_path(new_path)
        assert Path(new_path) in manager.plugin_paths

        # Test adding duplicate path
        original_length = len(manager.plugin_paths)
        manager.add_plugin_path(new_path)
        assert len(manager.plugin_paths) == original_length

    def test_register_adapter(self, manager):
        """Test registering adapter configs"""
        config = MockAdapterConfig("test", "test.path")
        manager.register_adapter(config)
        assert "test" in manager.adapter_configs
        assert manager.adapter_configs["test"] == config

    def test_register_inference_adapter(self, manager):
        """Test registering inference adapters"""
        manager.register_inference_adapter("test", MockBaseInterface)
        assert "test" in manager.inference_adapters
        assert manager.inference_adapters["test"] == MockBaseInterface

        # Test invalid adapter
        with pytest.raises(TypeError):
            manager.register_inference_adapter("invalid", object)

    def test_register_cache_adapter(self, manager):
        """Test registering cache adapters"""
        manager.register_cache_adapter("test", MockBaseInterface)
        assert "test" in manager.cache_adapters
        assert manager.cache_adapters["test"] == MockBaseInterface

        # Test invalid adapter
        with pytest.raises(TypeError):
            manager.register_cache_adapter("invalid", object)

    def test_register_distributed_adapter(self, manager):
        """Test registering distributed adapters"""
        manager.register_distributed_adapter("test", MockBaseInterface)
        assert "test" in manager.distributed_adapters
        assert manager.distributed_adapters["test"] == MockBaseInterface

        # Test invalid adapter
        with pytest.raises(TypeError):
            manager.register_distributed_adapter("invalid", object)

    def test_discover_adapters(self, manager, temp_plugin_dir):
        """Test adapter discovery"""
        manager.add_plugin_path(temp_plugin_dir)
        discovered = manager.discover_adapters()

        assert isinstance(discovered, dict)
        assert "inference" in discovered
        assert "cache" in discovered
        assert "distributed" in discovered

        # Should find our test adapters
        assert any("test_inference" in s for s in discovered["inference"])
        assert any("test_cache" in s for s in discovered["cache"])
        assert any("test_distributed" in s for s in discovered["distributed"])

    def test_infer_adapter_type(self, manager):
        """Test adapter type inference"""
        # Test path-based inference
        assert manager._infer_adapter_type(Path("/path/to/inference/adapter.py"), "Test") == "inference"
        assert manager._infer_adapter_type(Path("/path/to/cache/adapter.py"), "Test") == "cache"
        assert manager._infer_adapter_type(Path("/path/to/distributed/adapter.py"), "Test") == "distributed"

        # Test class name-based inference
        assert manager._infer_adapter_type(Path("/path/to/unknown.py"), "VLLMAdapter") == "inference"
        assert manager._infer_adapter_type(Path("/path/to/unknown.py"), "RedisCache") == "cache"
        assert manager._infer_adapter_type(Path("/path/to/unknown.py"), "ShardedCluster") == "distributed"

        # Test unknown type
        assert manager._infer_adapter_type(Path("/path/to/unknown.py"), "Unknown") is None

    def test_load_adapter(self, manager):
        """Test loading adapters"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)

        # Test successful load
        adapter = manager.load_adapter(config)
        assert isinstance(adapter, BaseInterface)
        assert "test" in manager._loaded_modules
        assert "test" in manager.plugin_info

        # Test already loaded
        with pytest.raises(RuntimeError):
            manager.load_adapter(config)

        # Test invalid adapter
        invalid_config = MockAdapterConfig("invalid", "nonexistent.module.Class")
        with pytest.raises(RuntimeError):
            manager.load_adapter(invalid_config)

    def test_load_class_from_path(self, manager):
        """Test loading class from path"""
        # Test valid class
        cls = manager._load_class_from_path("plugin_manager.MockBaseInterface")
        assert cls == MockBaseInterface

        # Test invalid module
        with pytest.raises(ImportError):
            manager._load_class_from_path("nonexistent.module.Class")

        # Test invalid class
        with pytest.raises(ImportError):
            manager._load_class_from_path("plugin_manager.NonexistentClass")

    def test_unload_adapter(self, manager):
        """Test unloading adapters"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test successful unload
        manager.unload_adapter("test")
        assert "test" not in manager._loaded_modules
        assert manager.plugin_info["test"].status == PluginStatus.UNLOADED

        # Test unload non-existent adapter
        with pytest.raises(Exception):
            manager.unload_adapter("nonexistent")

    def test_get_adapter_info(self, manager):
        """Test getting adapter info"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test get info
        info = manager.get_adapter_info("test")
        assert isinstance(info, PluginInfo)
        assert info.name == "test"

        # Test non-existent adapter
        assert manager.get_adapter_info("nonexistent") is None

    def test_list_loaded_adapters(self, manager):
        """Test listing loaded adapters"""
        assert manager.list_loaded_adapters() == []

        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        assert manager.list_loaded_adapters() == ["test"]

    def test_list_registered_adapters(self, manager):
        """Test listing registered adapters"""
        # Initial state
        registered = manager.list_registered_adapters()
        assert registered["inference"] == []
        assert registered["cache"] == []
        assert registered["distributed"] == []

        # After registration
        manager.register_inference_adapter("inf", MockBaseInterface)
        manager.register_cache_adapter("cache", MockBaseInterface)
        manager.register_distributed_adapter("dist", MockBaseInterface)

        registered = manager.list_registered_adapters()
        assert "inf" in registered["inference"]
        assert "cache" in registered["cache"]
        assert "dist" in registered["distributed"]

    def test_get_adapter_status(self, manager):
        """Test getting adapter status"""
        # Initial state
        status = manager.get_adapter_status()
        assert status == {}

        # After loading
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        status = manager.get_adapter_status()
        assert "test" in status
        assert status["test"]["status"] == PluginStatus.LOADED.value

    def test_reload_adapter(self, manager):
        """Test reloading adapters"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test reload
        reloaded = manager.reload_adapter("test")
        assert isinstance(reloaded, BaseInterface)

        # Test reload non-existent
        with pytest.raises(ValueError):
            manager.reload_adapter("nonexistent")

    def test_validate_dependencies(self, manager):
        """Test dependency validation"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test with no dependencies
        assert manager.validate_dependencies("test") is True

        # Test with mock dependencies
        with patch.dict(manager.plugin_info["test"].__dict__, {"dependencies": ["os"]}):
            assert manager.validate_dependencies("test") is True

        with patch.dict(manager.plugin_info["test"].__dict__, {"dependencies": ["nonexistent_module"]}):
            assert manager.validate_dependencies("test") is False

    def test_cleanup(self, manager):
        """Test cleanup"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test cleanup
        manager.cleanup()
        assert manager._loaded_modules == {}
        assert manager.plugin_info == {}

    def test_destructor(self, manager):
        """Test destructor behavior"""
        # Setup
        config = MockAdapterConfig("test", "plugin_manager.MockBaseInterface")
        manager.register_adapter(config)
        adapter = manager.load_adapter(config)

        # Test destructor
        manager.__del__()
        assert manager._loaded_modules == {}
        assert manager.plugin_info == {}