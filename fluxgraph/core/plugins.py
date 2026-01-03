"""
FluxGraph Plugin System

Enables extensibility through plugins:
- Custom LLM providers
- Custom memory backends
- Custom tools
- Lifecycle hooks
- Middleware
"""

import importlib
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, Type
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str] = None

    def __post_init__(self):
        self.dependencies = self.dependencies or []


class Plugin(ABC):
    """
    Base class for FluxGraph plugins

    Example:
        class MyCustomPlugin(Plugin):
            def initialize(self, app):
                print("Plugin initialized!")
                app.register("my_agent", MyAgent())

            def get_metadata(self):
                return PluginMetadata(
                    name="my-plugin",
                    version="1.0.0",
                    author="You",
                    description="Custom plugin"
                )
    """

    @abstractmethod
    def initialize(self, app: Any):
        """
        Initialize plugin with FluxApp instance

        Args:
            app: FluxApp instance
        """
        pass

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    def on_agent_start(self, agent_name: str, **kwargs):
        """Hook: Called before agent execution"""
        pass

    def on_agent_complete(self, agent_name: str, result: Any, **kwargs):
        """Hook: Called after successful agent execution"""
        pass

    def on_agent_error(self, agent_name: str, error: Exception, **kwargs):
        """Hook: Called when agent fails"""
        pass

    def shutdown(self):
        """Cleanup when plugin is unloaded"""
        pass


class PluginManager:
    """
    Manages plugin lifecycle

    Example:
        plugin_mgr = PluginManager()

        # Load plugin from class
        plugin_mgr.load_plugin(MyCustomPlugin())

        # Load from file
        plugin_mgr.load_from_file("path/to/plugin.py")

        # Load all from directory
        plugin_mgr.load_from_directory("plugins/")

        # Initialize all plugins
        plugin_mgr.initialize_all(app)
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "agent_start": [],
            "agent_complete": [],
            "agent_error": []
        }

    def load_plugin(self, plugin: Plugin):
        """
        Load a plugin instance

        Args:
            plugin: Plugin instance

        Raises:
            ValueError: If plugin with same name already loaded
        """
        metadata = plugin.get_metadata()

        if metadata.name in self.plugins:
            raise ValueError(f"Plugin '{metadata.name}' already loaded")

        # Check dependencies
        missing_deps = []
        for dep in metadata.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            logger.warning(
                f"⚠️ Plugin '{metadata.name}' missing dependencies: {missing_deps}"
            )

        self.plugins[metadata.name] = plugin

        # Register hooks
        if hasattr(plugin, "on_agent_start"):
            self._hooks["agent_start"].append(plugin.on_agent_start)
        if hasattr(plugin, "on_agent_complete"):
            self._hooks["agent_complete"].append(plugin.on_agent_complete)
        if hasattr(plugin, "on_agent_error"):
            self._hooks["agent_error"].append(plugin.on_agent_error)

        logger.info(
            f"📦 Loaded plugin: {metadata.name} v{metadata.version}"
        )

    def load_from_file(self, filepath: str):
        """
        Load plugin from Python file

        Args:
            filepath: Path to plugin file

        The file should contain a class that inherits from Plugin
        and is named with 'Plugin' suffix (e.g., MyCustomPlugin)
        """
        spec = importlib.util.spec_from_file_location("plugin_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Plugin classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, Plugin) and
                obj is not Plugin and
                name.endswith("Plugin")):

                plugin_instance = obj()
                self.load_plugin(plugin_instance)
                return

        raise ValueError(f"No Plugin class found in {filepath}")

    def load_from_directory(self, directory: str, pattern: str = "*.py"):
        """
        Load all plugins from directory

        Args:
            directory: Directory containing plugin files
            pattern: File pattern to match (default: *.py)
        """
        plugin_dir = Path(directory)

        if not plugin_dir.exists():
            logger.warning(f"⚠️ Plugin directory not found: {directory}")
            return

        for filepath in plugin_dir.glob(pattern):
            if filepath.name.startswith("_"):
                continue

            try:
                self.load_from_file(str(filepath))
            except Exception as e:
                logger.error(f"❌ Failed to load plugin {filepath}: {e}")

    def initialize_all(self, app: Any):
        """
        Initialize all loaded plugins

        Args:
            app: FluxApp instance
        """
        for name, plugin in self.plugins.items():
            try:
                plugin.initialize(app)
                logger.info(f"✅ Initialized plugin: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize plugin {name}: {e}")

    def unload_plugin(self, name: str):
        """
        Unload a plugin

        Args:
            name: Plugin name
        """
        if name in self.plugins:
            plugin = self.plugins[name]
            plugin.shutdown()
            del self.plugins[name]
            logger.info(f"🗑️ Unloaded plugin: {name}")

    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """
        Trigger a plugin hook

        Args:
            hook_name: Hook name (agent_start, agent_complete, agent_error)
            *args, **kwargs: Arguments to pass to hooks
        """
        if hook_name in self._hooks:
            for hook in self._hooks[hook_name]:
                try:
                    hook(*args, **kwargs)
                except Exception as e:
                    logger.error(f"❌ Hook error ({hook_name}): {e}")

    def list_plugins(self) -> List[Dict]:
        """
        List all loaded plugins with metadata

        Returns:
            List of plugin info dictionaries
        """
        return [
            {
                "name": plugin.get_metadata().name,
                "version": plugin.get_metadata().version,
                "author": plugin.get_metadata().author,
                "description": plugin.get_metadata().description
            }
            for plugin in self.plugins.values()
        ]

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name"""
        return self.plugins.get(name)


# Example plugins for reference


class LoggingPlugin(Plugin):
    """Example: Log all agent executions"""

    def get_metadata(self):
        return PluginMetadata(
            name="logging",
            version="1.0.0",
            author="FluxGraph",
            description="Logs all agent executions"
        )

    def initialize(self, app):
        logger.info("🔍 Logging plugin initialized")

    def on_agent_start(self, agent_name: str, **kwargs):
        logger.info(f"▶️ Agent started: {agent_name}")

    def on_agent_complete(self, agent_name: str, result: Any, **kwargs):
        logger.info(f"✅ Agent completed: {agent_name}")

    def on_agent_error(self, agent_name: str, error: Exception, **kwargs):
        logger.error(f"❌ Agent failed: {agent_name} - {error}")


class MetricsPlugin(Plugin):
    """Example: Collect agent execution metrics"""

    def __init__(self):
        self.executions = 0
        self.failures = 0

    def get_metadata(self):
        return PluginMetadata(
            name="metrics",
            version="1.0.0",
            author="FluxGraph",
            description="Collects execution metrics"
        )

    def initialize(self, app):
        logger.info("📊 Metrics plugin initialized")

    def on_agent_complete(self, agent_name: str, result: Any, **kwargs):
        self.executions += 1

    def on_agent_error(self, agent_name: str, error: Exception, **kwargs):
        self.failures += 1

    def get_stats(self) -> Dict:
        return {
            "total_executions": self.executions,
            "total_failures": self.failures,
            "success_rate": (
                (self.executions - self.failures) / self.executions * 100
                if self.executions > 0 else 0
            )
        }


class CustomProviderPlugin(Plugin):
    """Example: Add custom LLM provider"""

    def get_metadata(self):
        return PluginMetadata(
            name="custom-provider",
            version="1.0.0",
            author="You",
            description="Adds custom LLM provider",
            dependencies=["custom_llm_sdk"]
        )

    def initialize(self, app):
        from fluxgraph.models.provider import LLMProvider

        class CustomProvider(LLMProvider):
            async def generate(self, prompt, **kwargs):
                # Your custom implementation
                return {"text": "Custom response"}

        # Register with app (if app supports provider registration)
        logger.info("🔌 Custom provider registered")


# Plugin discovery helpers


def discover_plugins(directory: str = "plugins") -> PluginManager:
    """
    Discover and load plugins from directory

    Args:
        directory: Plugin directory path

    Returns:
        PluginManager with loaded plugins

    Example:
        plugin_mgr = discover_plugins("./my_plugins")
        plugin_mgr.initialize_all(app)
    """
    manager = PluginManager()
    manager.load_from_directory(directory)
    return manager


def load_plugin_from_spec(spec: str):
    """
    Load plugin from module spec

    Args:
        spec: Module spec (e.g., "my_package.plugins.MyPlugin")

    Example:
        plugin = load_plugin_from_spec("company.fluxgraph_plugins.CustomPlugin")
    """
    parts = spec.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid plugin spec: {spec}")

    module_name, class_name = parts
    module = importlib.import_module(module_name)
    plugin_class = getattr(module, class_name)

    if not issubclass(plugin_class, Plugin):
        raise ValueError(f"{class_name} is not a Plugin subclass")

    return plugin_class()
