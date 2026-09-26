"""FeatureBench Inference Module"""


from featurebench.infer.models import (
    AgentName,
    TaskInstance,
    InferConfig,
    InferResult,
    RunMetadata,
    TaskPaths
)
from featurebench.infer.config import InferConfigLoader, DatasetLoader
from featurebench.infer.container import ContainerManager
from featurebench.infer.runtime import RuntimeHandler
from featurebench.infer.output import OutputManager
from featurebench.infer.agents import (
    get_agent,
    BaseAgent,
    ClaudeCodeAgent,
    GeminiCliAgent,
    MiniSweAgent,
    OpenHandsAgent,
    CodexAgent,
)


def __getattr__(name):
    """Lazy import for run_infer to avoid RuntimeWarning when running as module."""
    if name in ("main", "InferenceRunner"):
        from featurebench.infer import run_infer
        return getattr(run_infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "main",
    "InferenceRunner",
    "AgentName",
    "TaskInstance",
    "InferConfig",
    "InferResult",
    "RunMetadata",
    "TaskPaths",
    "InferConfigLoader",
    "DatasetLoader",
    "ContainerManager",
    "RuntimeHandler",
    "OutputManager",
    "get_agent",
    "BaseAgent",
    "ClaudeCodeAgent",
    "GeminiCliAgent",
    "MiniSweAgent",
    "OpenHandsAgent",
    "CodexAgent",
]
