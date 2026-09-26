"""Configuration loader for the inference module."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import toml


class InferConfigLoader:
    """Load and manage inference configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config loader."""
        if config_path is None:
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent and config_path is None:
                for candidate_name in ("config.toml", "config_eval.toml"):
                    candidate = current_dir / candidate_name
                    if candidate.exists():
                        config_path = candidate
                        break
                current_dir = current_dir.parent

        if config_path is None or not config_path.exists():
            raise FileNotFoundError(
                "No config found. Create config_eval.toml (or config.toml) in "
                "the project root, or pass --config-path. See "
                "config_eval_example.toml and config_mutation_example.toml."
            )

        self.config_path = config_path
        self._config = toml.load(config_path)

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Return the raw parsed config.toml dictionary."""
        return self._config

    @property
    def env_vars(self) -> Dict[str, str]:
        """Get global environment variables."""
        return self._config.get("env_vars", {})

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get global LLM configuration."""
        return self._config.get("llm_config", {})

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        infer_config = self._config.get("infer_config", {})
        agent_config = infer_config.get(agent_name, {})
        return agent_config

    def get_cache_dir(self) -> Optional[Path]:
        """Get host cache directory for downloads (mounted to /download)."""
        cache_cfg = self._config.get("infer", {}) or {}
        path = cache_cfg.get("download_cache_dir")
        if path:
            return Path(path).expanduser()
        return None

    def get_agent_env_vars(self, agent_name: str) -> Dict[str, str]:
        """Get environment variables for a specific agent."""
        env_vars = dict(self.env_vars)

        agent_config = self.get_agent_config(agent_name)
        for key, value in agent_config.items():
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            env_vars[key] = str(value)

        self._apply_agent_defaults_and_validate(agent_name, env_vars)

        return env_vars

    def _apply_agent_defaults_and_validate(self, agent_name: str, env_vars: Dict[str, str]) -> None:
        """Apply minimal defaults and validate enum-like agent options."""
        if agent_name != "codex":
            return

        allowed_reasoning_effort = {"low", "medium", "high"}

        raw = env_vars.get("CODEX_REASONING_EFFORT") or env_vars.get("MODEL_REASONING_EFFORT")
        if raw is None or not str(raw).strip():
            env_vars["CODEX_REASONING_EFFORT"] = "medium"
            return

        value = str(raw).strip().lower()
        env_vars["CODEX_REASONING_EFFORT"] = value

    def get_llm_by_name(self, llm_name: str) -> Optional[Dict[str, Any]]:
        """Get LLM configuration by name."""
        llm_configs = self._config.get("llm", {})
        return llm_configs.get(llm_name)


class DatasetLoader:
    """Load dataset from HuggingFace."""

    def __init__(self, config_loader: InferConfigLoader):
        """Initialize the dataset loader."""
        self.config = config_loader
        self._setup_hf_env()

    def _setup_hf_env(self) -> None:
        """Set up HuggingFace environment variables."""
        env_vars = self.config.env_vars

        if env_vars.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = env_vars["HF_ENDPOINT"]

        if env_vars.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = env_vars["HF_TOKEN"]

    def load_dataset(
        self,
        dataset: str,
        split: Optional[str] = None,
        levels: Optional[List[int]] = None,
        task_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace."""
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)

        from datasets import load_dataset

        env_vars = self.config.env_vars
        hf_token = env_vars.get("HF_TOKEN")

        dataset_name = str(dataset or "").strip()
        if not dataset_name:
            raise ValueError("dataset parameter is required. Please specify --dataset (e.g., 'LiberCoders/FeatureBench')")

        if split is None:
            raise ValueError("split parameter is required. Please specify --split (e.g., 'lite', 'full')")

        all_instances = []

        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                token=hf_token
            )

            for item in dataset:
                item_dict = dict(item)

                instance_id = item_dict.get("instance_id", "")
                last_segment = instance_id.split(".")[-1] if instance_id else ""
                if last_segment == "lv1":
                    level = 1
                elif last_segment == "lv2":
                    level = 2
                else:
                    raise RuntimeError(f"Unknown level: {last_segment} in instance_id: {instance_id}")

                item_dict["level"] = level

                if levels is not None and level not in levels:
                    continue

                if task_ids is not None and instance_id not in task_ids:
                    continue

                all_instances.append(item_dict)

        except Exception as e:
            print(f"Warning: Failed to load split '{split}': {e}")

        return all_instances
