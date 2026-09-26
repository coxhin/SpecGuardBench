"""Pipeline configuration and data loading for the evaluation pipeline."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from featurebench.infer.config import InferConfigLoader, DatasetLoader

MUTATION_OUTPUT_SCHEMA_VERSION = 3

DEFAULT_CONTROL_ARM_K = 3


KNOWN_ADAPTERS = frozenset({
    "claude_code", "codex", "gemini_cli", "opencode", "openhands",
})


@dataclass
class BaselineConfig:
    """Configuration for a single baseline agent."""
    name: str
    model: str
    agent_config: str
    tool_version: str
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    price_cache_read_per_mtok: float = 0.0
    price_cache_write_per_mtok: float = 0.0
    adapter: str = ""

    def __post_init__(self) -> None:
        self.adapter = self.adapter or self.agent_config
        if self.adapter not in KNOWN_ADAPTERS:
            known = ", ".join(sorted(KNOWN_ADAPTERS))
            raise ValueError(
                f"baseline '{self.name}': adapter '{self.adapter}' is not one of "
                f"{known}. Set adapter = \"<one of those>\" when the baseline's "
                f"name or agent_config is not itself an adapter name."
            )


AGENT_MODEL_ENV_VAR = {
    "claude_code": "ANTHROPIC_MODEL",
    "codex": "OPENAI_MODEL",
    "gemini_cli": "GEMINI_MODEL",
    "opencode": "OPENCODE_MODEL",
}


@dataclass
class EvalPipelineConfig:
    """Configuration for an evaluation pipeline run."""
    output_dir: Path = field(default_factory=lambda: Path("SC-align/eval_output"))
    mutation_output: Path = field(default_factory=lambda: Path("SC-align/mutation_output/output.jsonl"))
    n_concurrent: int = 4
    timeout: Optional[int] = None
    max_agent_steps: Optional[int] = None
    hypothesis_timeout: int = 120
    restore_test_with_reverse_patch: bool = False
    instance_ids: Optional[List[str]] = None
    baselines: List[BaselineConfig] = field(default_factory=list)
    baseline_catalog: Dict[str, BaselineConfig] = field(default_factory=dict)
    proxy_port: Optional[int] = None
    gpu_ids: Optional[str] = None
    dataset: str = "LiberCoders/FeatureBench"
    split: str = "full"
    stages: Optional[List[int]] = None
    skip_existing: bool = False
    harness_form: Literal[
        "form1b_form6", "hypofuzz", "saved_inputs", "factory_oracle"
    ] = "form1b_form6"
    hypofuzz_timeout_seconds: int = 60
    saved_inputs_replay_timeout_seconds: int = 120
    container_isolation: Literal["instance", "spec", "arm"] = "arm"
    control_arm_k: Optional[int] = DEFAULT_CONTROL_ARM_K
    stage2_reference_state: Literal["gold", "mutant"] = "gold"
    egress: Literal["open", "api_only"] = "open"
    egress_allow_hosts: List[str] = field(default_factory=list)
    spec_subset: Optional[List[Tuple[str, str]]] = None
    spec_subset_path: Optional[Path] = None
    egress_proxy_image: str = "python:3.12-alpine"


def _load_baseline_catalog(raw: Dict[str, Any]) -> Dict[str, BaselineConfig]:
    """Read the ``[eval_pipeline.baseline.<name>]`` tables into a name-keyed catalog."""
    catalog: Dict[str, BaselineConfig] = {}
    for name, b in (raw.get("baseline") or {}).items():
        if not isinstance(b, dict):
            continue
        agent_config = b.get("agent_config", name)
        catalog[name] = BaselineConfig(
            name=name,
            model=b.get("model", ""),
            agent_config=agent_config,
            adapter=b.get("adapter", agent_config),
            tool_version=b.get("tool_version", ""),
            price_input_per_mtok=float(b.get("price_input_per_mtok", 0.0)),
            price_output_per_mtok=float(b.get("price_output_per_mtok", 0.0)),
            price_cache_read_per_mtok=float(b.get("price_cache_read_per_mtok", 0.0)),
            price_cache_write_per_mtok=float(b.get("price_cache_write_per_mtok", 0.0)),
        )
    return catalog


def load_eval_config_from_toml(config_loader: InferConfigLoader) -> EvalPipelineConfig:
    """Load EvalPipelineConfig from the [eval_pipeline] section of config_eval.toml."""
    raw = config_loader.raw_config.get("eval_pipeline", {})

    baseline_catalog = _load_baseline_catalog(raw)

    def _optional_int(value: Any) -> Optional[int]:
        """A budget the config may leave unset. 0 also means unset."""
        if value is None or value == "":
            return None
        return int(value) or None

    raw_instance_ids = raw.get("instance_ids", None)
    if isinstance(raw_instance_ids, list) and len(raw_instance_ids) == 0:
        instance_ids = None
    else:
        instance_ids = raw_instance_ids if raw_instance_ids else None

    raw_stages = raw.get("stages", None)
    stages = raw_stages if raw_stages else None

    return EvalPipelineConfig(
        output_dir=Path(raw.get("output_dir", "SC-align/eval_output")),
        n_concurrent=int(raw.get("n_concurrent", 4)),
        timeout=_optional_int(raw.get("timeout")),
        max_agent_steps=_optional_int(raw.get("max_agent_steps")),
        hypothesis_timeout=int(raw.get("hypothesis_timeout", 120)),
        restore_test_with_reverse_patch=bool(raw.get("restore_test_with_reverse_patch", False)),
        instance_ids=instance_ids,
        baselines=[],
        baseline_catalog=baseline_catalog,
        proxy_port=raw.get("proxy_port") or None,
        gpu_ids=raw.get("gpu_ids") or None,
        dataset=raw.get("dataset", "LiberCoders/FeatureBench"),
        split=raw.get("split", "full"),
        stages=stages,
        skip_existing=bool(raw.get("skip_existing", False)),
        harness_form=raw.get("harness_form", "form1b_form6"),
        hypofuzz_timeout_seconds=int(raw.get("hypofuzz_timeout_seconds", 60)),
        container_isolation=raw.get("container_isolation", "arm"),
        control_arm_k=(
            int(raw["control_arm_k"]) if "control_arm_k" in raw
            else DEFAULT_CONTROL_ARM_K
        ),
        saved_inputs_replay_timeout_seconds=int(
            raw.get("saved_inputs_replay_timeout_seconds", 120)
        ),
        stage2_reference_state=raw.get("stage2_reference_state", "gold"),
        egress=raw.get("egress", "open"),
        egress_allow_hosts=list(raw.get("egress_allow_hosts", [])),
        egress_proxy_image=raw.get("egress_proxy_image", "python:3.12-alpine"),
    )


def load_mutation_outputs(mutation_output_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read mutation output.jsonl and return dict grouped by instance_id."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    if not mutation_output_path.exists():
        return result

    with open(mutation_output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = record.get("instance_id", "")
            if iid:
                result.setdefault(iid, []).append(record)

    return result


def load_hf_metadata(
    config_loader: InferConfigLoader,
    config: EvalPipelineConfig,
) -> Dict[str, Dict[str, Any]]:
    """Load HuggingFace dataset metadata and return dict keyed by instance_id."""
    dataset_loader = DatasetLoader(config_loader)
    instances = dataset_loader.load_dataset(
        dataset=config.dataset,
        split=config.split,
        task_ids=config.instance_ids,
    )
    hf_data: Dict[str, Dict[str, Any]] = {}
    for item in instances:
        iid = item.get("instance_id", "")
        if iid:
            hf_data[iid] = item
    return hf_data
