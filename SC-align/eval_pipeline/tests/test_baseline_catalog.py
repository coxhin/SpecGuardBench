import sys
from pathlib import Path

import pytest

from . import test_concurrency as _stubs

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.config import (
    AGENT_MODEL_ENV_VAR,
    KNOWN_ADAPTERS,
    BaselineConfig,
    _load_baseline_catalog,
)


def test_stock_table_needs_neither_field():
    cat = _load_baseline_catalog(
        {"baseline": {"opencode": {"model": "deepseek-v4-flash", "tool_version": "1"}}}
    )
    b = cat["opencode"]
    assert (b.agent_config, b.adapter) == ("opencode", "opencode")


def test_second_endpoint_keeps_its_own_section_and_shares_the_cli():
    cat = _load_baseline_catalog({"baseline": {
        "opencode": {"model": "deepseek-v4-flash", "tool_version": "1"},
        "opencode_glm": {
            "adapter": "opencode", "agent_config": "opencode_glm",
            "model": "glm-5.2", "tool_version": "1",
        },
    }})
    glm = cat["opencode_glm"]
    assert glm.adapter == "opencode"
    assert glm.agent_config == "opencode_glm"
    assert cat["opencode"].agent_config == "opencode"


def test_adapter_defaults_to_agent_config_not_name():
    cat = _load_baseline_catalog({"baseline": {
        "cc_alt": {"agent_config": "claude_code", "model": "opus", "tool_version": "1"},
    }})
    assert cat["cc_alt"].adapter == "claude_code"


def test_unknown_adapter_fails_at_load_not_in_stage2():
    with pytest.raises(ValueError, match="opencode_glm.*not one of"):
        _load_baseline_catalog({"baseline": {
            "opencode_glm": {"model": "glm-5.2", "tool_version": "1"},
        }})


def test_every_adapter_with_a_model_env_var_is_known():
    assert set(AGENT_MODEL_ENV_VAR) <= KNOWN_ADAPTERS


def test_stage2_factory_and_model_env_var_key_on_adapter():
    from eval_pipeline.stages import stage2_baseline_execution as s2
    from eval_pipeline.adapters.opencode import OpenCodeAdapter

    glm = BaselineConfig(
        name="opencode_glm", model="glm-5.2", agent_config="opencode_glm",
        tool_version="1", adapter="opencode",
    )
    adapter = s2._create_adapter(glm, {"OPENCODE_BASE_URL": "https://x/v1"})
    assert isinstance(adapter, OpenCodeAdapter)
    assert adapter.name == "opencode_glm"
    assert AGENT_MODEL_ENV_VAR[glm.adapter] == "OPENCODE_MODEL"

    with pytest.raises(ValueError, match="not one of"):
        BaselineConfig(
            name="x", model="m", agent_config="x", tool_version="1", adapter="nope",
        )


def test_run_metadata_round_trip_and_pre_adapter_metadata():
    from eval_pipeline.run import _baseline_to_dict

    glm = BaselineConfig(
        name="opencode_glm", model="glm-5.2", agent_config="opencode_glm",
        tool_version="1", adapter="opencode",
    )
    assert BaselineConfig(**_baseline_to_dict(glm)) == glm

    old = {"name": "codex", "model": "gpt-5.4", "agent_config": "codex",
           "tool_version": "0.117.0"}
    assert BaselineConfig(**old).adapter == "codex"
