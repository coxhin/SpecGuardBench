import sys
from pathlib import Path
from unittest import mock

import pytest

from . import test_concurrency as _stubs

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.egress import ApiOnlyEgress, hosts_from_env


def test_hosts_from_env_takes_every_url_valued_entry():
    env = {
        "ANTHROPIC_BASE_URL": "https://anthropic.example.com",
        "OPENAI_BASE_URL": "https://openai.example.com/v1",
        "OPENCODE_BASE_URL": "https://anthropic.example.com/v1",
        "OPENAI_API_KEY": "sk-not-a-url",
        "ANTHROPIC_MODEL": "claude-opus-5",
    }
    assert hosts_from_env(env) == {"anthropic.example.com", "openai.example.com"}


def test_empty_allow_list_is_refused():
    with pytest.raises(ValueError, match="cut the agent off"):
        ApiOnlyEgress([], run_id="r", client=mock.MagicMock())


def _started(client):
    egress = ApiOnlyEgress(
        ["api.example.com", "API.example.com "], run_id="r1", client=client
    )
    proxy = client.containers.create.return_value
    proxy.logs.return_value = b"listening on 0.0.0.0:3128; allowed hosts: api.example.com\n"
    with mock.patch("eval_pipeline.egress.ApiOnlyEgress._ship_proxy_script"):
        egress.start()
    return egress


def test_start_builds_an_internal_network_and_a_proxy_on_both_networks():
    client = mock.MagicMock()
    egress = _started(client)

    client.networks.create.assert_called_once_with(
        "sc-egress-r1", driver="bridge", internal=True
    )
    create_kwargs = client.containers.create.call_args.kwargs
    assert create_kwargs["name"] == "sc-egress-proxy-r1"
    assert create_kwargs["environment"]["EGRESS_ALLOW_HOSTS"] == "api.example.com"
    client.networks.create.return_value.connect.assert_called_once_with(
        client.containers.create.return_value
    )
    client.containers.create.return_value.start.assert_called_once()
    assert egress.allow_hosts == ["api.example.com"]


def test_restrict_leaves_the_bridge_before_joining_the_internal_network():
    client = mock.MagicMock()
    egress = _started(client)
    container = mock.MagicMock(id="abc123", short_id="abc")
    order = []
    client.networks.get.return_value.disconnect.side_effect = (
        lambda *a, **k: order.append("bridge.disconnect")
    )
    client.networks.create.return_value.connect.side_effect = (
        lambda *a, **k: order.append("internal.connect")
    )

    assert egress.is_restricted(container) is False
    egress.restrict(container)

    client.networks.get.assert_called_with("bridge")
    assert order[-2:] == ["bridge.disconnect", "internal.connect"]
    assert egress.is_restricted(container) is True


def test_restrict_before_start_is_an_error():
    egress = ApiOnlyEgress(["h"], run_id="r", client=mock.MagicMock())
    with pytest.raises(RuntimeError, match="before start"):
        egress.restrict(mock.MagicMock(id="x"))


def test_shell_prelude_names_the_proxy_in_both_spellings():
    egress = ApiOnlyEgress(["h"], run_id="r9", client=mock.MagicMock())
    prelude = egress.shell_prelude()
    assert prelude.endswith("&& ")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert f"{var}=http://sc-egress-proxy-r9:3128" in prelude
    assert "NO_PROXY=localhost,127.0.0.1" in prelude


def test_stop_saves_the_decision_log_then_removes_everything(tmp_path):
    client = mock.MagicMock()
    egress = _started(client)
    proxy = client.containers.create.return_value
    proxy.logs.return_value = b"DENY CONNECT pypi.org:443\n"

    egress.stop(log_path=tmp_path / "egress_proxy.log")

    assert (tmp_path / "egress_proxy.log").read_bytes() == b"DENY CONNECT pypi.org:443\n"
    proxy.remove.assert_called_once_with(force=True)
    client.networks.create.return_value.remove.assert_called_once()


def _adapter():
    from eval_pipeline.adapters.claude_code import ClaudeCodeAdapter

    return ClaudeCodeAdapter("claude_code", "m", "v", {}, {})


def test_adapter_prefixes_commands_only_when_egress_is_set():
    adapter = _adapter()
    assert adapter._with_egress("claude -p x") == "claude -p x"
    egress = ApiOnlyEgress(["h"], run_id="r", client=mock.MagicMock())
    adapter.egress = egress
    assert adapter._with_egress("claude -p x") == egress.shell_prelude() + "claude -p x"


def test_adapter_does_not_reinstall_once_restricted(monkeypatch):
    adapter = _adapter()
    egress = mock.MagicMock()
    egress.is_restricted.return_value = True
    adapter.egress = egress
    install = mock.MagicMock(return_value=True)
    monkeypatch.setattr(adapter, "_install_tool", install)
    monkeypatch.setattr(adapter, "_install_step_gate", lambda c: True)
    monkeypatch.setattr(adapter, "_write_file_to_container", lambda *a, **k: None)

    assert adapter.setup_container(mock.MagicMock(), mock.MagicMock(), "prompt") is True
    install.assert_not_called()


def test_adapter_installs_when_not_yet_restricted(monkeypatch):
    adapter = _adapter()
    egress = mock.MagicMock()
    egress.is_restricted.return_value = False
    adapter.egress = egress
    install = mock.MagicMock(return_value=True)
    monkeypatch.setattr(adapter, "_install_tool", install)
    monkeypatch.setattr(adapter, "_install_step_gate", lambda c: True)
    monkeypatch.setattr(adapter, "_write_file_to_container", lambda *a, **k: None)

    assert adapter.setup_container(mock.MagicMock(), mock.MagicMock(), "prompt") is True
    install.assert_called_once()
