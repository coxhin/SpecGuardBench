from __future__ import annotations

from eval_pipeline.models import DifferentialResult, SpecCertOutcome
from eval_pipeline.stages.stage5_metrics import _sound_certs, _unsound_certs

INSTANCE = "inst-1"


def _outcome(spec_id: str, *, gold_kind: str, mutant_kind: str) -> SpecCertOutcome:
    return SpecCertOutcome(
        spec_id=spec_id,
        has_certificate=True,
        gold_triggered=gold_kind != "passed",
        mutant_triggered=mutant_kind != "passed",
        gold_kind=gold_kind,
        mutant_kind=mutant_kind,
    )


def _nc(firing=(), with_arm=()):
    return {
        "cross_spec_firing_certs": list(firing),
        "per_cert": {k: {"runs": 1, "fired": 0} for k in with_arm},
    }


def _key(baseline: str, model: str, spec_id: str) -> str:
    return f"{INSTANCE}::{baseline}_{model}::{spec_id}"


def test_sound_requires_all_three_arms() -> None:
    dr = DifferentialResult(INSTANCE, "b", "m", {
        "S1": _outcome("S1", gold_kind="passed", mutant_kind="assertion"),
        "S2": _outcome("S2", gold_kind="passed", mutant_kind="assertion"),
        "S3": _outcome("S3", gold_kind="assertion", mutant_kind="assertion"),
        "S4": _outcome("S4", gold_kind="passed", mutant_kind="passed"),
    })
    nc = _nc(
        firing=[_key("b", "m", "S2")],
        with_arm=[_key("b", "m", "S1"), _key("b", "m", "S2")],
    )
    out = _sound_certs([dr], nc)
    assert out["counts"] == {
        "certs": 4,
        "sound": 1,
        "not_valid": 2,
        "control_firing": 1,
        "sound_untested_on_controls": 0,
    }
    assert out["sound_cert_rate"] == 0.25


def test_sound_without_control_arm_is_counted_but_flagged() -> None:
    dr = DifferentialResult(INSTANCE, "b", "m", {
        "S1": _outcome("S1", gold_kind="passed", mutant_kind="assertion"),
    })
    out = _sound_certs([dr], _nc())
    assert out["counts"]["sound"] == 1
    assert out["counts"]["sound_untested_on_controls"] == 1


def test_firing_set_from_other_baseline_does_not_leak() -> None:
    dr = DifferentialResult(INSTANCE, "b", "m", {
        "S1": _outcome("S1", gold_kind="passed", mutant_kind="assertion"),
    })
    nc = _nc(firing=[_key("other", "x", "S1")], with_arm=[_key("b", "m", "S1")])
    out = _sound_certs([dr], nc)
    assert out["counts"]["sound"] == 1
    assert out["counts"]["control_firing"] == 0


def test_unsound_certs_file_a_gold_and_control_firing_once() -> None:
    dr = DifferentialResult(INSTANCE, "b", "m", {
        "S1": _outcome("S1", gold_kind="assertion", mutant_kind="assertion"),
        "S2": _outcome("S2", gold_kind="passed", mutant_kind="assertion"),
    })
    nc = _nc(
        firing=[_key("b", "m", "S1")],
        with_arm=[_key("b", "m", "S1"), _key("b", "m", "S2")],
    )

    unsound = _unsound_certs([dr], nc)
    assert unsound["counts"] == {
        "certs": 2, "unsound": 1, "gold_only": 0, "control_only": 0, "both": 1,
    }
    sound = _sound_certs([dr], nc)
    assert sound["counts"] == {
        "certs": 2, "sound": 1, "not_valid": 1, "control_firing": 0,
        "sound_untested_on_controls": 0,
    }
