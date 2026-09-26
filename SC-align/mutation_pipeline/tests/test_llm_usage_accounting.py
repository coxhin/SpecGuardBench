import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

_SC_ALIGN = Path(__file__).resolve().parents[2]
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from featurebench.llm.llm_caller import LLMCaller
from mutation_pipeline.pipeline import MutationPipeline


def _caller() -> LLMCaller:
    return LLMCaller(SimpleNamespace(llm_config={}))


def _responses_usage(inp, out, cached=0, cache_write=0, reasoning=0):
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out, total_tokens=inp + out,
        input_tokens_details=SimpleNamespace(
            cached_tokens=cached, cache_write_tokens=cache_write),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _chat_usage(prompt, completion, cached=None):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        total_tokens=prompt + completion,
        input_tokens=prompt, output_tokens=completion,
        prompt_tokens_details=SimpleNamespace(audio_tokens=None, cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
        input_tokens_details=None,
    )


def test_responses_usage_is_recorded_with_cache_and_reasoning():
    caller = _caller()
    caller._record_usage(_responses_usage(3251, 5, cached=2048, cache_write=64, reasoning=3))
    assert caller.usage_totals == {
        "calls": 1, "prompt_tokens": 3251, "completion_tokens": 5,
        "total_tokens": 3256, "cached_prompt_tokens": 2048,
        "cache_write_tokens": 64, "reasoning_tokens": 3,
    }


def test_chat_usage_is_recorded_under_the_same_keys():
    caller = _caller()
    caller._record_usage(_chat_usage(4358, 1))
    assert caller.usage_totals["prompt_tokens"] == 4358
    assert caller.usage_totals["completion_tokens"] == 1
    assert caller.usage_totals["total_tokens"] == 4359


def test_a_null_cached_count_contributes_zero_rather_than_failing():
    caller = _caller()
    caller._record_usage(_chat_usage(4358, 1, cached=None))
    assert caller.usage_totals["cached_prompt_tokens"] == 0
    assert caller.usage_totals["reasoning_tokens"] == 0


def test_usage_accumulates_across_calls():
    caller = _caller()
    caller._record_usage(_responses_usage(100, 10, cached=50))
    caller._record_usage(_responses_usage(200, 20, cached=80))
    assert caller.usage_totals["calls"] == 2
    assert caller.usage_totals["prompt_tokens"] == 300
    assert caller.usage_totals["completion_tokens"] == 30
    assert caller.usage_totals["cached_prompt_tokens"] == 130


def test_a_response_without_usage_still_counts_as_a_call():
    caller = _caller()
    caller._record_usage(None)
    assert caller.usage_totals["calls"] == 1
    assert caller.usage_totals["prompt_tokens"] == 0


def test_total_tokens_is_derived_when_the_provider_omits_it():
    caller = _caller()
    caller._record_usage(SimpleNamespace(prompt_tokens=70, completion_tokens=30))
    assert caller.usage_totals["total_tokens"] == 100


def test_cached_tokens_are_not_added_to_the_prompt_total():
    caller = _caller()
    caller._record_usage(_responses_usage(1000, 10, cached=900))
    assert caller.usage_totals["prompt_tokens"] == 1000
    assert caller.usage_totals["cached_prompt_tokens"] == 900


def test_a_snapshot_is_a_copy_and_not_the_live_account():
    caller = _caller()
    caller._record_usage(_responses_usage(100, 10))
    snapshot = caller.snapshot_usage()
    caller._record_usage(_responses_usage(200, 20))
    assert snapshot["prompt_tokens"] == 100
    assert caller.snapshot_usage()["prompt_tokens"] == 300


def test_concurrent_responses_do_not_drop_each_others_counts():
    caller = _caller()
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in range(8):
            executor.submit(
                lambda: [caller._record_usage(_responses_usage(10, 1, cached=5))
                         for _ in range(500)]
            )
    assert caller.usage_totals["calls"] == 4000
    assert caller.usage_totals["prompt_tokens"] == 40000
    assert caller.usage_totals["cached_prompt_tokens"] == 20000


def _pipeline(tmp_path: Path, llm_a: LLMCaller, llm_b: LLMCaller):
    pipeline = MutationPipeline.__new__(MutationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.llm_a = llm_a
    pipeline.llm_b = llm_b
    pipeline._stats = defaultdict(int)
    pipeline.config = SimpleNamespace(
        preservation_probe_k=0,
        llm_a_name="llm-a",
        llm_b_name="llm-b",
        timeout=600,
        hypothesis_timeout=120,
    )
    return pipeline


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_instances_record_holds_only_its_own_calls(tmp_path):
    a, b = _caller(), _caller()
    pipeline = _pipeline(tmp_path, a, b)

    a._record_usage(_responses_usage(100, 10, cached=40))
    before = pipeline._usage_snapshot()
    a._record_usage(_responses_usage(300, 30, cached=90, reasoning=7))
    b._record_usage(_responses_usage(50, 5))
    pipeline._save_instance_usage("inst-2", before)

    record = _read(tmp_path / "inst-2" / "llm_usage.json")
    assert record["instance_id"] == "inst-2"
    assert record["llm_a"] == {
        "calls": 1, "prompt_tokens": 300, "completion_tokens": 30,
        "total_tokens": 330, "cached_prompt_tokens": 90,
        "cache_write_tokens": 0, "reasoning_tokens": 7,
    }
    assert record["llm_b"]["prompt_tokens"] == 50
    assert record["llm_b"]["calls"] == 1


def test_a_failed_instance_still_records_what_it_spent(tmp_path):
    a, b = _caller(), _caller()
    pipeline = _pipeline(tmp_path, a, b)

    def _blow_up(pipeline_input):
        a._record_usage(_responses_usage(700, 70))
        raise RuntimeError("step 3 container died")

    pipeline.run_instance = _blow_up
    pipeline.run_all([SimpleNamespace(instance_id="inst-fail")])

    record = _read(tmp_path / "inst-fail" / "llm_usage.json")
    assert record["llm_a"]["prompt_tokens"] == 700


def test_run_totals_are_summed_from_the_per_instance_records(tmp_path):
    a, b = _caller(), _caller()
    pipeline = _pipeline(tmp_path, a, b)
    spend = {"i1": (100, 10), "i2": (300, 30)}

    def _run(pipeline_input):
        prompt, completion = spend[pipeline_input.instance_id]
        a._record_usage(_responses_usage(prompt, completion, cached=prompt // 2))
        b._record_usage(_responses_usage(20, 2))
        return []

    pipeline.run_instance = _run
    pipeline.run_all([SimpleNamespace(instance_id=i) for i in spend])

    usage = _read(tmp_path / "pipeline_stats.json")["llm_usage"]
    assert set(usage["per_instance"]) == {"i1", "i2"}
    assert usage["per_instance"]["i2"]["llm_a"]["prompt_tokens"] == 300
    assert usage["llm_a"]["prompt_tokens"] == 400
    assert usage["llm_a"]["cached_prompt_tokens"] == 200
    assert usage["llm_b"]["calls"] == 2


def test_totals_cover_instances_an_earlier_process_ran(tmp_path):
    earlier = tmp_path / "inst-done"
    earlier.mkdir()
    (earlier / "llm_usage.json").write_text(json.dumps({
        "instance_id": "inst-done",
        "llm_a": {"calls": 1, "prompt_tokens": 999, "completion_tokens": 9,
                  "total_tokens": 1008, "cached_prompt_tokens": 0,
                  "cache_write_tokens": 0, "reasoning_tokens": 0},
        "llm_b": {"calls": 1, "prompt_tokens": 11, "completion_tokens": 1,
                  "total_tokens": 12, "cached_prompt_tokens": 0,
                  "cache_write_tokens": 0, "reasoning_tokens": 0},
    }), encoding="utf-8")

    a, b = _caller(), _caller()
    pipeline = _pipeline(tmp_path, a, b)
    pipeline.run_instance = lambda pi: (a._record_usage(_responses_usage(1, 1)), [])[1]
    pipeline.run_all([SimpleNamespace(instance_id="inst-new")], completed_instances=1)

    usage = _read(tmp_path / "pipeline_stats.json")["llm_usage"]
    assert set(usage["per_instance"]) == {"inst-done", "inst-new"}
    assert usage["llm_a"]["prompt_tokens"] == 1000
    assert usage["llm_b"]["prompt_tokens"] == 11


def test_rerunning_an_instance_adds_to_what_the_failed_attempt_spent(tmp_path):
    a, b = _caller(), _caller()
    pipeline = _pipeline(tmp_path, a, b)

    before = pipeline._usage_snapshot()
    a._record_usage(_responses_usage(700, 70))
    pipeline._save_instance_usage("inst-retried", before)

    before = pipeline._usage_snapshot()
    a._record_usage(_responses_usage(500, 50))
    pipeline._save_instance_usage("inst-retried", before)

    record = _read(tmp_path / "inst-retried" / "llm_usage.json")
    assert record["llm_a"]["prompt_tokens"] == 1200
    assert record["llm_a"]["calls"] == 2


def test_a_malformed_record_is_skipped_rather_than_failing_the_run(tmp_path):
    broken = tmp_path / "inst-broken"
    broken.mkdir()
    (broken / "llm_usage.json").write_text("{not json", encoding="utf-8")

    pipeline = _pipeline(tmp_path, _caller(), _caller())
    assert pipeline._collect_instance_usage() == {}
