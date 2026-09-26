# SpecGuardBench

This artifact contains the SpecGuardBench dataset and the code that builds and evaluates it:

- `data/`: the Full Set (291 tasks over 30 FeatureBench instances) and the Lite Set (100 tasks).
- `SC-align/mutation_pipeline/`: the construction pipeline, which turns a FeatureBench instance into validation tasks.
- `SC-align/eval_pipeline/`: the evaluation pipeline, which runs a coding agent on each task and scores the validation harness it writes.
- `featurebench/`: a vendored subset of [FeatureBench](https://github.com/LiberCoders/FeatureBench), which provides container management, agent installation, and the LLM client.

## Layout

```
data/
  full_set.jsonl                 291 tasks, one JSON object per line
  lite_set.tsv                   100 (instance_id, spec_id) pairs
  instances.jsonl                the 30 FeatureBench instances
SC-align/
  mutation_pipeline/             construction pipeline (Steps 1-6, control filter, prompts)
  eval_pipeline/                 evaluation pipeline (Stages 1-5, agent adapters, harness driver, prompts)
  factory_oracle_harness_guide.md  harness interface given to the agent
  scripts/                       data preparation and launchers
featurebench/                    FeatureBench subset
tests/                           entry-point tests for both pipelines
config_eval_example.toml         template for config_eval.toml
config_mutation_example.toml     template for config_mutation.toml
```

## Data

Each line of `data/full_set.jsonl` is one task:

| Field | Content |
|---|---|
| `schema_version` | record format version (3) |
| `instance_id` | FeatureBench instance |
| `spec` | the atomic specification: `spec_id`, `text`, `category`, `sources`, and `code_mapping` (the gold code it maps to, which the agent is not shown); some records also carry `independence_check`, a note from decomposition, or `merged_from` |
| `gold_patch` | the gold implementation: the reverse of `mask_patch`, without the fail-to-pass test files |
| `mask_patch` | FeatureBench's patch that removes the feature from the repository |
| `test_patch` | FeatureBench's patch that deletes the instance's fail-to-pass test files; the pipelines reverse-apply it to restore them |
| `mutant_patch` | the violating implementation, as a patch on top of the gold implementation |
| `proof` | the differential witness: `witness_factory`, `witness_observer`, the assembled `witness_script`, the `case_contract`, the `trigger_condition`, the observations on the gold and violating implementations (`actual_gold_output`, `actual_mutant_output`), and a free-text `divergence_type` |
| `metadata` | `mutations` and `mutation_strategy` (the synthesizing LLM's account of the change), `filter_survival`, `step2_attempt`, and `preservation_probes` (the witnesses of other tasks run against this violating implementation) |

The valid alternative implementations of a task are violating implementations of other tasks in the same instance that the construction pipeline probed with the task's witness and whose preservation audit did not find the task's specification violated. These probes are listed in each violating implementation's `metadata.preservation_probes`. A probe's `verdict` records only whether the witness observed the same behavior as on the gold implementation (`preserved`, `diverged`, or `mutant_error`); admission rests on the audit. The evaluation pipeline scores up to three alternatives per task, those sharing the most code with the task (`control_arm_k` in `config_eval.toml`).

`data/lite_set.tsv` lists the Lite Set as tab-separated `instance_id` and `spec_id` pairs; lines starting with `#` are comments.

## Requirements

- Linux with Docker. The per-instance FeatureBench images (`docker.io/libercoders/featurebench-specs_*`) are pulled on first use.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Access to the Hugging Face dataset `LiberCoders/FeatureBench`, from which both pipelines read instance metadata.
- API access for the agents under evaluation (evaluation) or for the two LLMs (construction).

## Setup

```bash
uv sync
cp config_eval_example.toml config_eval.toml          # agent keys and endpoints
cp config_mutation_example.toml config_mutation.toml  # LLM keys and endpoints (construction only)
```

Fill in `HF_TOKEN` (and `HF_ENDPOINT` if you use a mirror) under `[env_vars]`, and the API key and base URL of each agent under `[infer_config.*]` in `config_eval.toml`, or of each LLM under `[llm.*]` in `config_mutation.toml`.

## Evaluating an agent

```bash
./SC-align/scripts/run_eval.sh
```

The variables at the top of `SC-align/scripts/run_eval.sh` define the run; arguments given to the script are appended to the command.

| Variable | Default | Meaning |
|---|---|---|
| `BASELINE` | `codex` | agent to evaluate, a `[eval_pipeline.baseline.<name>]` table in `config_eval.toml` |
| `MODEL` | empty | model override; empty uses the table's `model` |
| `MUTATION_OUTPUT` | `data/full_set.jsonl` | task file |
| `SPEC_SUBSET` | empty | set to `data/lite_set.tsv` for the Lite Set |
| `MAX_AGENT_STEPS` | `80` | step budget, in model calls per task |
| `STAGE2_REFERENCE_STATE` | empty (gold) | `mutant` shows the violating implementation to the agent as the reference |
| `EGRESS` | `api_only` | restricts the agent container's network to the model API after tool installation |
| `N_CONCURRENT` | `12` | tasks in flight |
| `TAG` | `full_set` | suffix of the run directory name |

The template defines one baseline table for each agent in the paper:

| `BASELINE` | Agent CLI | Model id | Paper |
|---|---|---|---|
| `codex` | Codex 0.117.0 | `gpt-5.6-sol` | Codex / GPT-5.6-sol |
| `claude_code` | Claude Code 2.1.77 | `claude-opus-5` | Claude Code / Claude Opus 5 |
| `opencode` | OpenCode 1.18.25 | `deepseek-v4-pro` | OpenCode / DeepSeek-V4-Pro |
| `opencode_glm` | OpenCode 1.18.25 | `glm-5.2` | OpenCode / GLM-5.3 |

One run evaluates one agent. The pipeline runs five stages:

1. Task assembly: joins the task file with FeatureBench metadata.
2. Agent execution: runs the agent in the instance container; the agent writes its validation harness to `/tmp/eval_output.json`.
3. Harness execution: runs the harness's checks on its own scenarios against the gold, violating, and alternative implementations, and measures coverage on the reference.
4. Per-task analysis.
5. Metrics.

Results go to `SC-align/eval_output/runs/<timestamp>__<TAG>/`. The headline metrics are in `eval_reports/summary.json`:

| Paper (Table 1) | `summary.json` |
|---|---|
| Detection | `detection_rate` |
| False alarm (rejects the gold or an alternative implementation) | `unsound_certs.unsound_cert_rate` |
| Faithfulness | `sound_cert_rate` |
| Edge coverage | `metric_A.edge_depth.mean` |
| Cost | `cost.total_cost_usd` divided by the number of tasks |
| Gold rejection (Table 2) | `gold_firing_rate` |

`eval_reports/per_instance.json` and `per_category.json` break the metrics down, and `differential/all_results.jsonl` holds the per-task outcomes. To resume an interrupted run, pass `--resume <run dir> --skip-existing`; to recompute the metrics only, add `--stages 4 5`.

Agent behavior and model serving vary across runs, so a rerun reproduces the paper's numbers approximately, not exactly.

## Constructing tasks

Prepare the inputs from the FeatureBench dataset:

```bash
.venv/bin/python SC-align/scripts/export_problem_statements.py --split full   # -> SC-align/problem_statements/
.venv/bin/python SC-align/scripts/extract_task_and_interfaces.py              # -> SC-align/extracted_v2/
.venv/bin/python SC-align/scripts/export_ground_truth.py --split full         # -> SC-align/ground_truth/
```

Then run the construction pipeline on the instances in `data/instances.jsonl`:

```bash
./SC-align/scripts/run_mutation.sh
```

The launcher defaults are the settings used for the paper: `LLM_A=claude-opus-5` synthesizes the violating implementations and witnesses, `LLM_B=gpt-5.6-sol` audits them with three votes (`STEP6_VOTES=3`), each specification gets up to three synthesis attempts (`STEP2_MAX_ATTEMPTS=3`), and each candidate is probed with the witnesses of up to three other specifications that share the most code with its target (`PRESERVATION_PROBE_K=3`), with every probe pair also run in the reverse direction. `LLM_A` and `LLM_B` name `[llm.*]` sections of `config_mutation.toml`, and must be different endpoints. LLM sampling makes a rerun produce a different set of tasks.

Results go to `SC-align/mutation_output/runs/<timestamp>__<TAG>/`; its `output.jsonl` has the format of `data/full_set.jsonl` and can be passed to `run_eval.sh` as `MUTATION_OUTPUT`. `SC-align/scripts/filter_uncontrolled_specs.py` applies the control filter (every task keeps at least one valid alternative) to a file merged from several runs.

## Terminology

The code uses older names for several concepts of the paper:

| Code | Paper |
|---|---|
| certificate, cert | validation harness |
| `build_cases()`, case | scenario generator, test scenario |
| `check_case(case)` | executable checks |
| mutant | violating implementation |
| control, control mutant | valid alternative implementation |
| gold | gold implementation |
| fired, silent | rejects, accepts |
| sound | faithful |
| gold firing, cross firing | false alarm on the gold, on an alternative implementation |
| witness | differential witness |
| reference state | reference implementation shown to the agent |

## License

The `featurebench/` directory is derived from FeatureBench and SWE-bench and is distributed under their MIT license (`LICENSE`).
