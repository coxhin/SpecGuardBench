#!/usr/bin/env bash
set -euo pipefail

BASELINE="codex"
MODEL=""
TAG="full_set"
MUTATION_OUTPUT="data/full_set.jsonl"
HARNESS_FORM="factory_oracle"
N_CONCURRENT=12
MAX_AGENT_STEPS="80"
TIMEOUT=""
STAGE2_REFERENCE_STATE=""
EGRESS="api_only"
INSTANCE_IDS=()
STAGES=()
SPEC_SUBSET=""
CONFIG_PATH="config_eval.toml"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CMD=(
    .venv/bin/python -m eval_pipeline.run
    --config-path "$CONFIG_PATH"
    --baseline "$BASELINE"
    --tag "$TAG"
    --mutation-output "$MUTATION_OUTPUT"
    --harness-form "$HARNESS_FORM"
    --n-concurrent "$N_CONCURRENT"
)

if [ -n "$MODEL" ]; then
    CMD+=(--model "$MODEL")
fi
if [ -n "$TIMEOUT" ]; then
    CMD+=(--timeout "$TIMEOUT")
fi
if [ -n "$MAX_AGENT_STEPS" ]; then
    CMD+=(--max-agent-steps "$MAX_AGENT_STEPS")
fi
if [ -n "$STAGE2_REFERENCE_STATE" ]; then
    CMD+=(--stage2-reference-state "$STAGE2_REFERENCE_STATE")
fi
if [ -n "$EGRESS" ]; then
    CMD+=(--egress "$EGRESS")
fi
if [ -n "$SPEC_SUBSET" ]; then
    CMD+=(--spec-subset "$SPEC_SUBSET")
fi
if [ ${#INSTANCE_IDS[@]} -gt 0 ]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi
if [ ${#STAGES[@]} -gt 0 ]; then
    CMD+=(--stages "${STAGES[@]}")
fi

CMD+=("$@")

export PYTHONPATH=".:SC-align"

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

exec "${CMD[@]}"
