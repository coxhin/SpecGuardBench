#!/usr/bin/env bash
set -euo pipefail

LLM_A="claude-opus-5"
LLM_B="gpt-5.6-sol"
TAG="full_set"
INSTANCES_FILE="data/instances.jsonl"
INSTANCE_IDS=()
CONTAINER_POOL=8
STEP2_MAX_ATTEMPTS=3
PRESERVATION_PROBE_K=3
STEP6_VOTES=3
STEP6B_CONCURRENCY=12
TIMEOUT=600
HYPOTHESIS_TIMEOUT=120
SPLIT="full"
CONFIG_PATH="config_mutation.toml"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -n "$INSTANCES_FILE" ] && [ ${#INSTANCE_IDS[@]} -eq 0 ]; then
    mapfile -t INSTANCE_IDS < <(
        .venv/bin/python -c 'import json, sys; [print(json.loads(l)["instance_id"]) for l in open(sys.argv[1]) if l.strip()]' "$INSTANCES_FILE"
    )
fi

CMD=(
    .venv/bin/python -m mutation_pipeline.run
    --config-path "$CONFIG_PATH"
    --llm-a "$LLM_A"
    --llm-b "$LLM_B"
    --tag "$TAG"
    --split "$SPLIT"
    --container-pool "$CONTAINER_POOL"
    --step2-max-attempts "$STEP2_MAX_ATTEMPTS"
    --preservation-probe-k "$PRESERVATION_PROBE_K"
    --step6-votes "$STEP6_VOTES"
    --step6b-concurrency "$STEP6B_CONCURRENCY"
    --timeout "$TIMEOUT"
    --hypothesis-timeout "$HYPOTHESIS_TIMEOUT"
)

if [ ${#INSTANCE_IDS[@]} -gt 0 ]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi

CMD+=("$@")

export PYTHONPATH=".:SC-align"

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

exec "${CMD[@]}"
