#!/usr/bin/env bash
# Run a focused ICoA prelim with LLaMA-3.3-70B against the stronger defenses.
#
# Default mode repeats the focused rebuttal-prep cells:
#   banking / user_task_0 / injection_task_0
#   icoa + important_instructions + chat_inject_llama33
#   task_shield
#
# To expand to a full suite, set USER_TASKS=all and INJECTION_TASKS=all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-llama3.3:70b}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-11451}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/runs_prelim/llama33_70b_strong_defenses}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"

ATTACKS="${ATTACKS:-${ATTACK:-icoa important_instructions chat_inject_llama33}}"
DEFENSES="${DEFENSES:-task_shield}"
SUITES="${SUITES:-banking}"
USER_TASKS="${USER_TASKS-user_task_0}"
INJECTION_TASKS="${INJECTION_TASKS-injection_task_0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LOCAL_LLM_PORT

add_repeated_flags() {
    local opt="$1"
    local values="$2"
    local -n out_ref="$3"

    if [[ -z "${values}" || "${values}" == "all" || "${values}" == "ALL" ]]; then
        return
    fi

    local items=()
    IFS=' ' read -r -a items <<< "${values}"
    for item in "${items[@]}"; do
        out_ref+=("${opt}" "${item}")
    done
}

echo "[$(date)] ICoA LLaMA-3.3-70B strong-defense prelim"
echo "  Repo:       ${REPO_ROOT}"
echo "  Python:     ${PYTHON_BIN}"
echo "  Ollama:     localhost:${LOCAL_LLM_PORT}"
echo "  Model:      ${MODEL_ID}"
echo "  Attacks:    ${ATTACKS}"
echo "  Defenses:   ${DEFENSES}"
echo "  Suites:     ${SUITES}"
echo "  User tasks: ${USER_TASKS:-all}"
echo "  Inj tasks:  ${INJECTION_TASKS:-all}"
echo "  Logdir:     ${LOGDIR}"
echo ""

mkdir -p "${LOGDIR}"

IFS=' ' read -r -a suite_items <<< "${SUITES}"
IFS=' ' read -r -a defense_items <<< "${DEFENSES}"
IFS=' ' read -r -a attack_items <<< "${ATTACKS}"

for attack in "${attack_items[@]}"; do
    for defense in "${defense_items[@]}"; do
        flags=(
            -m agentdojo.scripts.benchmark
            --model LOCAL
            --model-id "${MODEL_ID}"
            --benchmark-version "${BENCHMARK_VERSION}"
            --attack "${attack}"
            --defense "${defense}"
            --logdir "${LOGDIR}"
            --eval-results-dir "${LOGDIR}/eval/${attack}/${defense}"
        )

        for suite in "${suite_items[@]}"; do
            flags+=(--suite "${suite}")
        done

        add_repeated_flags "--user-task" "${USER_TASKS}" flags
        add_repeated_flags "--injection-task" "${INJECTION_TASKS}" flags

        if [[ "${FORCE_RERUN}" == "1" ]]; then
            flags+=(--force-rerun)
        fi

        echo "[$(date)] Running attack=${attack} defense=${defense}"
        "${PYTHON_BIN}" "${flags[@]}"
        echo "[$(date)] Finished attack=${attack} defense=${defense}"
        echo ""
    done
done

echo "[$(date)] Done. Logs are under ${LOGDIR}"
