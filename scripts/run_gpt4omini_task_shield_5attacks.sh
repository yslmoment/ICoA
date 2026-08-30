#!/bin/bash
# gpt-4o-mini × task_shield × 5 attacks × 4 suites = 20 cells.
# Attacks (paper table order): Direct, InjecAgent, Imp. message, ChatInject, ICoA.
#
# TaskShield (pi_detector.py Step 4) edits messages in place on detection,
# rather than appending a synthetic {"role":"tool", ...} message with no
# matching tool_calls. That matters for OpenAI-served models: an orphan tool
# message would be rejected with 400 "messages with role 'tool' must be a
# response to a preceeding message with 'tool_calls'". Local models never hit
# this because PromptingLLM flattens messages to text.
#
# Task Shield's internal judge falls back to the agent LLM (pi_detector.py:243),
# so judge = gpt-4o-mini, matching the self-judge protocol used for this model.
#
# CSR/OSR judging is a separate post-hoc step and is NOT done here; see
# auditor/judge.py or auditor/judge_batch.py, both of which default to gpt-4o.
#
# PARALLELISM: 4 suites run as concurrent processes, each on benchmark.py's default
# max_workers=1 path. Do NOT use --max-workers > 1: its Pool branch passes suite name
# strings where benchmark_suite expects suite objects and crashes immediately.
#
# No --force-rerun: completed cells are skipped, so this is safe to re-run to resume.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# API keys come from a .env at the repo root, if present.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL="OPENAI_GPT_4O_MINI_20240718"
DEFENSE="task_shield"
BENCHMARK_VERSION="v1.2.1"
SUITES=(banking slack travel workspace)
LOGDIR="${REPO_ROOT}/runs_full/gpt4omini_fullsuite_5attacks_taskshield"
LOGS="${LOGDIR}/logs"

[ -z "${OPENAI_API_KEY:-}" ] && { echo "ERROR: OPENAI_API_KEY not set"; exit 1; }
mkdir -p "$LOGS"

ATTACKS=(
    direct
    injecagent
    important_instructions
    chat_inject_gpt_oss
    icoa
)

echo "[$(date)] Start: $MODEL × $DEFENSE × ${#ATTACKS[@]} attacks × ${#SUITES[@]} suites (suites in parallel)"
echo "[$(date)] Logdir: $LOGDIR"

failed=()

# ── Phase 1: run all cells, 4 suites concurrently per attack ──────────────────
for ATTACK in "${ATTACKS[@]}"; do
    echo ""
    echo "[$(date)] ═════════ ATTACK: ${ATTACK} ═════════"

    pids=()
    for SUITE in "${SUITES[@]}"; do
        $PYTHON -m agentdojo.scripts.benchmark \
            --model "$MODEL" \
            --benchmark-version "$BENCHMARK_VERSION" \
            --attack "$ATTACK" --defense "$DEFENSE" \
            --suite "$SUITE" --logdir "$LOGDIR" \
            > "${LOGS}/${ATTACK}_${SUITE}.log" 2>&1 &
        pids+=($!)
        echo "[$(date)] ▶ launched ${ATTACK} × ${SUITE} (pid ${pids[-1]})"
    done

    for i in "${!SUITES[@]}"; do
        SUITE="${SUITES[$i]}"
        if wait "${pids[$i]}"; then
            echo "[$(date)] DONE_MARKER ${ATTACK}_task_shield_${SUITE}"
        else
            rc=$?
            echo "[$(date)] FAILED_MARKER ${ATTACK}_task_shield_${SUITE} rc=$rc"
            failed+=("${ATTACK}/${SUITE}")
        fi
    done
    echo "[$(date)] ✓ ATTACK ${ATTACK} all suites finished"
done

# ── Phase 2: emit eval JSONs (cells are cached, so this makes no API calls) ────
# Matches the runs_full/ convention: eval/<attack>/<defense>/{<suite>,combined}.json
echo ""
echo "[$(date)] ═════════ Phase 2: eval JSONs ═════════"
for ATTACK in "${ATTACKS[@]}"; do
    EVAL_DIR="${LOGDIR}/eval/${ATTACK}/${DEFENSE}"
    mkdir -p "$EVAL_DIR"
    if $PYTHON -m agentdojo.scripts.benchmark \
        --model "$MODEL" \
        --benchmark-version "$BENCHMARK_VERSION" \
        --attack "$ATTACK" --defense "$DEFENSE" \
        -s banking -s slack -s travel -s workspace \
        --logdir "$LOGDIR" --eval-results-dir "$EVAL_DIR" \
        > "${LOGS}/${ATTACK}_eval.log" 2>&1; then
        echo "[$(date)] EVAL_MARKER ${ATTACK} → ${EVAL_DIR}/combined.json"
    else
        echo "[$(date)] EVAL_FAILED ${ATTACK} — see ${LOGS}/${ATTACK}_eval.log"
        failed+=("${ATTACK}/eval")
    fi
done

echo ""
echo "[$(date)] ═══════════════════════════════════════"
if [ ${#failed[@]} -eq 0 ]; then
    echo "[$(date)] ALL DONE — gpt-4o-mini × task_shield × 5 attacks × 4 suites"
else
    echo "[$(date)] DONE WITH FAILURES: ${failed[*]}"
    echo "[$(date)] Re-run this script to resume (completed cells are skipped)."
fi
