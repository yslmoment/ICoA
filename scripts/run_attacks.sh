#!/usr/bin/env bash
# =============================================================================
# run_attacks.sh — ICoA benchmark reproduction script
#
# Runs the agentdojo benchmark over the attack × defense × suite × model
# matrix used in the ICoA paper. Results land in <REPO_ROOT>/runs/.
#
# PREREQUISITES
#   pip install -e .           (from <REPO_ROOT>)
#   Set API keys in environment or in a .env file at the repo root:
#     OPENAI_API_KEY     — GPT-4o-mini
#     GOOGLE_API_KEY     — Gemini-2.5-Flash
#
# LOCAL MODEL BACKENDS (LLaMA-3.3-70B and Qwen3-235B-mmap)
#   These two models were served via Ollama during the paper experiments and
#   require a running Ollama server before launching this script.
#
#   LLaMA-3.3-70B:
#     The paper experiments ran Ollama on port 11443 (GPU 3).
#     Ollama model tag used: llama3.3:70b
#     Start server:  CUDA_VISIBLE_DEVICES=3 LOCAL_LLM_PORT=11443 ollama serve
#     Pull model:    ollama pull llama3.3:70b
#
#   Qwen3-235B:
#     The paper experiments ran Ollama on port 11442 (GPU 2) with a
#     memory-mapped quantized variant registered under the local tag
#     qwen3-235b-mmap:latest.
#     Start server:  CUDA_VISIBLE_DEVICES=2 LOCAL_LLM_PORT=11442 ollama serve
#     The exact model weights and Ollama Modelfile are environment-specific;
#     if your Ollama install uses a different tag, set MODEL_ID_QWEN accordingly.
#
# --model flag note
#   benchmark.py's --model argument accepts the ModelsEnum member *name*
#   (e.g. OPENAI_GPT_4O_MINI_20240718, LOCAL), not the model-id string.
#   API-based models:  pass the enum name as --model <NAME>
#   Local Ollama models: pass LOCAL as --model, plus --model-id <ollama-tag>
#
# COST WARNING
#   A full sweep (9 attacks × 6 defenses × 4 suites × 4 models) is ~864
#   benchmark runs and will incur significant API cost. Expect hours of wall
#   time even with API-based models.
#
# USAGE
#   # Full matrix (all models, attacks, defenses, suites):
#   bash scripts/run_attacks.sh
#
#   # Override any dimension via environment variables (space-separated):
#   ATTACKS="icoa" DEFENSES="no_defense" SUITES="banking" \
#   bash scripts/run_attacks.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGDIR="${REPO_ROOT}/runs"

# ---------------------------------------------------------------------------
# Local-server configuration — adjust ports / tags to match your Ollama setup
# ---------------------------------------------------------------------------
LOCAL_LLM_PORT_LLAMA="${LOCAL_LLM_PORT_LLAMA:-11443}"   # Ollama port for LLaMA
LOCAL_LLM_PORT_QWEN="${LOCAL_LLM_PORT_QWEN:-11442}"     # Ollama port for Qwen

MODEL_ID_LLAMA="${MODEL_ID_LLAMA:-llama3.3:70b}"
MODEL_ID_QWEN="${MODEL_ID_QWEN:-qwen3-235b-mmap:latest}"

# ---------------------------------------------------------------------------
# Paper model definitions
#
# Each entry in MODEL_SPECS is a colon-separated record:
#   <slug>:<model_enum_name>[:<model_id>]
#
# slug         — used in log-directory names
# model_enum_name — passed as --model <name> to benchmark.py
# model_id     — (optional) passed as --model-id <id>; only for LOCAL models
#
# These are the EXACT models used in the ICoA paper:
#   • GPT-4o-mini      — OpenAI API, MODEL="OPENAI_GPT_4O_MINI_20240718"
#   • Gemini-2.5-Flash — Google AI Studio API, MODEL="GEMINI_2_5_FLASH"
#   • LLaMA-3.3-70B    — Ollama local server, MODEL="LOCAL",
#                         MODEL_ID="llama3.3:70b", port 11443
#   • Qwen3-235B       — Ollama local server, MODEL="LOCAL",
#                         MODEL_ID="qwen3-235b-mmap:latest", port 11442
# ---------------------------------------------------------------------------
DEFAULT_MODEL_SPECS=(
    "gpt4omini:OPENAI_GPT_4O_MINI_20240718"
    "gemini25flash:GEMINI_2_5_FLASH"
    # REQUIRES LOCAL OLLAMA SERVER — see header for setup instructions
    "llama33-70b:LOCAL:${MODEL_ID_LLAMA}"
    # REQUIRES LOCAL OLLAMA SERVER — see header for setup instructions
    "qwen3-235b:LOCAL:${MODEL_ID_QWEN}"
)

# ICoA + 4 baselines = 9 attacks total (anchored variants included)
DEFAULT_ATTACKS=(
    "icoa"
    "direct"
    "direct_anchored"
    "injecagent"
    "injecagent_anchored"
    "ignore_previous"
    "ignore_previous_anchored"
    "important_instructions"
    "important_instructions_anchored"
)

# no_defense + 5 defenses from the paper
DEFAULT_DEFENSES=(
    "no_defense"
    "spotlighting_with_delimiting"
    "transformers_pi_detector"
    "repeat_user_prompt"
    "instructional_prevention"
    "task_shield"
)

# All 4 AgentDojo suites
DEFAULT_SUITES=(
    "banking"
    "slack"
    "travel"
    "workspace"
)

# ---------------------------------------------------------------------------
# Allow environment-variable overrides (space-separated strings)
# MODEL_SPECS can be overridden, but the colon-separated format must be kept.
# ---------------------------------------------------------------------------
IFS=' ' read -r -a MODEL_SPECS <<< "${MODEL_SPECS:-${DEFAULT_MODEL_SPECS[*]}}"
IFS=' ' read -r -a ATTACKS    <<< "${ATTACKS:-${DEFAULT_ATTACKS[*]}}"
IFS=' ' read -r -a DEFENSES   <<< "${DEFENSES:-${DEFAULT_DEFENSES[*]}}"
IFS=' ' read -r -a SUITES     <<< "${SUITES:-${DEFAULT_SUITES[*]}}"

# ---------------------------------------------------------------------------
# Load .env if present (silently — benchmark.py also loads it, this just
# ensures API keys are available for any pre-flight checks you add here)
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

mkdir -p "${LOGDIR}"

echo "[$(date)] ICoA benchmark sweep"
echo "  Model specs: ${MODEL_SPECS[*]}"
echo "  Attacks:     ${ATTACKS[*]}"
echo "  Defenses:    ${DEFENSES[*]}"
echo "  Suites:      ${SUITES[*]}"
echo "  Output:      ${LOGDIR}"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SPEC in "${MODEL_SPECS[@]}"; do
    # Parse colon-separated spec: slug:model_enum_name[:model_id]
    IFS=':' read -r SLUG MODEL_ENUM MODEL_ID_ARG <<< "${SPEC}"

    # Build the --model [--model-id] fragment
    MODEL_FLAGS=(--model "${MODEL_ENUM}")
    if [[ -n "${MODEL_ID_ARG:-}" ]]; then
        MODEL_FLAGS+=(--model-id "${MODEL_ID_ARG}")
    fi

    for ATTACK in "${ATTACKS[@]}"; do
        for DEFENSE in "${DEFENSES[@]}"; do
            CELL_DIR="${LOGDIR}/${SLUG}--${ATTACK}--${DEFENSE}"
            mkdir -p "${CELL_DIR}"

            for SUITE in "${SUITES[@]}"; do
                echo "[$(date)] ▶ ${SLUG} × ${ATTACK} × ${DEFENSE} × ${SUITE}"

                DEFENSE_FLAG=()
                if [[ "${DEFENSE}" != "no_defense" ]]; then
                    DEFENSE_FLAG=(--defense "${DEFENSE}")
                fi

                python -m agentdojo.scripts.benchmark \
                    "${MODEL_FLAGS[@]}" \
                    --attack "${ATTACK}" \
                    "${DEFENSE_FLAG[@]}" \
                    --suite "${SUITE}" \
                    --logdir "${CELL_DIR}"

                echo "[$(date)] ✓ ${SLUG} × ${ATTACK} × ${DEFENSE} × ${SUITE}"
            done
        done
    done
done

echo ""
echo "[$(date)] ALL DONE — results in ${LOGDIR}"
