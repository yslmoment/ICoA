#!/usr/bin/env bash
# =============================================================================
# run_auditor.sh — ICoA LLM-judge covert/overt validation
#
# Runs auditor/judge.py over the runs directory produced by run_attacks.sh
# to generate per-case covert (CSR) and overt (OSR) judgments.
#
# PREREQUISITES
#   pip install -e .           (from <REPO_ROOT>)
#   OPENAI_API_KEY must be set — judge.py calls GPT-4o by default.
#
# USAGE
#   # Judge all runs in the default runs/ directory:
#   bash scripts/run_auditor.sh
#
#   # Judge a specific runs directory:
#   bash scripts/run_auditor.sh /path/to/custom/runs
#
#   # Set output path explicitly:
#   OUT=/path/to/results.json bash scripts/run_auditor.sh
#
#   # Use a different judge model:
#   JUDGE_MODEL=gpt-4o-mini bash scripts/run_auditor.sh
#
# OUTPUT
#   A single JSON file at <OUT> containing per-case records and a summary.
#   Default: <REPO_ROOT>/runs/audit_results.json
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Positional arg overrides the default runs directory
RUNS_DIR="${1:-${REPO_ROOT}/runs}"
OUT="${OUT:-${RUNS_DIR}/audit_results.json}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Build --run NAME=PATH arguments for every attack-cell subdirectory.
# judge.py expects each PATH to contain a 'local*' subdir (agentdojo output).
# ---------------------------------------------------------------------------
RUN_ARGS=()
for cell_dir in "${RUNS_DIR}"/*/; do
    [[ -d "${cell_dir}" ]] || continue
    # Skip the audit output directory itself if it already exists
    [[ "${cell_dir}" == "${RUNS_DIR}/audit_results"* ]] && continue
    name="$(basename "${cell_dir}")"
    RUN_ARGS+=(--run "${name}=${cell_dir}")
done

if [[ ${#RUN_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No run cell directories found under ${RUNS_DIR}" >&2
    echo "       Run 'bash scripts/run_attacks.sh' first." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"

echo "[$(date)] ICoA auditor"
echo "  Runs dir:    ${RUNS_DIR}"
echo "  Run cells:   ${#RUN_ARGS[@]} directories"
echo "  Judge model: ${JUDGE_MODEL}"
echo "  Output:      ${OUT}"
echo ""

python "${REPO_ROOT}/auditor/judge.py" \
    "${RUN_ARGS[@]}" \
    --judge-model "${JUDGE_MODEL}" \
    --out "${OUT}"

echo ""
echo "[$(date)] Audit complete — results at ${OUT}"
