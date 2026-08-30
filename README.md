# Will the User Ever Know? Covert Indirect Prompt Injection Attacks on Tool-Using LLM Agents

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-b31b1b)](https://yslmoment.github.io/ICoA/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.30362-b31b1b)](https://arxiv.org/abs/2608.30362)
[![Project Page](https://img.shields.io/badge/Project%20Page-ICoA-1b4f8f)](https://yslmoment.github.io/ICoA/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Yunseok Lee<sup>\*</sup>, Yunji Kim<sup>\*</sup>, Woojin Lee<sup>†</sup>

Dongguk University-Seoul &nbsp;·&nbsp; `{yslee0005, 2022113147, wj926}@dgu.ac.kr`

<sub><sup>\*</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding author.</sub>

Official implementation of **ICoA (Induced Covert Attack)**, accepted to **EMNLP 2026 (Main Conference)**.

---

## Overview

![Overt vs. covert outcomes](docs/teaser.png)

Attack Success Rate counts whether an injection succeeded. It says nothing about what the
user sees afterwards. Looking at successful injections, two outcomes appear: the agent
executes the injected task and then reports it, or it executes the injected task and
returns an ordinary-looking answer that never mentions it.

We call these **overt** and **covert** successes, and decompose ASR into the **Covert
Success Rate (CSR)** and the **Overt Success Rate (OSR)**. What separates them is where
the trajectory ends — covert traces hand control back to the user task before finishing,
overt traces stop at the attack. ICoA exploits this by steering the agent back to the
user's task after the injection fires.

## Highlights

- **A metric that measures what the user notices.** CSR and OSR split ASR by whether the
  agent's final reply reveals the injected action. An `auditor/` LLM judge assigns the label.
- **An attack built for covertness, not just success.** ICoA reaches the highest CSR across
  all four target models, gaining 3.79 to 12.01 points over the strongest existing baseline.
- **The RETURN anchor as a portable add-on.** Appending the anchor to any baseline attack
  raises its CSR, isolating the mechanism from the rest of the payload (`_anchored` variants).
- **Six defense settings, including a task-alignment defense.** Every attack is evaluated
  with no defense, four prompting and detection defenses, and Task Shield — under which
  ICoA still holds the highest CSR.

## Released artifacts

| Artifact | Path | What it is |
| --- | --- | --- |
| ICoA attack | `src/agentdojo/attacks/icoa.py` | The attack itself |
| RETURN anchor | `src/agentdojo/attacks/anchor.py` | Add-on producing the `_anchored` baselines |
| Baseline attacks | `src/agentdojo/attacks/baseline_attacks.py` | Direct, InjecAgent, ChatInject (3 model-specific variants) |
| Task Shield | `src/agentdojo/agent_pipeline/pi_detector.py` | Task-alignment defense, self-judged per target |
| LLM auditor | `auditor/judge.py` | Covert/overt classification of ASR-positive traces |
| Batch auditor | `auditor/judge_batch.py` | Same judgment through the OpenAI Batch API |
| Sweep scripts | `scripts/` | Full paper matrix and the Task Shield runners |
| Name map | `docs/attack_name_map.md` | Code names ↔ paper names |

## Quick start

Python 3.10+ is required. Install into a clean environment — this is a vendored fork of
AgentDojo and keeps the upstream package name.

```bash
pip install -e .
pip install -e ".[transformers]"   # only for the PI Detector defense
```

Set the keys for whichever targets you run, as environment variables or in a `.env` at the
repository root:

```
OPENAI_API_KEY=...      # GPT-4o-mini target, and the default auditor judge
GOOGLE_API_KEY=...      # Gemini-2.5-Flash target
```

LLaMA-3.3-70B and Qwen3-235B are served locally through [Ollama](https://ollama.com/).

Run one benchmark cell:

```bash
python -m agentdojo.scripts.benchmark \
    --model OPENAI_GPT_4O_MINI_20240718 \
    --attack icoa --suite banking --logdir runs/
```

Local models take `LOCAL` plus a `--model-id`, and `--defense` selects a defense. The five
defenses reported in the paper are `transformers_pi_detector`, `instructional_prevention`,
`spotlighting_with_delimiting`, `repeat_user_prompt` and `task_shield`:

```bash
python -m agentdojo.scripts.benchmark \
    --model LOCAL --model-id llama3.3:70b \
    --attack icoa --defense task_shield \
    --suite banking --logdir runs/
```

## Reproducing the paper

**1. Sweep the benchmark.**

```bash
bash scripts/run_attacks.sh
```

Results land in `runs/`. The full matrix is 9 attacks × 6 defenses × 4 suites × 4 models,
about 864 benchmark runs — substantial wall time and API cost. Every dimension is
overridable by environment variable; see the script header.

**2. Run the Task Shield rows.** Task Shield judges with a chat model rather than a local
classifier, so it has its own runners:

```bash
# local models via Ollama
MODEL_ID=llama3.3:70b LOCAL_LLM_PORT=11452 \
ATTACKS="direct injecagent important_instructions chat_inject_llama33 icoa" \
DEFENSES=task_shield SUITES="banking slack travel workspace" \
USER_TASKS=all INJECTION_TASKS=all LOGDIR=runs/llama33_task_shield \
bash scripts/run_llama33_70b_strong_defenses.sh

# GPT-4o-mini via the OpenAI API
bash scripts/run_gpt4omini_task_shield_5attacks.sh
```

**3. Audit the traces.** The auditor labels each ASR-positive trace covert or overt, which
is what turns ASR into CSR and OSR:

```bash
bash scripts/run_auditor.sh            # synchronous, judges with GPT-4o
```

For a full sweep, the batch auditor does the same judgment through the OpenAI Batch API at
roughly half the cost:

```bash
python auditor/judge_batch.py prepare   --run icoa=runs/llama33_task_shield --out-dir runs/audit_batch
python auditor/judge_batch.py submit    --out-dir runs/audit_batch
python auditor/judge_batch.py status    --out-dir runs/audit_batch
python auditor/judge_batch.py download  --out-dir runs/audit_batch
python auditor/judge_batch.py summarize --out-dir runs/audit_batch --out runs/audit_results.json
```

Both auditors judge only ASR-positive traces, and both abort on an API error rather than
scoring a failed call as covert.

## Repository structure

```
ICoA/
├── src/agentdojo/                  # vendored fork of AgentDojo 0.1.34
│   ├── attacks/                    # icoa.py, anchor.py, baselines, registry
│   ├── agent_pipeline/             # pipeline elements and 6 registered defenses
│   ├── default_suites/             # AgentDojo task suites (v1 … v1_2_1)
│   └── scripts/benchmark.py        # benchmark entry point
├── auditor/
│   ├── judge.py                    # synchronous covert/overt judge
│   ├── judge_batch.py              # Batch API judge (prepare/submit/status/download/summarize)
│   └── prompts/judge_prompt.txt    # judge prompt
├── scripts/
│   ├── run_attacks.sh              # full paper sweep
│   ├── run_auditor.sh              # auditor sweep
│   ├── run_llama33_70b_strong_defenses.sh
│   └── run_gpt4omini_task_shield_5attacks.sh
├── docs/
│   ├── index.html                  # project page
│   ├── paper.pdf                   # camera-ready
│   ├── teaser.png
│   └── attack_name_map.md          # code names ↔ paper names
├── tests/test_release_port.py      # release guards
├── pyproject.toml
└── LICENSE
```

## Reproduction notes

These describe how the paper's numbers were actually produced. They are not bugs, and they
are not fixed here — changing any of them would move a published number.

- **ChatInject is model-specific, and `run_attacks.sh` cannot express that.** The script
  sweeps one `ATTACKS` list across every model, but ChatInject has three registered
  variants, each targeting one model's chat template: `chat_inject_llama33` (LLaMA-3.3-70B),
  `chat_inject_qwen3` (Qwen3-235B), `chat_inject_gpt_oss` (GPT-4o-mini). Gemini-2.5-Flash
  has no ChatInject variant and no ChatInject row in the paper. Reproduce these cells by
  giving each model its own `ATTACKS=` override.

- **The two auditors reconstruct the judge's input from different places.** `judge.py` uses
  the task suite's canonical `GOAL` and `PROMPT` strings; `judge_batch.py` reads the
  rendered injection payload and the first user message out of the trace record. Both are
  legitimate, but they are not guaranteed to produce identical judge inputs case for case,
  so do not expect the two to agree exactly on the same traces.

- **Task Shield fails open.** If a judge call raises, or its answer will not parse, the tool
  call is permitted rather than blocked (`ContributeTo` defaults to `1.0`). This is faithful
  to the defense as evaluated.

- **Task Shield judges every target with that target's own model.** Google models need one
  extra step: a genai client exposes `.chats` rather than the `.chat` surface the judge
  calls, so the pipeline hands Task Shield a small adapter
  (`GenaiChatCompletionsAdapter` in `agent_pipeline/llms/google_llm.py`). Without it the
  defense would fall back to an OpenAI judge, and a Google target would be the only row
  judged by a different model than the agent it defends.

- **Both auditors take the first `local*` subdirectory they find in a run directory.** If
  one log directory holds more than one defense's output, only the alphabetically first is
  audited and the rest is silently skipped. Use one log directory per defense.

- **`pyproject.toml` carries the upstream package identity.** `name = "agentdojo"`,
  `version = "0.1.34"` describe the release this is forked from, while the fork's behavior
  differs. Install in a clean environment rather than alongside stock AgentDojo.

## Citation

```bibtex
@inproceedings{lee2026icoa,
  title     = {Will the User Ever Know? Covert Indirect Prompt Injection Attacks on Tool-Using LLM Agents},
  author    = {Lee, Yunseok and Kim, Yunji and Lee, Woojin},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  eprint    = {2608.30362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## Acknowledgements

This release is a vendored fork of **AgentDojo 0.1.34**, developed by Edoardo Debenedetti,
Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr at ETH
Zurich SPY Lab ([homepage](https://agentdojo.spylab.ai) · [repository](https://github.com/ethz-spylab/agentdojo)).
AgentDojo is released under the MIT License; see `LICENSE`. The ICoA attack, the RETURN
anchor, the Task Shield port and the auditors are original contributions of this paper.
