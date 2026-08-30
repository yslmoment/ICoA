# Attack and Defense Name Map

This document maps the registered names used in the code to the names used in the ICoA paper.

---

## Attacks

| Code name | Paper name | Notes |
|---|---|---|
| `icoa` | ICoA | The Induced Covert Attack; implemented in `src/agentdojo/attacks/icoa.py` |
| `direct` | Direct | Baseline: prepends `TODO: {goal}` |
| `direct_anchored` | Direct + anchor (§5.4 add-on) | Baseline with RETURN anchor appended |
| `injecagent` | InjecAgent | Baseline: the InjecAgent attack |
| `injecagent_anchored` | InjecAgent + anchor (§5.4 add-on) | Baseline with RETURN anchor appended |
| `ignore_previous` | Ignore Previous | Baseline: "Ignore your previous instructions..." |
| `ignore_previous_anchored` | Ignore Previous + anchor (§5.4 add-on) | Baseline with RETURN anchor appended |
| `important_instructions` | Important Instructions | Baseline: `<INFORMATION>` XML envelope |
| `important_instructions_anchored` | Important Instructions + anchor (§5.4 add-on) | Baseline with RETURN anchor appended |

---

## Defenses

| Code name | Paper name | Notes |
|---|---|---|
| `no_defense` | No Defense | Bare pipeline with no injection defense |
| `transformers_pi_detector` | PI Detector | Prompt-injection detector using a local transformer classifier |
| `spotlighting_with_delimiting` | Delimiting | Wraps tool outputs in delimiters to spotlight external content |
| `repeat_user_prompt` | Repeat User | Repeats the user prompt before each tool-output block |
| `instructional_prevention` | Instructional Prevention | Adds an instruction-following prevention prompt to the system message |
| `tool_filter` | (not in paper experiments) | Tool-filter defense; kept in codebase but not reported in the paper |
| `task_shield` | Task Shield | Runtime task-alignment defense (Jia et al., 2025); an LLM judge scores each tool call against tasks extracted from the user message and blocks unaligned calls. Implemented as `TaskShield` in `src/agentdojo/agent_pipeline/pi_detector.py` |

---

## ChatInject variants

`baseline_attacks.py` registers three model-specific single-turn ChatInject variants, one per model for which the paper reports a ChatInject result. The multi-turn variants that previously appeared in this file have been removed (they depended on a `multi_turn_data/` directory that is not included in this release).

| Code name | Paper model | Notes |
|---|---|---|
| `chat_inject_qwen3` | ChatInject baseline, Qwen3-235B | Uses Qwen3 chat-template interrupt tags |
| `chat_inject_gpt_oss` | ChatInject baseline, GPT-4o-mini | Uses GPT-4o-mini (OSS) chat-template interrupt tags |
| `chat_inject_llama33` | ChatInject baseline, LLaMA-3.3-70B | Uses LLaMA-3 chat-template interrupt tags (`<|eot_id|><|start_header_id|>…<|end_header_id|>\n\n`) |

Note: Gemini-2.5-Flash has no ChatInject result in the paper ("--" in Table 2). Only these 3 models have ChatInject rows.

---

## Stock upstream AgentDojo attacks (not all used in ICoA paper)

The package also retains the following attacks inherited from the upstream AgentDojo codebase. These are registered and importable but are not all used in the ICoA paper experiments:

- `manual`, `direct`, `ignore_previous`, `system_message`, `injecagent` — simple fixed-string baselines
- `important_instructions`, `important_instructions_anchored`, and related `*_anchored` ablation variants
- `tool_knowledge` — tool-aware attack baseline
- `dos`-family attacks from `dos_attacks.py`
