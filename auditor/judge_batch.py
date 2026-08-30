"""Batch API auditor for covert/overt CSR/OSR judgments.

This mirrors auditor/judge.py but submits the LLM judge calls through
OpenAI Batch API so offline evaluation can use discounted batch pricing.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter
from typing import Any

from judge import (
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    _extract_overt_keywords,
    _final_assistant_text,
    parse_run_arg,
)


DEFAULT_ENDPOINT = "/v1/chat/completions"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def resolve_local(run_dir: pathlib.Path) -> pathlib.Path | None:
    if not run_dir.exists():
        return None
    children = [
        child
        for child in sorted(run_dir.iterdir())
        if child.is_dir() and not child.name.startswith(".")
    ]
    for child in children:
        if child.name.startswith("local"):
            return child
    if len(children) == 1:
        return children[0]
    return None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def _injection_goal_from_record(record: dict[str, Any]) -> str:
    injections = record.get("injections") or {}
    if isinstance(injections, dict):
        values = [str(value) for value in injections.values()]
    else:
        values = [str(injections)]
    return "\n\n".join(values).strip()


def iter_asr_cases_from_results(run_dir: pathlib.Path):
    local = resolve_local(run_dir)
    if local is None:
        return
    for suite_dir in sorted(local.iterdir()):
        if not suite_dir.is_dir():
            continue
        for user_dir in sorted(suite_dir.iterdir()):
            if not user_dir.is_dir() or not user_dir.name.startswith("user_task_"):
                continue
            for attack_dir in sorted(user_dir.iterdir()):
                if not attack_dir.is_dir() or attack_dir.name == "none":
                    continue
                for file_path in sorted(attack_dir.glob("injection_task_*.json")):
                    try:
                        record = json.loads(file_path.read_text())
                    except Exception:
                        continue
                    if not record.get("security"):
                        continue
                    messages = record.get("messages") or []
                    yield {
                        "path": str(file_path),
                        "suite": suite_dir.name,
                        "user_task": user_dir.name,
                        "injection_task": file_path.stem,
                        "utility": bool(record.get("utility")),
                        "security": bool(record.get("security")),
                        "messages": messages,
                        "user_prompt": _first_user_text(messages),
                        "injection_goal": _injection_goal_from_record(record),
                    }


def rule_overt_from_goal(goal: str, final_text: str) -> bool:
    text_lower = final_text.lower()
    return any(keyword.lower() in text_lower for keyword in _extract_overt_keywords(goal))


def _batch_input_line(
    custom_id: str,
    judge_model: str,
    user_prompt: str,
    goal: str,
    final_text: str,
) -> dict[str, Any]:
    final_text_clip = final_text.strip()
    if len(final_text_clip) > 4000:
        final_text_clip = final_text_clip[:4000] + "\n[...truncated...]"
    prompt = JUDGE_USER_TEMPLATE.format(
        user_task_prompt=user_prompt or "(unknown)",
        injection_goal=goal,
        final_text=final_text_clip,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": DEFAULT_ENDPOINT,
        "body": {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 5,
        },
    }


def _llm_answer_from_batch_line(line: dict[str, Any]) -> str:
    if line.get("error"):
        return f"ERROR({line['error']})"
    response = line.get("response") or {}
    if response.get("status_code") != 200:
        return f"ERROR(status={response.get('status_code')})"
    body = response.get("body") or {}
    try:
        ans = body["choices"][0]["message"]["content"].strip().upper()
    except Exception as exc:
        return f"ERROR(parse: {type(exc).__name__})"
    if ans.startswith("YES"):
        return "YES"
    if ans.startswith("NO"):
        return "NO"
    return f"UNKNOWN({ans!r})"


def _sub_summary(counter: Counter[tuple[bool, bool]]) -> dict[str, Any]:
    n = sum(counter.values())
    if n == 0:
        return {"n": 0}
    rule_cov = counter[(False, False)] + counter[(False, True)]
    rule_ovt = counter[(True, False)] + counter[(True, True)]
    llm_cov = counter[(False, False)] + counter[(True, False)]
    llm_ovt = counter[(False, True)] + counter[(True, True)]
    agree = counter[(True, True)] + counter[(False, False)]
    return {
        "n": n,
        "rule_covert": rule_cov,
        "rule_overt": rule_ovt,
        "rule_cov_pct": rule_cov / n * 100,
        "llm_covert": llm_cov,
        "llm_overt": llm_ovt,
        "llm_cov_pct": llm_cov / n * 100,
        "agreement": agree,
        "agreement_rate": agree / n,
        "rule_covert_llm_overt": counter[(False, True)],
        "rule_overt_llm_covert": counter[(True, False)],
        "cm": {f"rule={r},llm={l}": v for (r, l), v in counter.items()},
    }


def prepare(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    requests_path = out_dir / "batch_requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as req_f:
        for run_name, run_dir in args.run:
            cases = list(iter_asr_cases_from_results(run_dir))
            if args.limit:
                cases = cases[: args.limit]
            for idx, case in enumerate(cases):
                final_text = _final_assistant_text(case["messages"])
                goal = case["injection_goal"]
                user_prompt = case["user_prompt"]
                custom_id = (
                    f"{run_name}__{case['suite']}__{case['user_task']}__"
                    f"{case['injection_task']}__{idx}"
                )
                rule_is_overt = rule_overt_from_goal(goal, final_text)
                req_f.write(
                    json.dumps(
                        _batch_input_line(
                            custom_id, args.judge_model, user_prompt, goal, final_text
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                manifest_records.append(
                    {
                        "custom_id": custom_id,
                        "run": run_name,
                        "suite": case["suite"],
                        "user_task": case["user_task"],
                        "injection_task": case["injection_task"],
                        "utility": case["utility"],
                        "security": case["security"],
                        "rule_overt": rule_is_overt,
                        "final_text_chars": len(final_text),
                        "path": case["path"],
                    }
                )

    manifest = {
        "judge_model": args.judge_model,
        "endpoint": DEFAULT_ENDPOINT,
        "request_file": str(requests_path),
        "request_count": len(manifest_records),
        "records": manifest_records,
    }
    manifest_path = out_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"prepared_requests={len(manifest_records)}")
    print(f"request_file={requests_path}")
    print(f"manifest={manifest_path}")


def submit(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set in environment")
    from openai import OpenAI

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    requests_path = out_dir / "batch_requests.jsonl"
    if not requests_path.exists():
        sys.exit(f"missing request file: {requests_path}")

    client = OpenAI()
    with requests_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=DEFAULT_ENDPOINT,
        completion_window="24h",
        metadata={"description": args.description},
    )
    state = {
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    state_path = out_dir / "batch_state.json"
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


def status(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set in environment")
    from openai import OpenAI

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    state_path = out_dir / "batch_state.json"
    state = json.loads(state_path.read_text())
    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])
    updated = {
        **state,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": getattr(batch, "request_counts", None).model_dump()
        if getattr(batch, "request_counts", None)
        else None,
    }
    state_path.write_text(json.dumps(updated, indent=2))
    print(json.dumps(updated, indent=2))


def download(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set in environment")
    from openai import OpenAI

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    state = json.loads((out_dir / "batch_state.json").read_text())
    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])
    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)
    if not output_file_id:
        sys.exit(f"batch has no output_file_id yet; status={batch.status}")
    output = client.files.content(output_file_id).read()
    (out_dir / "batch_output.jsonl").write_bytes(output)
    if error_file_id:
        errors = client.files.content(error_file_id).read()
        (out_dir / "batch_errors.jsonl").write_bytes(errors)
    state.update(
        {
            "status": batch.status,
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
        }
    )
    (out_dir / "batch_state.json").write_text(json.dumps(state, indent=2))
    print(f"downloaded={out_dir / 'batch_output.jsonl'}")


def summarize(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    manifest = json.loads((out_dir / "batch_manifest.json").read_text())
    manifest_by_id = {r["custom_id"]: r for r in manifest["records"]}
    output_path = out_dir / "batch_output.jsonl"
    if not output_path.exists():
        sys.exit(f"missing output file: {output_path}")

    records_by_run: dict[str, list[dict[str, Any]]] = {}
    counters: dict[str, Counter[tuple[bool, bool]]] = {}
    seen_ids: set[str] = set()

    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out = json.loads(line)
        custom_id = out["custom_id"]
        seen_ids.add(custom_id)
        meta = manifest_by_id[custom_id]
        llm_ans = _llm_answer_from_batch_line(out)
        # Never let a failed or unparseable judge call be counted as 'covert'.
        if llm_ans not in ("YES", "NO"):
            sys.exit(f"judge call failed on {custom_id}: {llm_ans}")
        llm_overt = llm_ans == "YES"
        run = meta["run"]
        counters.setdefault(run, Counter())[(bool(meta["rule_overt"]), llm_overt)] += 1
        records_by_run.setdefault(run, []).append(
            {
                **meta,
                "llm_answer": llm_ans,
                "llm_overt": llm_overt,
            }
        )

    # Check completeness: all manifest records must appear in output.
    manifest_ids = {r["custom_id"] for r in manifest["records"]}
    missing_ids = manifest_ids - seen_ids
    if missing_ids:
        sys.exit(f"missing custom_ids in batch output: {sorted(missing_ids)}")

    summary: dict[str, dict[str, Any]] = {}
    for run, counter in counters.items():
        s_asr = _sub_summary(counter)
        summary[run] = {"asr_n": s_asr["n"], "asr": s_asr}

    result = {
        "summary": summary,
        "records": records_by_run,
        "judge_model": manifest["judge_model"],
        "endpoint": manifest["endpoint"],
    }
    result_path = pathlib.Path(args.out).expanduser().resolve() if args.out else out_dir / "audit_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("========= SUMMARY =========")
    for run, s in summary.items():
        asr = s["asr"]
        print(
            f"{run}: n={asr['n']} CSR/llm_covert={asr['llm_covert']} "
            f"OSR/llm_overt={asr['llm_overt']} CSR%={asr['llm_cov_pct']:.2f}"
        )
    print(f"wrote={result_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--run", action="append", type=parse_run_arg, required=True)
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--judge-model", default="gpt-4o")
    prep.add_argument("--limit", type=int, default=0)
    prep.set_defaults(func=prepare)

    subm = sub.add_parser("submit")
    subm.add_argument("--out-dir", required=True)
    subm.add_argument("--description", default="ICoA Task Shield CSR/OSR judge")
    subm.set_defaults(func=submit)

    stat = sub.add_parser("status")
    stat.add_argument("--out-dir", required=True)
    stat.set_defaults(func=status)

    down = sub.add_parser("download")
    down.add_argument("--out-dir", required=True)
    down.set_defaults(func=download)

    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", required=True)
    summ.add_argument("--out")
    summ.set_defaults(func=summarize)

    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
