"""Guards on what the ICoA release ships.

Run: PYTHONPATH=src python tests/test_release_port.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_ipiguard():
    """IPIGuard is out of scope for this release and must not ship here."""
    hits = subprocess.run(
        ["grep", "-ril", "ipiguard", "."],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
    hits = [h for h in hits if not h.startswith("./.git/") and h != "./tests/test_release_port.py"]
    assert hits == [], f"IPIGuard leaked into the release: {hits}"


def test_no_run_data_or_secrets():
    """Experiment traces and credentials must not ship."""
    bad = [p for p in ROOT.rglob("*")
           if ".git/" not in str(p)
           and (p.name.startswith("runs_") or p.name == ".env"
                or "openai_key" in p.name)]
    assert bad == [], f"must not be committed: {bad}"


def test_task_shield_importable():
    from agentdojo.agent_pipeline.pi_detector import TaskShield
    assert TaskShield.__name__ == "TaskShield"


def test_task_shield_registered():
    from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
    assert "task_shield" in DEFENSES


def test_auditor_has_error_guard():
    """A failed judge call must abort, never be scored as covert."""
    src = (ROOT / "auditor" / "judge.py").read_text()
    assert 'llm_ans.startswith("ERROR(")' in src


def test_batch_auditor_present():
    assert (ROOT / "auditor" / "judge_batch.py").is_file()


def test_task_shield_self_judges_on_google():
    """Every target must be judged by its own model, Google included.

    A genai client has `.chats`, not `.chat`, so without the adapter TaskShield
    falls back to an OpenAI judge and the Google row alone is judged by a
    different model than the agent it defends.
    """
    import os

    os.environ.setdefault("GOOGLE_API_KEY", "dummy-for-construction")
    os.environ.setdefault("OPENAI_API_KEY", "dummy-for-construction")
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

    pipeline = AgentPipeline.from_config(
        PipelineConfig(llm="gemini-2.5-flash", model_id=None, defense="task_shield",
                       system_message_name=None, system_message=None, tool_output_format=None)
    )

    def walk(element):
        yield element
        for child in getattr(element, "elements", []):
            yield from walk(child)

    shields = [e for e in walk(pipeline) if type(e).__name__ == "TaskShield"]
    assert len(shields) == 1, f"expected one TaskShield, got {len(shields)}"
    shield = shields[0]
    assert shield.judge_model == shield.llm.model, (
        f"judge is {shield.judge_model!r} but the agent is {shield.llm.model!r}"
    )
    assert type(shield.client).__name__ == "GenaiChatCompletionsAdapter", (
        f"judge client is {type(shield.client).__name__}, not the genai adapter"
    )


def test_task_shield_runners_present():
    for name in ("run_llama33_70b_strong_defenses.sh",
                 "run_gpt4omini_task_shield_5attacks.sh"):
        assert (ROOT / "scripts" / name).is_file(), name


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
