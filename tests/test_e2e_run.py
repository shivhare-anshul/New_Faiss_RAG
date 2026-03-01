"""
End-to-end tests for the RAG pipeline.

Runs run.py as a subprocess (same as the evaluator will) and checks output.

Variables:
  result: subprocess.CompletedProcess. result.returncode (0 = success), result.stdout/stderr (str).
  We assert stdout contains "User Query:", "Retrieved Chunks:", "LLM Response:".

Skips if TEAMIFIED_OPENAI_API_KEY (or OPENAI_API_KEY) is not set.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _has_api_key() -> bool:
    return bool(os.environ.get("TEAMIFIED_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="No API key set; skip E2E")
@pytest.mark.parametrize("parser", ["pymupdf", "docling"])
def test_e2e_run_py_produces_expected_output(parser):
    """Run `python run.py --parser <parser>` and assert output contains expected sections."""
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--parser", parser],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    assert result.returncode == 0, (
        f"run.py --parser {parser} exited with {result.returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "User Query:" in stdout, f"[{parser}] Expected 'User Query:' in output"
    assert "Retrieved Chunks:" in stdout, f"[{parser}] Expected 'Retrieved Chunks:' in output"
    assert "LLM Response:" in stdout, f"[{parser}] Expected 'LLM Response:' in output"


def test_e2e_run_py_fails_gracefully_without_pdf():
    """run.py with non-existent PDF exits non-zero with a helpful error message."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--pdf", "/nonexistent/philippine_history.pdf"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "not found" in combined or "pdf" in combined
