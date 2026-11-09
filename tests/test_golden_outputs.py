"""Every number in the tutorial must be reproducible.

Runs each demo script and compares its output, byte for byte, with the files
in expected_output/. The Monte Carlo script is skipped by default because it
takes about a minute: run it with  RUN_SLOW=1 python -m pytest -q
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

FAST = ["demo_leaks", "demo_overfitting", "audit"]
SLOW = ["monte_carlo"]


def run_script(name: str) -> str:
    result = subprocess.run(
        [sys.executable, f"{name}.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.mark.parametrize("name", FAST)
def test_output_matches_tutorial(name):
    assert run_script(name) == (ROOT / "expected_output" / f"{name}.txt").read_text()


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to run")
@pytest.mark.parametrize("name", SLOW)
def test_slow_output_matches_tutorial(name):
    assert run_script(name) == (ROOT / "expected_output" / f"{name}.txt").read_text()
