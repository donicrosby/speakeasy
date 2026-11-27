"""The universal lock must resolve each accelerator extra to its own stack.

In particular "no accelerator extra selected" must resolve to the CPU build.
That is what CI's `uv sync --all-groups` installs, and what a bare `uv run`
syncs to before running. The case still gets a torch, because `accelerate`
is a base dependency — and uv folds it into the fork of the extra named LAST
in `[tool.uv] conflicts`. With `rocm101` last, every no-extras install
silently pulled the ROCm 10.1 stack (~1 GB). `cpu` is last on purpose now.

This pins the resolved result rather than the list order, so a reordered
conflicts block or a relock that behaves differently fails here, not in CI's
install step. Only `uv export --frozen` runs: no sync, no downloads.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Packages that only exist in a GPU stack.
GPU_PREFIXES = ("rocm", "amd-torch-device", "nvidia-", "triton")

SELECTIONS = [
    pytest.param(
        ["--extra", "cpu"],
        ["torchaudio==\\S*\\+cpu"],
        GPU_PREFIXES,
        id="cpu",
    ),
    pytest.param(
        ["--extra", "cu130"],
        ["torch==\\S*\\+cu\\d+"],
        ("rocm", "amd-torch-device"),
        id="cu130",
    ),
    pytest.param(
        ["--extra", "rocm"],
        [
            "torch==\\S*\\+rocm[\\d.]+",
            "torchaudio==\\S*\\+rocm[\\d.]+",
            "triton-rocm==",
        ],
        ("nvidia-", "amd-torch-device"),
        id="rocm",
    ),
    pytest.param(
        ["--extra", "rocm101"],
        [
            "torch==\\S*\\+rocm[\\d.]+",
            "torchaudio==\\S*\\+rocm",
            "rocm-sdk-core==",
        ],
        ("nvidia-",),
        id="rocm101",
    ),
    pytest.param(
        ["--extra", "cpu", "--extra", "tts"],
        ["reverb-tts==", "torchaudio==\\S*\\+cpu"],
        GPU_PREFIXES,
        id="cpu+tts",
    ),
    pytest.param(
        ["--extra", "cu130", "--extra", "tts"],
        ["reverb-tts==", "torch==\\S*\\+cu\\d+"],
        ("rocm", "amd-torch-device"),
        id="cu130+tts",
    ),
    pytest.param(
        ["--extra", "rocm101", "--extra", "tts"],
        ["reverb-tts==", "rocm-sdk-core==", "torchaudio==\\S*\\+rocm"],
        ("nvidia-",),
        id="rocm101+tts",
    ),
]


def export(*extra: str) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    # Never let uv target another environment than the one running pytest.
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": sys.prefix}
    out = subprocess.run(
        [
            uv,
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            *extra,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, f"uv export {' '.join(extra)} failed:\n{out.stderr}"
    reqs = []
    for line in out.stdout.replace("\\\n", " ").splitlines():
        line = line.split(" --hash=")[0].rstrip(" \\")
        if line and not line.startswith("#"):
            reqs.append(line.strip())
    return reqs


def name_of(req: str) -> str:
    return re.split(r"[\s=;<>\[!~(]", req, maxsplit=1)[0]


def test_no_extras_resolves_to_the_cpu_build():
    reqs = export()
    leaked = sorted({name_of(r) for r in reqs if name_of(r).startswith(GPU_PREFIXES)})
    assert not leaked, (
        f"a no-extras install pulls a GPU stack: {leaked}. uv folds 'no "
        "accelerator selected' into the fork of the extra named LAST in "
        "[tool.uv] conflicts — that must be `cpu`."
    )
    # The export is universal: it lists the darwin wheel and the +cpu wheel.
    assert any(r.startswith("torch==") and "+cpu" in r for r in reqs), reqs


@pytest.mark.parametrize(("extra", "must_match", "forbidden"), SELECTIONS)
def test_extra_resolves_to_its_own_stack(extra, must_match, forbidden):
    reqs = export(*extra)
    for pattern in must_match:
        assert any(re.match(pattern, r) for r in reqs), (
            f"{extra}: no requirement matches {pattern!r}"
        )
    foreign = sorted({name_of(r) for r in reqs if name_of(r).startswith(forbidden)})
    assert not foreign, f"{extra}: packages from another stack: {foreign}"
