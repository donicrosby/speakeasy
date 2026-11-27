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
    pytest.param(
        ["--extra", "cpu", "--extra", "tts", "--extra", "tts-voxcpm"],
        ["voxcpm==", "reverb-tts==", "torchaudio==\\S*\\+cpu"],
        GPU_PREFIXES,
        id="cpu+tts+tts-voxcpm",
    ),
    pytest.param(
        ["--extra", "cu130", "--extra", "tts", "--extra", "tts-voxcpm"],
        ["voxcpm==", "reverb-tts==", "torch==\\S*\\+cu\\d+"],
        ("rocm", "amd-torch-device"),
        id="cu130+tts+tts-voxcpm",
    ),
    pytest.param(
        ["--extra", "rocm101", "--extra", "tts", "--extra", "tts-voxcpm"],
        ["voxcpm==", "reverb-tts==", "rocm-sdk-core==", "torchaudio==\\S*\\+rocm"],
        ("nvidia-",),
        id="rocm101+tts+tts-voxcpm",
    ),
]

# voxcpm's metadata pulls these in; none of them may reach any install. The
# lock's override-dependencies list staples this marker to each of them.
NEVER = "sys_platform == 'never'"
SUPPLY_CHAIN_LEAKS = (
    "torchcodec",
    "gradio",
    "gradio-client",
    "hf-gradio",
    "funasr",
    "spaces",
    "datasets",
    "argbind",
    "modelscope",
    "modelscope-hub",
)

ACCELERATORS = ("cpu", "cu130", "rocm", "rocm101")


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


# A requirement line (`name==version`, optionally marker-terminated). `uv
# export` also emits `# via ...` annotation comments; those belong to uv, not
# to the resolution, and they move when an extra adds a second route.
REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==")


def pins(reqs: list[str]) -> set[str]:
    return {req for req in reqs if REQUIREMENT.match(req)}


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


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_voxcpm2_extra_excludes_supply_chain_leaks(accelerator):
    """Adding the voxcpm extra adds packages, and re-pins nothing.

    voxcpm's metadata pulls ten packages its runtime path never imports. The
    lock's override-dependencies staple `sys_platform == 'never'` to each, so
    no real install can reach one; the export still lists every locked package,
    with that marker as the guard. The overrides also stop the fsspec
    downgrade one of them drags in. An existing pin that changes version on
    the way is equally a failure.
    """
    before = export("--extra", accelerator, "--extra", "tts")
    after = export("--extra", accelerator, "--extra", "tts", "--extra", "tts-voxcpm")

    named = {name_of(r): r for r in after if name_of(r) in SUPPLY_CHAIN_LEAKS}
    leaked = sorted(name for name, req in named.items() if NEVER not in req)
    assert not leaked, f"{accelerator}: voxcpm installs excluded packages: {leaked}"

    changed = sorted(pins(before) - pins(after))
    assert not changed, (
        f"{accelerator}: adding the voxcpm extra changed pins: {changed}"
    )

    assert any(r.startswith("voxcpm==") for r in after), (
        f"{accelerator}: the voxcpm extra adds no voxcpm"
    )
    fsspec = {r for r in after if name_of(r) == "fsspec"}
    assert fsspec == {"fsspec==2026.9.0"}, f"{accelerator}: fsspec moved: {fsspec}"
