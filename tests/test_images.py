"""The images install the VoxCPM2 engine and prove it imports.

Both images add `--extra tts-voxcpm` beside `--extra tts` on every `uv sync`
line, and each sanity step imports the engine and pins its major version, so a
broken wheel fails the build rather than the first request. The ROCm image
replaces the locked PyPI ctranslate2 wheel with a source build in its last
step; a `uv sync` after that step would reinstall the PyPI wheel and undo it.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Each image's accelerator extra and the warm import its sanity step adds
# beyond `import voxcpm`.
IMAGES = {
    "Dockerfile.cuda": {"accelerator": "cu130", "warm_import": None},
    "Dockerfile.rocm": {
        "accelerator": "rocm101",
        "warm_import": "from voxcpm import VoxCPM",
    },
}


def instructions(dockerfile: str) -> list[str]:
    """The Dockerfile's logical lines, with ``\\`` continuations joined."""
    text = (ROOT / dockerfile).read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if line.strip()]


def sync_lines(dockerfile: str) -> list[str]:
    """The ``uv sync`` commands, without the (comment) lines that cite them."""
    return [
        match.group(0)
        for line in instructions(dockerfile)
        if not line.startswith("#")
        for match in [re.search(r"\buv sync\b.*", line)]
        if match
    ]


def sanity_commands(dockerfile: str) -> list[str]:
    """The RUN steps that execute the venv's python with ``-c``."""
    return [
        line
        for line in instructions(dockerfile)
        if line.startswith("RUN ") and ".venv/bin/python -c" in line
    ]


@pytest.mark.parametrize("dockerfile", sorted(IMAGES))
def test_each_sync_line_carries_both_extras(dockerfile):
    accelerator = IMAGES[dockerfile]["accelerator"]
    lines = sync_lines(dockerfile)
    assert len(lines) == 2, lines
    for line in lines:
        tokens = line.split()
        assert tokens[:2] == ["uv", "sync"], line
        extras = [
            tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "--extra"
        ]
        assert accelerator in extras, line
        assert "tts" in extras, line
        assert "tts-voxcpm" in extras, line


@pytest.mark.parametrize("dockerfile", sorted(IMAGES))
def test_sanity_step_imports_the_engine_and_pins_its_version(dockerfile):
    commands = sanity_commands(dockerfile)
    assert commands, f"{dockerfile}: no RUN step runs the venv python"
    importing = [command for command in commands if "import voxcpm" in command]
    assert importing, f"{dockerfile}: the sanity step does not import voxcpm"
    assert any("assert" in c and "2.0." in c for c in importing), (
        f"{dockerfile}: the sanity step does not assert the voxcpm version"
    )
    warm = IMAGES[dockerfile]["warm_import"]
    if warm is not None:
        assert any(warm in command for command in commands), (
            f"{dockerfile}: no {warm!r}"
        )


def test_rocm_syncs_before_it_replaces_the_ctranslate2_wheel():
    """The source-built wheel is installed last and must stay installed."""
    lines = [
        line for line in instructions("Dockerfile.rocm") if not line.startswith("#")
    ]
    wheel = [
        index
        for index, line in enumerate(lines)
        if "uv pip install" in line and "ct2-wheel" in line
    ]
    assert wheel, "Dockerfile.rocm: no RUN installs the source-built wheel"
    after = [line for line in lines[wheel[0] + 1 :] if re.search(r"\buv sync\b", line)]
    assert not after, f"Dockerfile.rocm: a uv sync follows the wheel install: {after}"
