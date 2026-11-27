#!/usr/bin/env python3
"""Assemble the spike 005 listening set: Reverb (spike 004) vs VoxCPM2, same reference, same words.

    /usr/bin/python3 ab.py

Writes ab/ (wav copies) and ab/listen.html. Open listen.html from the host:
    /opt/data/workspace/spike-005-scratch/ab/listen.html
Both engines clone the same reference clip, which is itself a synthetic preset
voice (Reverb's en_male_narrator_1), not a person.
"""

import html
import shutil
import wave
from pathlib import Path

S = Path("/workspace/spike-005-scratch")
V = S / "out" / "timing"  # VoxCPM2 takes
R = Path("/workspace/spike-004-scratch/out")  # Reverb takes from spike 004
AB = S / "ab"
AB.mkdir(exist_ok=True)


def dur(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def copy(src, name):
    dst = AB / f"{name}.wav"
    shutil.copyfile(src, dst)
    return f"{name}.wav", round(dur(dst), 1)


def audio(item):
    if item is None:
        return "<p><em>missing</em></p>"
    f, d = item
    return (
        f'<p><audio controls preload="none" src="{f}"></audio> <small>{d} s</small></p>'
    )


rows = []


def pair(title, note, reverb, voxcpm):
    rows.append(
        f'<section><h2>{html.escape(title)}</h2><p class="note">{html.escape(note)}</p>'
        f'<div class="pair"><div><h3>Reverb-TTS (in speakeasy today)</h3>{audio(reverb)}</div>'
        f"<div><h3>VoxCPM2 (candidate)</h3>{audio(voxcpm)}</div></div></section>"
    )


def single(title, note, items):
    body = "".join(
        f"<div><h3>{html.escape(label)}</h3>{audio(it)}</div>" for label, it in items
    )
    rows.append(
        f'<section><h2>{html.escape(title)}</h2><p class="note">{html.escape(note)}</p>'
        f'<div class="pair">{body}</div></section>'
    )


TEXTS = {
    "short": "Hello from an AMD graphics processor. This voice was cloned on ROCm.",
    "medium": "Speakeasy transcribes voice notes with Whisper and Moonshine. Tonight it learned to talk back. ... (mentions Reverb: the sentence is shared verbatim with spike 004)",
    "long": "Masked diffusion text to speech works nothing like the older autoregressive models. ... (133 words, shared verbatim with spike 004)",
    "polish": "Dzień dobry, jak się masz? Miło mi cię poznać.",
}

intro = copy(S / "out" / "reference.wav", "reference")
for key, label in (
    ("short", "Short"),
    ("medium", "Medium"),
    ("long", "Long"),
    ("polish", "Polish"),
):
    vox_name = "polish_clone" if key == "polish" else f"{key}_clone_warm1"
    pair(
        f"{label} sentence, cloned",
        TEXTS[key],
        copy(R / f"{key}.wav", f"reverb_{key}")
        if (R / f"{key}.wav").exists()
        else None,
        copy(V / f"{vox_name}.wav", f"voxcpm_{key}_clone"),
    )

single(
    "VoxCPM2's other modes (short sentence)",
    "Plain = no reference voice. Ultimate = clone by continuing the reference audio plus its transcript. "
    "Styled = clone with a style hint in the text. Design = a new voice from a description, no reference.",
    [
        ("Plain", copy(V / "short_plain_warm1.wav", "voxcpm_short_plain")),
        ("Ultimate clone", copy(V / "ultimate_short.wav", "voxcpm_short_ultimate")),
        ("Styled clone", copy(V / "styled_clone.wav", "voxcpm_short_styled")),
        ("Voice design", copy(V / "design.wav", "voxcpm_short_design")),
    ],
)
single(
    "VoxCPM2 quality dial: diffusion steps (short sentence, cloned)",
    "More steps is supposed to sound better and costs time: on this GPU 5 steps ran at RTF 0.91, "
    "10 at 1.06, 16 at 1.25. Whether 10 or 16 sounds better than 5 is the question.",
    [
        ("5 steps", copy(V / "sweep_short_clone_s5_warm1.wav", "voxcpm_short_s5")),
        (
            "10 steps (their default)",
            copy(V / "short_clone_warm1.wav", "voxcpm_short_s10"),
        ),
        ("16 steps", copy(V / "sweep_short_clone_s16_warm1.wav", "voxcpm_short_s16")),
    ],
)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>VoxCPM2 vs Reverb-TTS on the Radeon 8060S (spike 005)</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin:2rem 0 .3rem}} h3{{font-size:.95rem;margin:.4rem 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}} @media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}
.note{{color:#555;margin:.1rem 0 .6rem}} audio{{width:100%;max-width:420px}}
section{{border-top:1px solid #ddd;padding-top:.6rem}} small{{color:#777}}
</style></head><body>
<h1>VoxCPM2 vs Reverb-TTS on the Radeon 8060S (spike 005)</h1>
<p>Both engines clone the <b>same reference clip</b> and read the <b>same words</b>. The reference is a
synthetic preset voice, not a person. All VoxCPM2 takes here use the high-quality default
(CFG 2.0, 10 steps), which ran at about 1.05&times; real time on this GPU; Reverb runs at about 0.04.</p>
<h2>The reference both engines clone</h2>{audio(intro)}
<p class="note">"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."</p>
{"".join(rows)}
</body></html>"""
(AB / "listen.html").write_text(page, encoding="utf-8")
n = len(list(AB.glob("*.wav")))
print(f"wrote {AB / 'listen.html'} and {n} wav files")
