"""Spike 004: where do ~75 s of cached model load go? Times each step of TTS.from_pretrained."""

import time

import torch

t0 = time.monotonic()


def lap(label: str, since: float) -> float:
    now = time.monotonic()
    print(f"{label:42s} {now - since:6.1f}s   (total {now - t0:6.1f}s)", flush=True)
    return now


from reverb_tts.model import Network  # noqa: E402
from reverb_tts.tts import (  # noqa: E402
    CODEC,
    CODEC_REVISION,
    REPOSITORY,
    REVISION,
    _resolve,
)
from transformers import HiggsAudioV2TokenizerModel, Qwen2Tokenizer  # noqa: E402

t = time.monotonic()
weights, tokenizer = _resolve(REPOSITORY, "int8", REVISION, None)
t = lap("resolve (snapshot_download, offline)", t)
t = lap("import torch", time.monotonic())
torch.zeros(1, device="cuda")  # HIP init
t = lap("HIP context init", t)
network = Network.load(weights)
t = lap("Network.load (safetensors + dequantize, CPU)", t)
network = network.to("cuda", torch.bfloat16)
torch.cuda.synchronize()
t = lap("network.to(cuda, bf16)", t)
codec = HiggsAudioV2TokenizerModel.from_pretrained(CODEC, revision=CODEC_REVISION)
t = lap("codec from_pretrained (CPU, fp32)", t)
codec = codec.to("cuda").eval()
torch.cuda.synchronize()
t = lap("codec.to(cuda)", t)
Qwen2Tokenizer.from_pretrained(tokenizer)
t = lap("tokenizer", t)
network2 = Network.load(weights)
t = lap("Network.load again (warm)", t)
codec2 = HiggsAudioV2TokenizerModel.from_pretrained(CODEC, revision=CODEC_REVISION)
t = lap("codec from_pretrained again (warm)", t)
