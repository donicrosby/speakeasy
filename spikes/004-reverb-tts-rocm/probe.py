"""Environment probe for spike 004: does the gfx1151 HIP stack come up from the sandbox?"""

import json

import torch

info = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    info.update(
        {
            "device": torch.cuda.get_device_name(0),
            "gcnArchName": getattr(p, "gcnArchName", None),
            "total_memory_gb": round(p.total_memory / 2**30, 1),
            "multi_processor_count": p.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "sdpa": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            },
        }
    )
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float()
    torch.cuda.synchronize()
    info["bf16_matmul_ok"] = bool(torch.isfinite(y).all().item())
    free, total = torch.cuda.mem_get_info()
    info["mem_free_gb"] = round(free / 2**30, 1)
    info["mem_total_gb"] = round(total / 2**30, 1)

import torchaudio  # noqa: E402
import transformers  # noqa: E402

info["transformers"] = transformers.__version__
info["torchaudio"] = torchaudio.__version__
import reverb_tts  # noqa: E402
import verbalis  # noqa: E402

info["reverb_tts"] = reverb_tts.__file__
info["verbalis_ok"] = (
    verbalis.Verbalizer("en").process("I have $49 and 3 files.")
    if hasattr(verbalis, "Verbalizer")
    else "n/a"
)
print(json.dumps(info, indent=1))
