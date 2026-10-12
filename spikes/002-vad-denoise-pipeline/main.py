"""Spike 002: VAD + denoise pipeline for noisy audio.

Validates: given noisy audio, when MetricGAN+ denoise runs before Whisper,
then transcription accuracy improves vs VAD-only.
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal
from faster_whisper import WhisperModel
from speechbrain.inference.enhancement import SpectralMaskEnhancement


def add_noise(audio_path: str, snr_db: float = 5.0) -> str:
    """Add white noise to audio at target SNR."""
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=1)  # stereo to mono
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    noisy = waveform + noise

    noisy_path = tempfile.mktemp(suffix=".wav")
    sf.write(noisy_path, noisy.numpy(), sr)
    return noisy_path


def denoise(audio_path: str, enhancer) -> str:
    """Run MetricGAN+ denoise on audio file."""
    # Load audio
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=1)  # stereo to mono

    # Resample to 16kHz if needed (MetricGAN+ expects 16kHz)
    if sr != 16000:
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = torch.from_numpy(
            scipy_signal.resample(waveform.numpy(), num_samples)
        ).float()

    # SpeechBrain expects (batch, time)
    waveform = waveform.unsqueeze(0)

    # Enhance
    enhanced = enhancer.enhance_batch(waveform, lengths=torch.tensor([1.0]))
    enhanced = enhanced.squeeze(0).cpu().numpy()

    # Save
    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, enhanced, 16000)
    return out_path


def transcribe(model, audio_path: str, vad_filter: bool = True) -> str:
    """Transcribe with faster-whisper."""
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=vad_filter,
    )
    return "".join(s.text for s in segments).strip()


def main():
    print("Loading models...")
    t0 = time.monotonic()

    whisper = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="/tmp/metricgan-plus",
    )
    print(f"Models loaded in {time.monotonic()-t0:.1f}s")

    # Use JFK clip as clean reference
    clean_path = "/tmp/jfk.flac"
    if not os.path.exists(clean_path):
        print("Downloading JFK test file...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
            clean_path
        )

    expected = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

    print("\n" + "="*60)
    print("BASELINE: clean audio")
    print("="*60)
    result = transcribe(whisper, clean_path, vad_filter=True)
    print(f"Result: {result}")
    print(f"Match: {result.lower() == expected.lower()}")

    # Test at various noise levels
    for snr in [10, 5, 0]:
        print(f"\n{'='*60}")
        print(f"SNR: {snr} dB")
        print("="*60)

        noisy_path = add_noise(clean_path, snr_db=snr)

        # Path 1: VAD only
        t0 = time.monotonic()
        result_vad = transcribe(whisper, noisy_path, vad_filter=True)
        t_vad = time.monotonic() - t0
        print(f"VAD only ({t_vad:.1f}s): {result_vad[:80]}...")

        # Path 2: Denoise + VAD
        t0 = time.monotonic()
        denoised_path = denoise(noisy_path, enhancer)
        result_denoised = transcribe(whisper, denoised_path, vad_filter=True)
        t_denoised = time.monotonic() - t0
        print(f"Denoise+VAD ({t_denoised:.1f}s): {result_denoised[:80]}...")

        # Compare
        vad_match = expected.lower() in result_vad.lower() or result_vad.lower() in expected.lower()
        denoised_match = expected.lower() in result_denoised.lower() or result_denoised.lower() in expected.lower()
        print(f"VAD match: {vad_match}, Denoised match: {denoised_match}")

        # Cleanup
        os.unlink(noisy_path)
        os.unlink(denoised_path)


if __name__ == "__main__":
    main()
