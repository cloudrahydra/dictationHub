"""Qwen3-ASR backend (Qwen/Qwen3-ASR-1.7B).

Requires: uv sync --extra qwen
"""

import numpy as np

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

_model = None


def _load():
    global _model
    if _model is not None:
        return

    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"Loading {MODEL_ID} ...")
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print("  Model loaded.")


def transcribe(audio: np.ndarray, sample_rate: int = 16_000, language: str | None = None) -> str:
    """Transcribe audio array to text.

    Args:
        audio: float32 numpy array at the given sample_rate.
        sample_rate: Sample rate of the audio (default 16kHz).
        language: Language name (e.g. "English", "Chinese") or None for auto-detect.

    Returns:
        Transcribed text string.
    """
    _load()

    results = _model.transcribe(
        audio=[(audio, sample_rate)],
        language=[language] if language else None,
    )

    return results[0].text if results else ""


def name() -> str:
    return "Qwen3-ASR 1.7B"
