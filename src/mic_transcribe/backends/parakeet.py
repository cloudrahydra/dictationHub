"""NVIDIA Parakeet CTC 1.1B backend (nvidia/parakeet-ctc-1.1b).

Requires: uv sync --extra parakeet
Compatible with transformers==4.57.x (can coexist with Qwen).
CTC decoder — no autoregressive generation, so inference is fast.
English only.
"""

import numpy as np
import torch

MODEL_ID = "nvidia/parakeet-ctc-1.1b"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return

    from transformers import AutoModelForCTC, AutoProcessor

    print(f"Loading {MODEL_ID} ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForCTC.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    _model.eval()
    print(f"  Model loaded on {device}.")


def transcribe(audio: np.ndarray, sample_rate: int = 16_000, language: str | None = None) -> str:
    """Transcribe audio array to text.

    Args:
        audio: float32 numpy array at the given sample_rate.
        sample_rate: Sample rate of the audio (default 16kHz).
        language: Ignored (English only).

    Returns:
        Transcribed text string.
    """
    _load()

    inputs = _processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(_model.device, dtype=_model.dtype) if v.is_floating_point() else v.to(_model.device)
              for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    # CTC greedy decode: take argmax, collapse repeats, remove blanks
    predicted_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(predicted_ids)
    return text


def name() -> str:
    return "Parakeet CTC 1.1B"
