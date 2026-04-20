"""NVIDIA Parakeet TDT 0.6B V3 backend (nvidia/parakeet-tdt-0.6b-v3).

Requires: uv sync --extra parakeet-v3
Supports HuggingFace Transformers >= 5.4.0.
TDT (Token-and-Duration Transducer) decoder. English only.
Successor to V2 with improved accuracy.
"""

import numpy as np

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"Loading {MODEL_ID} ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
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

    import torch

    inputs = _processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(_model.device, dtype=_model.dtype) if v.is_floating_point() else v.to(_model.device)
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=256)

    text = _processor.batch_decode(outputs, skip_special_tokens=True)
    return text[0] if text else ""


def name() -> str:
    return "Parakeet TDT 0.6B V3"
