"""NVIDIA Parakeet TDT 0.6B V2 backend (nvidia/parakeet-tdt-0.6b-v2).

Requires: uv sync --extra parakeet-v2
NeMo-only model (.nemo archive, no Transformers support).
TDT (Token-and-Duration Transducer) decoder. English only.
"""

import tempfile
import numpy as np

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"

_model = None


def _load():
    global _model
    if _model is not None:
        return

    import nemo.collections.asr as nemo_asr

    print(f"Loading {MODEL_ID} ...")
    _model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
    _model.eval()

    import torch
    if torch.cuda.is_available():
        _model = _model.cuda()
        print("  Model loaded on cuda.")
    else:
        print("  Model loaded on cpu.")


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

    from mic_transcribe.mic import save_wav

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        save_wav(audio, f.name)
        tmp_path = f.name

    output = _model.transcribe([tmp_path])

    import os
    os.unlink(tmp_path)

    if output and hasattr(output[0], "text"):
        return output[0].text
    elif output and isinstance(output[0], str):
        return output[0]
    return ""


def name() -> str:
    return "Parakeet TDT 0.6B V2"
