"""Cohere Transcribe backend (CohereLabs/cohere-transcribe-03-2026).

Requires: uv sync --extra cohere
Note: This is a gated model — run `huggingface-cli login` first.
"""

import tempfile
import numpy as np

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is not None:
        return

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    print(f"Loading {MODEL_ID} ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, trust_remote_code=True
    ).to(device)
    _model.eval()
    print(f"  Model loaded on {device}.")


def transcribe(audio: np.ndarray, sample_rate: int = 16_000, language: str = "en") -> str:
    """Transcribe audio array to text.

    Args:
        audio: float32 numpy array at the given sample_rate.
        sample_rate: Sample rate of the audio (default 16kHz).
        language: ISO 639-1 language code (default "en").

    Returns:
        Transcribed text string.
    """
    _load()

    from mic_transcribe.mic import save_wav

    # The trust_remote_code .transcribe() API accepts file paths
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        save_wav(audio, f.name)
        tmp_path = f.name

    texts = _model.transcribe(
        processor=_processor,
        audio_files=[tmp_path],
        language=language,
    )

    import os
    os.unlink(tmp_path)

    return texts[0] if texts else ""


def name() -> str:
    return "Cohere Transcribe"
