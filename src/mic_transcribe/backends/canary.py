"""NVIDIA Canary-Qwen-2.5B backend.

Requires: uv sync --extra canary
Note: This is the heaviest install (NeMo toolkit from git). English only.
"""

import tempfile
import numpy as np

MODEL_ID = "nvidia/canary-qwen-2.5b"

_model = None


def _load():
    global _model
    if _model is not None:
        return

    from nemo.collections.speechlm2.models import SALM

    print(f"Loading {MODEL_ID} ...")
    _model = SALM.from_pretrained(MODEL_ID)
    print("  Model loaded.")


def transcribe(audio: np.ndarray, sample_rate: int = 16_000, language: str = "en") -> str:
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

    answer_ids = _model.generate(
        prompts=[
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {_model.audio_locator_tag}",
                    "audio": [tmp_path],
                }
            ]
        ],
        max_new_tokens=256,
    )

    import os
    os.unlink(tmp_path)

    return _model.tokenizer.ids_to_text(answer_ids[0].cpu())


def name() -> str:
    return "NVIDIA Canary-Qwen 2.5B"
