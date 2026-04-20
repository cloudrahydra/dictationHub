"""Persistent inference statistics using Welford's online algorithm.

Stores running (n, mean, M2, min, max) per metric so the file stays
constant-size regardless of how many sessions are accumulated.
"""

import json
import math
from pathlib import Path

STATS_DIR = Path.home() / ".cache" / "dictation_stats"

METRICS = ("audio_duration", "inference_time", "rtf")


def _stats_path(backend_name: str, model_id: str) -> Path:
    safe_name = f"{backend_name}_{model_id.replace('/', '_')}.json"
    return STATS_DIR / safe_name


def _empty_accumulator() -> dict:
    return {m: {"n": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}
            for m in METRICS}


def load(backend_name: str, model_id: str) -> dict:
    path = _stats_path(backend_name, model_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _empty_accumulator()


def save(backend_name: str, model_id: str, data: dict):
    path = _stats_path(backend_name, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def reset(backend_name: str, model_id: str):
    path = _stats_path(backend_name, model_id)
    if path.exists():
        path.unlink()


def update(acc: dict, audio_duration: float, inference_time: float):
    """Update accumulators with one observation (Welford's online algorithm)."""
    rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
    values = {"audio_duration": audio_duration, "inference_time": inference_time, "rtf": rtf}
    for m in METRICS:
        x = values[m]
        s = acc[m]
        s["n"] += 1
        delta = x - s["mean"]
        s["mean"] += delta / s["n"]
        delta2 = x - s["mean"]
        s["M2"] += delta * delta2
        s["min"] = min(s["min"], x)
        s["max"] = max(s["max"], x)


def summarize(acc: dict) -> str | None:
    """Format a one-line summary from the accumulators. None if empty."""
    n = acc[METRICS[0]]["n"]
    if n == 0:
        return None

    parts = [f"n={n}"]
    labels = {"audio_duration": "audio(s)", "inference_time": "infer(s)", "rtf": "RTF"}
    for m in METRICS:
        s = acc[m]
        std = math.sqrt(s["M2"] / s["n"]) if s["n"] > 0 else 0.0
        parts.append(
            f"{labels[m]}: mean={s['mean']:.3f} std={std:.3f} "
            f"min={s['min']:.3f} max={s['max']:.3f}"
        )
    return "  |  ".join(parts)
