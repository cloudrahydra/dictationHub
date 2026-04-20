"""Dictation mode: continuous mic streaming with keyboard output.

Listens to the microphone, detects speech using energy-based VAD,
transcribes each utterance, and types the result as keyboard input.
"""

import subprocess
import shutil
import sys
import time
import threading
import numpy as np
import sounddevice as sd

from mic_transcribe.mic import SAMPLE_RATE, CHANNELS

# VAD parameters
BLOCK_DURATION = 0.05  # 50ms blocks for responsiveness
ENERGY_THRESHOLD = 0.01  # RMS energy threshold for speech detection
SILENCE_TIMEOUT = 1.5  # seconds of silence before finalizing utterance
MIN_UTTERANCE_DURATION = 0.3  # ignore very short blips


def _find_typer():
    """Find a tool to simulate keyboard input. Returns (name, callable)."""
    if shutil.which("xdotool"):
        def type_text(text):
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
                check=True,
            )
        return "xdotool", type_text

    if shutil.which("wtype"):
        def type_text(text):
            subprocess.run(["wtype", "--", text], check=True)
        return "wtype", type_text

    # Try pynput as fallback
    try:
        from pynput.keyboard import Controller
        kb = Controller()
        def type_text(text):
            kb.type(text)
        return "pynput", type_text
    except ImportError:
        pass

    return None, None


def run(backend, language, device=None, energy_threshold=ENERGY_THRESHOLD,
        silence_timeout=SILENCE_TIMEOUT, reset_stats=False):
    """Run dictation mode: stream mic -> transcribe -> type.

    Args:
        backend: A loaded backend module with .transcribe() and .name().
        language: Language code/name to pass to the backend.
        device: Audio input device index, or None for default.
        energy_threshold: RMS threshold to distinguish speech from silence.
        silence_timeout: Seconds of silence before ending an utterance.
        reset_stats: If True, clear accumulated stats before starting.
    """
    from mic_transcribe import stats

    backend_name = backend.name()
    model_id = getattr(backend, "MODEL_ID", backend_name)

    if reset_stats:
        stats.reset(backend_name, model_id)
        print(f"  Stats reset for {model_id}")

    stats_data = stats.load(backend_name, model_id)
    session_acc = stats._empty_accumulator()
    prev = stats.summarize(stats_data)
    if prev:
        print(f"  Prior stats: {prev}")
    typer_name, type_fn = _find_typer()
    if type_fn is None:
        print("Error: No keyboard input tool found.", file=sys.stderr)
        print("Install one of:", file=sys.stderr)
        print("  sudo apt install xdotool   (X11)", file=sys.stderr)
        print("  sudo apt install wtype     (Wayland)", file=sys.stderr)
        print("  uv pip install pynput      (fallback)", file=sys.stderr)
        sys.exit(1)

    print(f"Dictation mode")
    print(f"  Backend:  {backend.name()}")
    print(f"  Typer:    {typer_name}")
    print(f"  Device:   {device or 'default'}")
    print(f"  Threshold: {energy_threshold}")
    print()
    print("Warming up model...", flush=True)

    # Warm up with a tiny silent clip so first real transcription is fast
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    backend.transcribe(silent, sample_rate=SAMPLE_RATE, language=language)
    print("Ready. Start speaking — text will be typed into the focused window.")
    print("Press Ctrl+C to stop.")
    print()

    # Shared state between audio callback and main loop
    lock = threading.Lock()
    audio_buffer: list[np.ndarray] = []
    is_speaking = False
    silence_start: float | None = None

    # Accumulated utterance chunks
    utterance_chunks: list[np.ndarray] = []
    utterance_start_time: float | None = None
    pending_utterance: np.ndarray | None = None
    pending_event = threading.Event()

    block_size = int(BLOCK_DURATION * SAMPLE_RATE)

    def callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_start, utterance_start_time, pending_utterance

        if status:
            print(f"  [audio: {status}]", file=sys.stderr)

        chunk = indata[:, 0].copy()
        rms = np.sqrt(np.mean(chunk ** 2))

        with lock:
            if rms >= energy_threshold:
                # Speech detected
                if not is_speaking:
                    is_speaking = True
                    utterance_start_time = time.monotonic()
                silence_start = None
                utterance_chunks.append(chunk)
            elif is_speaking:
                # Silence while we were speaking
                utterance_chunks.append(chunk)  # keep the trailing silence
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= silence_timeout:
                    # Utterance complete
                    duration = time.monotonic() - (utterance_start_time or 0)
                    if duration >= MIN_UTTERANCE_DURATION and utterance_chunks:
                        pending_utterance = np.concatenate(utterance_chunks)
                        pending_event.set()
                    utterance_chunks.clear()
                    is_speaking = False
                    silence_start = None
                    utterance_start_time = None

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_size,
        device=device,
        callback=callback,
    )

    try:
        with stream:
            while True:
                pending_event.wait(timeout=0.1)
                if not pending_event.is_set():
                    continue

                with lock:
                    audio = pending_utterance
                    pending_utterance = None
                    pending_event.clear()

                if audio is None or len(audio) < int(MIN_UTTERANCE_DURATION * SAMPLE_RATE):
                    continue

                audio_duration = len(audio) / SAMPLE_RATE
                sys.stdout.write(f"\r  transcribing {audio_duration:.1f}s ...")
                sys.stdout.flush()

                t0 = time.monotonic()
                text = backend.transcribe(audio, sample_rate=SAMPLE_RATE, language=language)
                inference_time = time.monotonic() - t0
                text = text.strip()

                stats.update(session_acc, audio_duration, inference_time)
                stats.update(stats_data, audio_duration, inference_time)

                if text:
                    sys.stdout.write(f"\r  > {text}\033[K\n")
                    sys.stdout.flush()
                    type_fn(text + " ")
                else:
                    sys.stdout.write(f"\r\033[K")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        stats.save(backend_name, model_id, stats_data)
        session_summary = stats.summarize(session_acc)
        all_time_summary = stats.summarize(stats_data)
        print("\nDictation stopped.")
        if session_summary:
            print(f"  Session:  {session_summary}")
        if all_time_summary:
            print(f"  All-time: {all_time_summary}")
            print(f"  Saved to: {stats._stats_path(backend_name, model_id)}")
