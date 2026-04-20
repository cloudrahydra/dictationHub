"""Microphone capture utilities.

Records audio from the default input device at 16kHz mono,
which is the sample rate expected by all three ASR backends.
"""

import sys
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except OSError as e:
    if "PortAudio" in str(e):
        print("Error: PortAudio library not found.", file=sys.stderr)
        print("Install it with:  sudo apt install libportaudio2  (Debian/Ubuntu)", file=sys.stderr)
        print("                  brew install portaudio           (macOS)", file=sys.stderr)
        sys.exit(1)
    raise

SAMPLE_RATE = 16_000
CHANNELS = 1


def list_devices():
    """Print available audio input devices."""
    print("Available audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']} (inputs: {d['max_input_channels']}){marker}")
    print()
    print("  * = default input device")


def record_chunk(duration: float, device: int | None = None) -> np.ndarray:
    """Record a single chunk of audio from the microphone.

    Returns float32 numpy array of shape (samples,) at 16kHz.
    """
    print(f"  Recording {duration:.1f}s ...", end=" ", flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("done.")
    return audio.squeeze()


def record_until_enter(device: int | None = None) -> np.ndarray:
    """Record audio until the user presses Enter.

    Returns float32 numpy array of shape (samples,) at 16kHz.
    """
    chunks: list[np.ndarray] = []
    block_duration = 0.5  # seconds per callback block

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio warning: {status}]", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(block_duration * SAMPLE_RATE),
        device=device,
        callback=callback,
    )

    print("  Recording ... press Enter to stop.", flush=True)
    with stream:
        input()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).squeeze()


def save_wav(audio: np.ndarray, path: str):
    """Save audio array to a WAV file at 16kHz."""
    sf.write(path, audio, SAMPLE_RATE)
