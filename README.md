# mic-transcribe

Live microphone transcription using state-of-the-art ASR models.

Supports six backends:

| Backend | Model | Languages | Notes |
|---------|-------|-----------|-------|
| `cohere` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 14 | Gated model, needs `huggingface-cli login` |
| `qwen` | [Qwen/Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) | 52 | Auto language detection |
| `parakeet` | [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | English | Fast CTC model, lightweight install |
| `parakeet-v2` | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | English | TDT architecture, NeMo (heavy install) |
| `parakeet-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | English | TDT improved, lightweight install |
| `canary` | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | English | Largest NVIDIA model, NeMo (heavy install) |

Most backends have conflicting dependencies — install one at a time (or clone into separate directories). The exceptions are `qwen`+`parakeet`, which can coexist.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- PortAudio system library (for microphone access):
  ```bash
  # Debian/Ubuntu
  sudo apt install libportaudio2

  # Fedora
  sudo dnf install portaudio

  # Arch
  sudo pacman -S portaudio

  # macOS (via Homebrew)
  brew install portaudio
  ```
- A CUDA-capable GPU with 6+ GB VRAM is recommended (CPU works but is slow)

## Setup

```bash
# Lightweight options (recommended)
uv sync --extra qwen         # 52 languages, auto-detect
uv sync --extra parakeet     # English, fast CTC
uv sync --extra parakeet-v3  # English, TDT improved

# Gated model (requires huggingface-cli login)
uv sync --extra cohere

# NeMo-based (heavy install — builds from git)
uv sync --extra parakeet-v2
uv sync --extra canary
```

For Cohere, accept the model terms and log in:
```bash
uv run huggingface-cli login
```

For best performance with Qwen, install FlashAttention:
```bash
uv pip install flash-attn --no-build-isolation
```

## Usage

```bash
# Record until you press Enter, then transcribe
uv run mic-transcribe

# Record a fixed 5-second clip
uv run mic-transcribe --duration 5

# Specify backend and language explicitly
uv run mic-transcribe --backend qwen --language English

# Continuous mode: transcribe 10s chunks in a loop
uv run mic-transcribe --backend cohere --language en --duration 10 --continuous

# Dictation mode: stream mic and type transcribed text as keyboard input
uv run mic-transcribe --backend parakeet-v2 --dictate

# List audio input devices
uv run mic-transcribe --list-devices

# Use a specific input device
uv run mic-transcribe --device 2

# Save the recorded audio to a file
uv run mic-transcribe --save-audio recording.wav
```

## Options

```
-b, --backend BACKEND       ASR backend (auto-detected if omitted)
                            choices: cohere, qwen, parakeet, parakeet-v2, parakeet-v3, canary
-d, --duration SECS         Fixed recording duration (default: until Enter)
-l, --language LANG         Language code/name (e.g. 'en', 'English')
--device INDEX              Audio input device index (see --list-devices)
--list-devices              List available input devices and exit
--save-audio PATH           Save recorded audio to a WAV file
--continuous                Loop: record + transcribe repeatedly (requires --duration)
--dictate                   Dictation mode: stream mic and type transcribed text as keyboard input
--threshold FLOAT           Speech energy threshold for dictation mode (default: 0.01)
--silence-timeout SECS      Silence before finalizing an utterance in dictation mode (default: 0.4)
--reset-stats               Clear accumulated inference stats for the selected backend before starting
```

## GPU

All models benefit significantly from a CUDA GPU and will fall back to CPU if unavailable (slow). A GPU with at least 6 GB VRAM is recommended. The NeMo-based backends (`parakeet-v2`, `canary`) require torch 2.6–2.7.
