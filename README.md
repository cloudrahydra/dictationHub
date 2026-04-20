# mic-transcribe

Live microphone transcription using state-of-the-art ASR models.

Supports three backends (install one at a time — they have conflicting dependencies):

| Backend | Model | Languages | Notes |
|---------|-------|-----------|-------|
| `cohere` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 14 | Gated model, needs `huggingface-cli login` |
| `qwen` | [Qwen/Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) | 52 | Auto language detection |
| `canary` | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | English only | Heaviest install (NeMo from git) |

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
# Install base + one backend
uv sync --extra qwen       # recommended: easiest install, 52 languages
# OR
uv sync --extra cohere     # best English WER, but gated model
# OR
uv sync --extra canary     # NVIDIA NeMo ecosystem
```

For Cohere, you also need to accept the model terms and log in:
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

# List audio input devices
uv run mic-transcribe --list-devices

# Use a specific input device
uv run mic-transcribe --device 2

# Save the recorded audio to a file
uv run mic-transcribe --save-audio recording.wav
```

## Options

```
-b, --backend {cohere,qwen,canary}  ASR backend (auto-detected if omitted)
-d, --duration SECS                 Fixed recording duration (default: until Enter)
-l, --language LANG                 Language code/name
--device INDEX                      Audio input device index
--list-devices                      List available input devices
--save-audio PATH                   Save recorded audio to WAV
--continuous                        Loop: record + transcribe repeatedly
```

## GPU

All three models benefit significantly from a CUDA GPU. They will fall back to CPU
but transcription will be slow. A GPU with at least 6GB VRAM is recommended for
the 1.7B-2.5B parameter models.
