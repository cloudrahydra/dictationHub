"""CLI entry point for live microphone transcription."""

import argparse
import sys


BACKENDS = {
    "cohere": "mic_transcribe.backends.cohere",
    "qwen": "mic_transcribe.backends.qwen",
    "canary": "mic_transcribe.backends.canary",
    "parakeet": "mic_transcribe.backends.parakeet",
    "parakeet-v2": "mic_transcribe.backends.parakeet_v2",
    "parakeet-v3": "mic_transcribe.backends.parakeet_v3",
}


def _load_backend(name: str):
    """Import and return a backend module, with a helpful error on missing deps."""
    import importlib
    try:
        return importlib.import_module(BACKENDS[name])
    except ImportError as e:
        extras = {k: k for k in BACKENDS}
        print(f"Error: Missing dependencies for the '{name}' backend.", file=sys.stderr)
        print(f"Install them with:  uv sync --extra {extras[name]}", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        sys.exit(1)


def _detect_backend():
    """Try to auto-detect which backend is installed by checking upstream deps.

    Auto-detection is best-effort. When ambiguous (e.g. NeMo installed but
    could be canary or parakeet-v2), use --backend to be explicit.
    """
    import importlib
    # qwen_asr is unique to Qwen — check first since it also installs transformers
    try:
        importlib.import_module("qwen_asr")
        return "qwen"
    except ImportError:
        pass
    # nemo could be canary or parakeet-v2 — default to parakeet-v2
    try:
        importlib.import_module("nemo")
        return "parakeet-v2"
    except ImportError:
        pass
    # transformers — version determines which backend
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            return "cohere"
        else:
            return "parakeet"
    except ImportError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Live microphone transcription with state-of-the-art ASR models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
backends:
  cohere       CohereLabs/cohere-transcribe-03-2026  (14 langs, gated)
  qwen         Qwen/Qwen3-ASR-1.7B                  (52 langs, auto-detect)
  parakeet     nvidia/parakeet-ctc-1.1b              (English, fast CTC)
  parakeet-v2  nvidia/parakeet-tdt-0.6b-v2           (English, TDT, NeMo)
  parakeet-v3  nvidia/parakeet-tdt-0.6b-v3           (English, TDT, improved)
  canary       nvidia/canary-qwen-2.5b               (English, NeMo)

install one backend at a time (or clone into separate dirs):
  uv sync --extra qwen                  # recommended
  uv sync --extra qwen --extra parakeet # these two can coexist
  uv sync --extra cohere
  uv sync --extra parakeet-v2           # NeMo (heavy)
  uv sync --extra parakeet-v3
  uv sync --extra canary                # NeMo (heavy)
""",
    )
    parser.add_argument(
        "-b", "--backend",
        choices=list(BACKENDS.keys()),
        default=None,
        help="ASR backend to use (auto-detected if omitted)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Record for a fixed duration in seconds (default: record until Enter)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code/name (e.g. 'en', 'English'). Default depends on backend.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (see --list-devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        metavar="PATH",
        help="Save recorded audio to a WAV file",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously record and transcribe in a loop",
    )
    parser.add_argument(
        "--dictate",
        action="store_true",
        help="Dictation mode: stream mic and type transcribed text as keyboard input",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Speech energy threshold for dictation mode (default: 0.01)",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=0.4,
        help="Seconds of silence before finalizing an utterance in dictation mode (default: 0.4)",
    )
    parser.add_argument(
        "--reset-stats",
        action="store_true",
        help="Clear accumulated inference stats for the selected backend before starting",
    )

    args = parser.parse_args()

    from mic_transcribe.mic import list_devices, record_chunk, record_until_enter, save_wav

    if args.list_devices:
        list_devices()
        return

    # Resolve backend
    backend_name = args.backend
    if backend_name is None:
        backend_name = _detect_backend()
        if backend_name is None:
            print("Error: No ASR backend installed.", file=sys.stderr)
            print("Install one with:", file=sys.stderr)
            print("  uv sync --extra cohere", file=sys.stderr)
            print("  uv sync --extra qwen", file=sys.stderr)
            print("  uv sync --extra canary", file=sys.stderr)
            sys.exit(1)

    backend = _load_backend(backend_name)
    print(f"Backend: {backend.name()}")
    print()

    # Set default language per backend
    language = args.language
    if language is None:
        if backend_name == "cohere":
            language = "en"
        # qwen and canary: None means auto-detect / English-only

    if args.dictate:
        from mic_transcribe.dictate import run as run_dictate
        run_dictate(
            backend=backend,
            language=language,
            device=args.device,
            energy_threshold=args.threshold,
            silence_timeout=args.silence_timeout,
            reset_stats=args.reset_stats,
        )
        return

    def _do_one_recording():
        if args.duration:
            audio = record_chunk(args.duration, device=args.device)
        else:
            audio = record_until_enter(device=args.device)

        if len(audio) == 0:
            print("  No audio captured.")
            return

        duration_s = len(audio) / 16_000
        print(f"  Captured {duration_s:.1f}s of audio.")

        if args.save_audio:
            save_wav(audio, args.save_audio)
            print(f"  Audio saved to {args.save_audio}")

        print("  Transcribing ...", flush=True)
        text = backend.transcribe(audio, sample_rate=16_000, language=language)
        print()
        print(f"  >>> {text}")
        print()

    if args.continuous:
        if args.duration is None:
            print("Error: --continuous requires --duration", file=sys.stderr)
            sys.exit(1)
        print(f"Continuous mode: recording {args.duration}s chunks. Ctrl+C to stop.")
        print()
        try:
            while True:
                _do_one_recording()
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        _do_one_recording()


if __name__ == "__main__":
    main()
