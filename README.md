# qwencandle

Qwen3-ASR-0.6B speech-to-text inference in Rust, built on [Burn](https://github.com/tracel-ai/burn).

## Motivation

The main existing Python implementations of Qwen ASR (via PyTorch / Transformers) suffer from memory leaks when running inference repeatedly on Apple Silicon with Metal/MPS. Memory grows with each transcription call and is never fully reclaimed, eventually forcing a restart. This makes them unsuitable for long-running services or batch processing workflows on macOS.

This project reimplements Qwen3-ASR inference in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework with the wgpu backend, which provides Metal GPU support without the memory leak issues. The result is a lightweight, memory-stable binary (and Python library) that can transcribe audio indefinitely on Metal without degradation.

## Build

```
cargo build --release
```

With Metal GPU acceleration (macOS):

```
cargo build --release --features metal
```

## Usage

Input is WAV float32 16kHz mono on stdin. Use ffmpeg to convert any audio format:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle
```

The model is automatically downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/`.

### Options

```
--model <id>       HuggingFace model ID or local path (default: Qwen/Qwen3-ASR-0.6B)
--device <dev>     Device: cpu, metal (default: cpu)
--language <lang>  Force output language (e.g. English, Chinese, Japanese)
--context <text>   Condition on previous text (system prompt for consistency)
--help             Show help
```

### Thread count

### Metal GPU

On macOS, build with `--features metal` and use `--device metal` for GPU acceleration:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --device metal
```

### Language forcing

By default the model auto-detects the spoken language. Use `--language` to force a specific output language:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --language Japanese
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

### Context conditioning

Pass previous transcript text to improve consistency across segments:

```
ffmpeg -i segment2.wav -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --context "Previously the speaker discussed climate change."
```

### Local model

To use a locally downloaded model instead of auto-downloading:

```
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir qwen3-asr-0.6b
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --model ./qwen3-asr-0.6b
```

## Rust library

```rust
use qwencandle::QwenAsr;

let mut model = QwenAsr::load("Qwen/Qwen3-ASR-0.6B")?;
let text = model.transcribe(&samples, Some("English"), None)?;
```

## Python bindings

### Install from GitHub Pages

change `VERSION` to the version you want to install (e.g. `0.0.1a1`):

```
pip install --extra-index-url https://andrewtheguy.github.io/qwencandle/simple/ qwencandle==VERSION
```

Or with uv:

```
uv pip install --extra-index-url https://andrewtheguy.github.io/qwencandle/simple/ qwencandle==VERSION
```

### Install from source

```
uv venv
uv pip install numpy maturin
maturin develop --release
```

With Metal GPU support:

```
maturin develop --release --features metal
```

### Usage

```python
import numpy as np
import qwencandle

model = qwencandle.QwenAsr()  # auto-downloads from HuggingFace
text = model.transcribe(samples)  # samples: numpy float32 array, 16kHz mono

# with options
model = qwencandle.QwenAsr(device="metal")
text = model.transcribe(samples, language="English", context="Previous sentence.")
```

### API

```python
qwencandle.DEFAULT_MODEL_ID   # "Qwen/Qwen3-ASR-0.6B"
qwencandle.SUPPORTED_LANGUAGES  # list of 30 language names

class QwenAsr:
    def __init__(self, model_id=None, device="auto"): ...
    def transcribe(self, samples, *, language=None, context=None) -> str: ...
```
