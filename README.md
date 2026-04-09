# qwencandle

Qwen3-ASR-0.6B speech-to-text inference in Rust, built on [Candle](https://github.com/huggingface/candle).

## Build

```
cargo build --release
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
--language <lang>  Force output language (e.g. English, Chinese, Japanese)
--help             Show help
```

### Language forcing

By default the model auto-detects the spoken language. Use `--language` to force a specific output language:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --language Japanese
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

### Local model

To use a locally downloaded model instead of auto-downloading:

```
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir qwen3-asr-0.6b
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle --model ./qwen3-asr-0.6b
```
