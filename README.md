# qwencandle

Qwen3-ASR-0.6B speech-to-text inference in Rust, built on [Candle](https://github.com/huggingface/candle).

## Build

```
cargo build --release
```

## Usage

Input is WAV float32 16kHz mono on stdin. Use ffmpeg to convert any audio format:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle <model_dir>
```

### Language forcing

By default the model auto-detects the spoken language. Use `--language` to force a specific output language:

```
ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle <model_dir> --language English
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Model

Download from HuggingFace:

```
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir qwen3-asr-0.6b
```

Then:

```
ffmpeg -i audio.wav -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | ./target/release/qwencandle ./qwen3-asr-0.6b
```
