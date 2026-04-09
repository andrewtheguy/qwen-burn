mod audio;
mod decoder;
mod encoder;
mod tokenizer;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::io::Read;
use std::path::PathBuf;

// Special token IDs
const TOKEN_IM_START: u32 = 151644;
const TOKEN_IM_END: u32 = 151645;
const TOKEN_AUDIO_START: u32 = 151669;
const TOKEN_AUDIO_END: u32 = 151670;
const TOKEN_AUDIO_PAD: u32 = 151676;
const TOKEN_ENDOFTEXT: u32 = 151643;
const TOKEN_ASR_TEXT: u32 = 151704;

// Prompt template
const PROMPT_PREFIX: &[u32] = &[
    TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198,
    TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START,
];
const PROMPT_SUFFIX: &[u32] = &[
    TOKEN_AUDIO_END, TOKEN_IM_END, 198,
    TOKEN_IM_START, 77091, 198,
];

// Supported languages (from config.json support_languages)
const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = None;
    let mut language: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--language" | "-l" => {
                i += 1;
                if i >= args.len() {
                    bail!("--language requires a value");
                }
                language = Some(args[i].clone());
            }
            arg if !arg.starts_with('-') => {
                model_dir = Some(PathBuf::from(arg));
            }
            "--help" | "-h" => {
                eprintln!("Usage: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle <model_dir> [--language <lang>]");
                eprintln!();
                eprintln!("Arguments:");
                eprintln!("  <model_dir>        Path to Qwen3-ASR-0.6B model directory");
                eprintln!("  --language <lang>  Force output language (e.g. English, Chinese, Japanese)");
                eprintln!();
                eprintln!("Supported languages:");
                eprintln!("  {}", SUPPORTED_LANGUAGES.join(", "));
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle ./qwen3-asr-0.6b");
                eprintln!("  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle ./qwen3-asr-0.6b -l Japanese");
                std::process::exit(0);
            }
            _ => bail!("Unknown argument: {}", args[i]),
        }
        i += 1;
    }

    let model_dir = model_dir.context(
        "Usage: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | \
         qwencandle <model_dir> [--language <lang>]\n\n\
         Arguments:\n  \
           <model_dir>        Path to Qwen3-ASR-0.6B model directory\n  \
           --language <lang>  Force output language (e.g. English, Chinese, Japanese)\n\n\
         Input: WAV float32 16kHz mono on stdin\n\n\
         Examples:\n  \
           ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle ./qwen3-asr-0.6b\n  \
           ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle ./qwen3-asr-0.6b -l Japanese",
    )?;

    // ── Read WAV from stdin ──
    let samples = read_wav_stdin()?;
    eprintln!(
        "Audio: {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // ── Mel spectrogram ──
    let n_frames = audio::mel_frames(samples.len());
    let mel = audio::compute_mel_spectrogram(&samples);
    eprintln!("Mel spectrogram: [128, {}]", n_frames);

    // ── Load model weights ──
    let device = Device::Cpu;
    let dtype = DType::F32;

    let safetensors_path = find_safetensors(&model_dir)?;
    eprintln!("Loading weights from {:?}...", model_dir);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensors_path, dtype, &device)?
    };

    // ── Encoder ──
    eprintln!("Running encoder...");
    let enc = encoder::AudioEncoder::load(vb.pp("thinker.audio_tower"), &device)?;
    let audio_embeds = enc.forward(&mel, n_frames)?;
    let n_audio = audio_embeds.dim(0)?;
    eprintln!("Audio embeddings: [{}, {}]", n_audio, audio_embeds.dim(1)?);

    // ── Tokenizer ──
    let tok = tokenizer::Tokenizer::load(&model_dir)?;

    // ── Decoder ──
    eprintln!("Loading decoder...");
    let mut dec = decoder::Decoder::load(vb.pp("thinker"), &device)?;

    // ── Build prompt ──
    let lang_tokens = match &language {
        Some(lang) => {
            // Validate language
            if !SUPPORTED_LANGUAGES.iter().any(|&l| l.eq_ignore_ascii_case(lang)) {
                bail!(
                    "Unsupported language: {}\nSupported: {}",
                    lang,
                    SUPPORTED_LANGUAGES.join(", ")
                );
            }
            // Tokenize "language <Name>" + <asr_text>
            let mut toks = tok.encode(&format!("language {}", lang))?;
            toks.push(TOKEN_ASR_TEXT);
            eprintln!("Language forcing: {} ({} tokens)", lang, toks.len());
            toks
        }
        None => Vec::new(),
    };

    let mut input_ids: Vec<u32> = Vec::new();
    input_ids.extend_from_slice(PROMPT_PREFIX);
    input_ids.extend(std::iter::repeat_n(TOKEN_AUDIO_PAD, n_audio));
    input_ids.extend_from_slice(PROMPT_SUFFIX);
    input_ids.extend_from_slice(&lang_tokens);
    let prompt_len = input_ids.len();
    eprintln!("Prompt: {} tokens ({} audio pads)", prompt_len, n_audio);

    // ── Embed tokens and replace audio positions ──
    let mut input_embeds = dec.embed_tokens(&input_ids)?; // [prompt_len, hidden]

    // Replace audio_pad positions with audio embeddings
    let prefix_len = PROMPT_PREFIX.len();
    let before = input_embeds.narrow(0, 0, prefix_len)?;
    let after = input_embeds.narrow(0, prefix_len + n_audio, prompt_len - prefix_len - n_audio)?;
    input_embeds = Tensor::cat(&[&before, &audio_embeds, &after], 0)?;

    // ── Prefill (all but last token) ──
    eprintln!("Prefilling {} tokens...", prompt_len - 1);
    let prefill_embeds = input_embeds.narrow(0, 0, prompt_len - 1)?;
    dec.forward_embed(&prefill_embeds, 0)?;

    // ── Generate first token from last prefill position ──
    let last_embed = input_embeds.narrow(0, prompt_len - 1, 1)?;
    let logits = dec.forward_embed(&last_embed, prompt_len - 1)?;
    let mut token = logits.argmax(1)?.to_vec1::<u32>()?[0];

    let mut generated = vec![token];

    // ── Autoregressive generation ──
    let max_new_tokens = 1024;
    for step in 0..max_new_tokens - 1 {
        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }
        let pos = prompt_len + step;
        let logits = dec.forward_token(token, pos)?;
        token = logits.argmax(1)?.to_vec1::<u32>()?[0];
        generated.push(token);
    }

    // Remove trailing EOS
    while let Some(&last) = generated.last() {
        if last == TOKEN_ENDOFTEXT || last == TOKEN_IM_END {
            generated.pop();
        } else {
            break;
        }
    }

    eprintln!("Generated {} tokens", generated.len());

    // ── Decode to text ──
    let text = tok.decode(&generated);
    println!("{text}");

    Ok(())
}

/// Read WAV float32 16kHz mono from stdin. Errors on wrong format.
fn read_wav_stdin() -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf)?;

    if buf.len() < 44 {
        bail!("Input too short for WAV");
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        bail!("Not a WAV file");
    }

    // Parse fmt chunk
    let mut pos = 12;
    let mut fmt_found = false;
    let mut audio_format: u16 = 0;
    let mut num_channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;

    loop {
        if pos + 8 > buf.len() {
            bail!("Could not find data chunk");
        }
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into()?) as usize;

        if chunk_id == b"fmt " && chunk_size >= 16 {
            let d = pos + 8;
            audio_format = u16::from_le_bytes(buf[d..d + 2].try_into()?);
            num_channels = u16::from_le_bytes(buf[d + 2..d + 4].try_into()?);
            sample_rate = u32::from_le_bytes(buf[d + 4..d + 8].try_into()?);
            bits_per_sample = u16::from_le_bytes(buf[d + 14..d + 16].try_into()?);
            fmt_found = true;
        }

        if chunk_id == b"data" {
            if !fmt_found {
                bail!("WAV missing fmt chunk before data");
            }
            // audio_format: 1=PCM int, 3=IEEE float
            if audio_format != 3 || bits_per_sample != 32 {
                bail!(
                    "Wrong WAV format: audio_format={} bits={} (expected float32).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | {}",
                    audio_format, bits_per_sample,
                    std::env::args().collect::<Vec<_>>().join(" ")
                );
            }
            if num_channels != 1 {
                bail!(
                    "Wrong WAV channels: {} (expected mono).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | {}",
                    num_channels,
                    std::env::args().collect::<Vec<_>>().join(" ")
                );
            }
            if sample_rate != 16000 {
                bail!(
                    "Wrong WAV sample rate: {} (expected 16000).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | {}",
                    sample_rate,
                    std::env::args().collect::<Vec<_>>().join(" ")
                );
            }

            let data_start = pos + 8;
            let data_end = std::cmp::min(data_start + chunk_size, buf.len());
            let data = &buf[data_start..data_end];
            let n_samples = data.len() / 4;
            let mut samples = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into()?;
                samples.push(f32::from_le_bytes(bytes));
            }
            return Ok(samples);
        }

        pos += 8 + chunk_size;
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }
}

/// Find safetensors file(s) in model directory.
fn find_safetensors(model_dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;
        let weight_map = index
            .get("weight_map")
            .context("Missing weight_map in index")?
            .as_object()
            .context("weight_map not object")?;

        let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for shard in weight_map.values() {
            if let Some(s) = shard.as_str() {
                shards.insert(s.to_string());
            }
        }

        let paths: Vec<PathBuf> = shards.iter().map(|s| model_dir.join(s)).collect();
        return Ok(paths);
    }

    bail!(
        "No model.safetensors or model.safetensors.index.json found in {:?}",
        model_dir
    );
}
