pub mod audio;
mod decoder;
mod encoder;
pub mod tokenizer;
pub mod weights;

#[cfg(feature = "python")]
mod python;

use anyhow::{bail, Context, Result};
use burn::tensor::{Int, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

pub type B = Wgpu<f32, i32>;
pub type Device = WgpuDevice;

pub const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-ASR-0.6B";

// Special token IDs
const TOKEN_IM_START: u32 = 151644;
const TOKEN_IM_END: u32 = 151645;
const TOKEN_AUDIO_START: u32 = 151669;
const TOKEN_AUDIO_END: u32 = 151670;
const TOKEN_AUDIO_PAD: u32 = 151676;
const TOKEN_ENDOFTEXT: u32 = 151643;
const TOKEN_ASR_TEXT: u32 = 151704;

const PROMPT_SUFFIX: &[u32] = &[
    TOKEN_AUDIO_END, TOKEN_IM_END, 198,
    TOKEN_IM_START, 77091, 198,
];

pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
];

pub struct QwenAsr {
    encoder: encoder::AudioEncoder,
    decoder: decoder::Decoder,
    tokenizer: tokenizer::Tokenizer,
}

impl QwenAsr {
    /// Load model on CPU from a HuggingFace model ID or local directory path.
    pub fn load(model_id: &str) -> Result<Self> {
        Self::load_on(model_id, &Default::default())
    }

    /// Load model on a specific device from a HuggingFace model ID or local directory path.
    pub fn load_on(model_id: &str, device: &Device) -> Result<Self> {
        let (safetensors_paths, model_dir) = resolve_model(model_id)?;

        let store = weights::TensorStore::open(&safetensors_paths)?;
        let tensors = store.tensors()?;

        eprintln!("  Loading encoder...");
        let encoder = encoder::AudioEncoder::load(&tensors, "thinker.audio_tower", device)?;
        eprintln!("  Loading tokenizer...");
        let tokenizer = tokenizer::Tokenizer::load(&model_dir)?;
        eprintln!("  Loading decoder...");
        let decoder = decoder::Decoder::load(&tensors, "thinker", device)?;
        eprintln!("  Model loaded.");

        Ok(Self { encoder, decoder, tokenizer })
    }

    /// Transcribe f32 PCM audio samples (16kHz mono).
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<String> {
        // Validate language
        if let Some(lang) = language {
            if !SUPPORTED_LANGUAGES.iter().any(|&l| l.eq_ignore_ascii_case(lang)) {
                bail!(
                    "Unsupported language: {}\nSupported: {}",
                    lang,
                    SUPPORTED_LANGUAGES.join(", ")
                );
            }
        }

        // Mel spectrogram
        let n_frames = audio::mel_frames(samples.len());
        let mel = audio::compute_mel_spectrogram(samples);

        // Encoder
        eprintln!("  Encoding audio...");
        let audio_embeds = self.encoder.forward(&mel, n_frames);
        let n_audio = audio_embeds.dims()[0];
        // Debug: audio embedding stats
        let ae_data: Vec<f32> = audio_embeds.clone().into_data().to_vec().unwrap();
        let ae_mean: f32 = ae_data.iter().sum::<f32>() / ae_data.len() as f32;
        let ae_max = ae_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ae_min = ae_data.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("  Audio encoded: {} tokens, mean={ae_mean:.4}, range=[{ae_min:.4}, {ae_max:.4}]", n_audio);

        // Tokenize context and language
        let context_tokens = match context {
            Some(ctx) if !ctx.is_empty() => self.tokenizer.encode(ctx)?,
            _ => Vec::new(),
        };

        let lang_tokens = match language {
            Some(lang) => {
                let mut toks = self.tokenizer.encode(&format!("language {}", lang))?;
                toks.push(TOKEN_ASR_TEXT);
                toks
            }
            None => Vec::new(),
        };

        // Build input_ids
        let mut input_ids: Vec<u32> = Vec::new();
        input_ids.push(TOKEN_IM_START);
        input_ids.push(8948); // "system"
        input_ids.push(198);  // "\n"
        input_ids.extend_from_slice(&context_tokens);
        input_ids.push(TOKEN_IM_END);
        input_ids.push(198);  // "\n"
        input_ids.push(TOKEN_IM_START);
        input_ids.push(872);  // "user"
        input_ids.push(198);  // "\n"
        input_ids.push(TOKEN_AUDIO_START);
        let prefix_len = input_ids.len();
        input_ids.extend(std::iter::repeat_n(TOKEN_AUDIO_PAD, n_audio));
        input_ids.extend_from_slice(PROMPT_SUFFIX);
        input_ids.extend_from_slice(&lang_tokens);
        let prompt_len = input_ids.len();

        // Embed tokens and replace audio positions
        let input_embeds = self.decoder.embed_tokens_ids(&input_ids);
        let before = input_embeds.clone().narrow(0, 0, prefix_len);
        let after = input_embeds
            .clone()
            .narrow(0, prefix_len + n_audio, prompt_len - prefix_len - n_audio);
        let input_embeds = Tensor::cat(vec![before, audio_embeds, after], 0);

        // Reset KV cache for fresh transcription
        self.decoder.clear_kv_cache();

        // Prefill
        eprintln!("  Prefilling {} tokens...", prompt_len - 1);
        let prefill_embeds = input_embeds.clone().narrow(0, 0, prompt_len - 1);
        self.decoder.forward_embed(&prefill_embeds, 0);
        eprintln!("  Prefill done.");

        // First token
        eprintln!("  Generating first token...");
        let last_embed = input_embeds.narrow(0, prompt_len - 1, 1);
        let logits = self.decoder.forward_embed(&last_embed, prompt_len - 1);
        // Debug: print top-5 logits
        let logits_data: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
        let mut indexed: Vec<(usize, f32)> = logits_data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("  Top-5 logits: {:?}", &indexed[..5]);
        let tok_tensor: Tensor<B, 2, Int> = logits.argmax(1);
        let tok_data: Vec<i64> = tok_tensor.into_data().to_vec().unwrap();
        let mut token = tok_data[0] as u32;
        let mut generated = vec![token];
        eprintln!("  First token: {}", token);

        // Autoregressive decode
        let max_new_tokens = 1024;
        for step in 0..max_new_tokens - 1 {
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
            let pos = prompt_len + step;
            let logits = self.decoder.forward_token(token, pos);
            let tok_tensor: Tensor<B, 2, Int> = logits.argmax(1);
            let tok_data: Vec<i64> = tok_tensor.into_data().to_vec().unwrap();
            token = tok_data[0] as u32;
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

        Ok(self.tokenizer.decode(&generated))
    }
}

/// Resolve model: local directory or HuggingFace hub download.
fn resolve_model(model_id: &str) -> Result<(Vec<PathBuf>, PathBuf)> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        let safetensors = find_safetensors(&local)?;
        return Ok((safetensors, local));
    }

    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let safetensors_path = repo.get("model.safetensors")
        .context("Failed to download model.safetensors")?;
    let vocab_path = repo.get("vocab.json")
        .context("Failed to download vocab.json")?;
    let _merges_path = repo.get("merges.txt")
        .context("Failed to download merges.txt")?;
    let _ = repo.get("tokenizer_config.json");
    let _ = repo.get("tokenizer.json");

    let model_dir = vocab_path.parent().unwrap().to_path_buf();
    Ok((vec![safetensors_path], model_dir))
}

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
        return Ok(shards.iter().map(|s| model_dir.join(s)).collect());
    }

    bail!("No model.safetensors found in {:?}", model_dir);
}
