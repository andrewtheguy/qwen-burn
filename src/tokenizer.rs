/// Tokenizer wrapper around HuggingFace tokenizers crate.
/// Loads from tokenizer.json if available, otherwise builds from vocab.json + merges.txt.
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::models::bpe::BPE;

const TOKEN_ASR_TEXT: u32 = 151704;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let json_path = model_dir.join("tokenizer.json");
        let inner = if json_path.exists() {
            tokenizers::Tokenizer::from_file(&json_path)
                .map_err(|e| anyhow::anyhow!("{}", e))
                .with_context(|| format!("Failed to load {:?}", json_path))?
        } else {
            // Build from vocab.json + merges.txt
            let vocab_path = model_dir.join("vocab.json");
            let merges_path = model_dir.join("merges.txt");
            let bpe = BPE::from_file(
                vocab_path.to_str().unwrap(),
                merges_path.to_str().unwrap(),
            )
            .build()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

            let mut tok = tokenizers::Tokenizer::new(bpe);
            // Qwen2 uses byte-level pre-tokenizer
            tok.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
            ));
            tok.with_decoder(Some(
                tokenizers::decoders::byte_level::ByteLevel::default(),
            ));
            tok
        };
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs to text, then parse ASR output (strip "language X<asr_text>" prefix).
    pub fn decode(&self, token_ids: &[u32]) -> String {
        // If <asr_text> token present, decode only tokens after it
        if let Some(pos) = token_ids.iter().position(|&t| t == TOKEN_ASR_TEXT) {
            let after = &token_ids[pos + 1..];
            return self
                .inner
                .decode(after, true)
                .unwrap_or_default()
                .trim()
                .to_string();
        }

        self.inner
            .decode(token_ids, true)
            .unwrap_or_default()
            .trim()
            .to_string()
    }
}
