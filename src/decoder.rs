/// Qwen3 LLM Decoder for ASR.
/// GQA with Q/K RMSNorm, NeoX RoPE, KV cache, SwiGLU Mlp, tied embeddings.
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};

// 0.6B decoder config
const HIDDEN_SIZE: usize = 1024;
const N_LAYERS: usize = 28;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 3072;
pub const VOCAB_SIZE: usize = 151936;
const ROPE_THETA: f64 = 1_000_000.0;
const RMS_EPS: f64 = 1e-6;
const MAX_SEQ_LEN: usize = 65536; // model config max_position_embeddings (official ASR limit: 1200s / 20min)

// ── RoPE ────────────────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1.0 / ROPE_THETA.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE. q,k shape: [B, H, L, D]
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    kv_cache: ConcatKvCache,
}

impl Attention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(HIDDEN_SIZE, N_HEADS * HEAD_DIM, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(N_HEADS * HEAD_DIM, HIDDEN_SIZE, vb.pp("o_proj"))?,
            q_norm: candle_nn::rms_norm(HEAD_DIM, RMS_EPS, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(HEAD_DIM, RMS_EPS, vb.pp("k_norm"))?,
            kv_cache: ConcatKvCache::new(2), // dim=2 for [B,H,S,D]
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        offset: usize,
        use_causal_sdpa: bool,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: [B, L, H, D] → [B, H, L, D]
        let q = q.reshape((b, l, N_HEADS, HEAD_DIM))?.transpose(1, 2)?;
        let k = k.reshape((b, l, N_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;
        let v = v.reshape((b, l, N_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;

        // Per-head RMSNorm on Q and K
        let q_flat = q.flatten(0, 2)?; // [B*H*L, D]
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, N_HEADS, l, HEAD_DIM))?;
        let k = k_flat.reshape((b, N_KV_HEADS, l, HEAD_DIM))?;

        // RoPE
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        if x.device().is_metal() && (use_causal_sdpa || (mask.is_none() && l == 1)) {
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();
            let ctx = candle_nn::ops::sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                None,
                use_causal_sdpa,
                scale as f32,
                1.0,
            )?;

            return ctx
                .transpose(1, 2)?
                .reshape((b, l, N_HEADS * HEAD_DIM))?
                .apply(&self.o_proj);
        }

        // GQA: repeat KV heads
        let n_groups = N_HEADS / N_KV_HEADS;
        let k = if n_groups > 1 {
            let (b_sz, n_kv, seq, hd) = k.dims4()?;
            Tensor::cat(&vec![&k; n_groups], 2)?
                .reshape((b_sz, n_kv * n_groups, seq, hd))?
                .contiguous()?
        } else {
            k.contiguous()?
        };
        let v = if n_groups > 1 {
            let (b_sz, n_kv, seq, hd) = v.dims4()?;
            Tensor::cat(&vec![&v; n_groups], 2)?
                .reshape((b_sz, n_kv * n_groups, seq, hd))?
                .contiguous()?
        } else {
            v.contiguous()?
        };

        // Attention
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // Merge heads
        ctx.transpose(1, 2)?
            .reshape((b, l, N_HEADS * HEAD_DIM))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ── Mlp ─────────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(HIDDEN_SIZE, INTERMEDIATE, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(HIDDEN_SIZE, INTERMEDIATE, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(INTERMEDIATE, HIDDEN_SIZE, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"))?,
            mlp: Mlp::load(vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(HIDDEN_SIZE, RMS_EPS, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(
                HIDDEN_SIZE,
                RMS_EPS,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        offset: usize,
        use_causal_sdpa: bool,
    ) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self
            .self_attn
            .forward(&h, mask, rotary, offset, use_causal_sdpa)?;
        let x = (x + h)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        x + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Decoder ────────────────────────────────────────────────────────────

pub struct Decoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor, // tied with embed_tokens
    rotary: RotaryEmbedding,
    device: Device,
    dtype: DType,
}

impl Decoder {
    pub fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(VOCAB_SIZE, HIDDEN_SIZE, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        let vb_l = vb_model.pp("layers");
        for i in 0..N_LAYERS {
            layers.push(DecoderLayer::load(vb_l.pp(i))?);
        }

        let norm = candle_nn::rms_norm(HIDDEN_SIZE, RMS_EPS, vb_model.pp("norm"))?;

        // Tied embeddings: lm_head shares weights with embed_tokens
        let lm_head_weight = embed_tokens.embeddings().clone();

        let rotary = RotaryEmbedding::new(vb.dtype(), device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight,
            rotary,
            device: device.clone(),
            dtype: vb.dtype(),
        })
    }

    /// Embed a single token ID.
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![token_id], (1,), &self.device)?;
        self.embed_tokens.forward(&ids)
    }

    /// Embed multiple token IDs. Returns [seq, hidden].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(token_ids.to_vec(), (token_ids.len(),), &self.device)?;
        self.embed_tokens.forward(&ids)
    }

    fn causal_mask(&self, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Forward pass on embeddings. Returns hidden states [B, L, hidden].
    fn forward_hidden(&mut self, h: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l, _) = h.dims3()?;
        // Metal SDPA has no CPU implementation and its fast causal path only applies
        // to offset=0 prefill sequences that are large enough to use the full kernel.
        let use_causal_sdpa = self.device.is_metal() && offset == 0 && l > 8;
        let mask = if l == 1 || use_causal_sdpa {
            None
        } else {
            Some(self.causal_mask(l, offset)?)
        };

        let mut h = h.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rotary, offset, use_causal_sdpa)?;
        }
        self.norm.forward(&h)
    }

    /// Forward pass with pre-built embeddings (for prefill with audio embeddings).
    /// embeds: [seq, hidden] → returns logits [vocab]
    pub fn forward_embed(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let embeds_3d = embeds.unsqueeze(0)?; // [1, seq, hidden]
        let h = self.forward_hidden(&embeds_3d, offset)?;
        // Take last position
        let seq_len = h.dim(1)?;
        let last = h.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [1, hidden]
        last.to_dtype(DType::F32)?.matmul(&self.lm_head_weight.t()?)
    }

    /// Forward pass for a single token during autoregressive generation.
    /// Returns logits [1, vocab].
    pub fn forward_token(&mut self, token_id: u32, offset: usize) -> Result<Tensor> {
        let embed = self.embed_token(token_id)?; // [1, hidden]
        let embed_3d = embed.unsqueeze(0)?; // [1, 1, hidden]
        let h = self.forward_hidden(&embed_3d, offset)?;
        let h = h.squeeze(1)?; // [1, hidden]
        h.to_dtype(DType::F32)?.matmul(&self.lm_head_weight.t()?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
