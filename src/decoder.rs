/// Qwen3 LLM Decoder for ASR.
/// GQA with Q/K RMSNorm, RoPE, KV cache, SwiGLU Mlp, tied embeddings.
use burn::nn::{Embedding, Linear, RmsNorm};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::weights::Tensors;

// 0.6B decoder config
#[allow(dead_code)]
const HIDDEN_SIZE: usize = 1024;
const N_LAYERS: usize = 28;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
#[allow(dead_code)]
const INTERMEDIATE: usize = 3072;
#[allow(dead_code)]
pub const VOCAB_SIZE: usize = 151936;
const ROPE_THETA: f64 = 1_000_000.0;
const RMS_EPS: f64 = 1e-6;
const MAX_SEQ_LEN: usize = 65536;

// ── RoPE (split-half / NeoX convention) ────────────────────────────────────

struct RotaryEmbedding<B: Backend> {
    sin: Tensor<B, 2>, // [MAX_SEQ_LEN, HEAD_DIM/2]
    cos: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    fn new(device: &B::Device) -> Self {
        let half = HEAD_DIM / 2;
        let inv_freq: Vec<f32> = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1.0 / ROPE_THETA.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect();

        let mut sin_data = vec![0.0f32; MAX_SEQ_LEN * half];
        let mut cos_data = vec![0.0f32; MAX_SEQ_LEN * half];
        for t in 0..MAX_SEQ_LEN {
            for i in 0..half {
                let angle = t as f32 * inv_freq[i];
                sin_data[t * half + i] = angle.sin();
                cos_data[t * half + i] = angle.cos();
            }
        }

        let sin = Tensor::<B, 2>::from_data(
            TensorData::new(sin_data, [MAX_SEQ_LEN, half]),
            device,
        );
        let cos = Tensor::<B, 2>::from_data(
            TensorData::new(cos_data, [MAX_SEQ_LEN, half]),
            device,
        );
        Self { sin, cos }
    }

    /// Apply RoPE to q and k of shape [B, H, L, D].
    fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_, _, l, d] = q.dims();
        let half = d / 2;

        let cos = self.cos.clone().narrow(0, offset, l);
        let sin = self.sin.clone().narrow(0, offset, l);

        let cos = cos.reshape([1, 1, l, half]);
        let sin = sin.reshape([1, 1, l, half]);

        let q_out = Self::rope_tensor(q, &cos, &sin, half);
        let k_out = Self::rope_tensor(k, &cos, &sin, half);
        (q_out, k_out)
    }

    /// Split-half rotation: x1 = x[..., :d/2], x2 = x[..., d/2:]
    fn rope_tensor(
        x: Tensor<B, 4>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        half: usize,
    ) -> Tensor<B, 4> {
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);

        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin.clone() + x2 * cos.clone();

        Tensor::cat(vec![out1, out2], 3)
    }
}

// ── KV Cache ───────────────────────────────────────────────────────────────

struct KvCache<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn append(
        &mut self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = match self.k.take() {
            Some(old) => Tensor::cat(vec![old, k], 2),
            None => k,
        };
        let v = match self.v.take() {
            Some(old) => Tensor::cat(vec![old, v], 2),
            None => v,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        (k, v)
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    kv_cache: KvCache<B>,
}

impl<B: Backend> Attention<B> {
    fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            q_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.q_proj"), device)?,
            k_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.k_proj"), device)?,
            v_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.v_proj"), device)?,
            o_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.o_proj"), device)?,
            q_norm: tensors.load_rms_norm::<B>(&format!("{prefix}.q_norm"), RMS_EPS, device)?,
            k_norm: tensors.load_rms_norm::<B>(&format!("{prefix}.k_norm"), RMS_EPS, device)?,
            kv_cache: KvCache::new(),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor<B, 3>,
        mask: Option<&Tensor<B, 4>>,
        rotary: &RotaryEmbedding<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x.clone());

        // Reshape: [B, L, H*D] → [B, H, L, D]
        let q = q
            .reshape([b, l, N_HEADS, HEAD_DIM])
            .swap_dims(1, 2);
        let k = k
            .reshape([b, l, N_KV_HEADS, HEAD_DIM])
            .swap_dims(1, 2);
        let v = v
            .reshape([b, l, N_KV_HEADS, HEAD_DIM])
            .swap_dims(1, 2);

        // Per-head RMSNorm on Q and K
        let q_flat: Tensor<B, 2> = q.reshape([b * N_HEADS * l, HEAD_DIM]);
        let k_flat: Tensor<B, 2> = k.reshape([b * N_KV_HEADS * l, HEAD_DIM]);
        let q_flat = self.q_norm.forward(q_flat);
        let k_flat = self.k_norm.forward(k_flat);
        let q: Tensor<B, 4> = q_flat.reshape([b, N_HEADS, l, HEAD_DIM]);
        let k: Tensor<B, 4> = k_flat.reshape([b, N_KV_HEADS, l, HEAD_DIM]);

        // RoPE
        let (q, k) = rotary.apply(q, k, offset);

        // KV cache
        let (k, v) = self.kv_cache.append(k, v);

        // GQA: repeat KV heads
        let n_groups = N_HEADS / N_KV_HEADS;
        let k = if n_groups > 1 {
            let [b_sz, n_kv, seq, hd] = k.dims();
            Tensor::cat(vec![k; n_groups], 2)
                .reshape([b_sz, n_kv * n_groups, seq, hd])
        } else {
            k
        };
        let v = if n_groups > 1 {
            let [b_sz, n_kv, seq, hd] = v.dims();
            Tensor::cat(vec![v; n_groups], 2)
                .reshape([b_sz, n_kv * n_groups, seq, hd])
        } else {
            v
        };

        // Attention
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        if let Some(m) = mask {
            scores = scores + m.clone();
        }
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v);

        // Merge heads
        let out = ctx
            .swap_dims(1, 2)
            .reshape([b, l, N_HEADS * HEAD_DIM]);
        self.o_proj.forward(out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ── Mlp ─────────────────────────────────────────────────────────────────────

struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.gate_proj"), device)?,
            up_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.up_proj"), device)?,
            down_proj: tensors.load_linear_no_bias::<B>(&format!("{prefix}.down_proj"), device)?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer<B: Backend> {
    self_attn: Attention<B>,
    mlp: Mlp<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            self_attn: Attention::load(tensors, &format!("{prefix}.self_attn"), device)?,
            mlp: Mlp::load(tensors, &format!("{prefix}.mlp"), device)?,
            input_layernorm: tensors.load_rms_norm::<B>(
                &format!("{prefix}.input_layernorm"),
                RMS_EPS,
                device,
            )?,
            post_attention_layernorm: tensors.load_rms_norm::<B>(
                &format!("{prefix}.post_attention_layernorm"),
                RMS_EPS,
                device,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor<B, 3>,
        mask: Option<&Tensor<B, 4>>,
        rotary: &RotaryEmbedding<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let h = self.input_layernorm.forward(x.clone());
        let h = self.self_attn.forward(&h, mask, rotary, offset);
        let x = x.clone() + h;
        let h = self.post_attention_layernorm.forward(x.clone());
        let h = self.mlp.forward(h);
        x + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Decoder ────────────────────────────────────────────────────────────

pub struct Decoder<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<DecoderLayer<B>>,
    norm: RmsNorm<B>,
    lm_head_weight: Tensor<B, 2>, // tied with embed_tokens
    rotary: RotaryEmbedding<B>,
    device: B::Device,
}

impl<B: Backend> Decoder<B> {
    pub fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        let embed_tokens =
            tensors.load_embedding::<B>(&format!("{prefix}.model.embed_tokens"), device)?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(DecoderLayer::load(
                tensors,
                &format!("{prefix}.model.layers.{i}"),
                device,
            )?);
        }

        let norm = tensors.load_rms_norm::<B>(&format!("{prefix}.model.norm"), RMS_EPS, device)?;
        let lm_head_weight = embed_tokens.weight.val();
        let rotary = RotaryEmbedding::new(device);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight,
            rotary,
            device: device.clone(),
        })
    }

    /// Embed a single token ID.
    pub fn embed_token(&self, token_id: u32) -> Tensor<B, 2> {
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![token_id as i32], [1, 1]),
            &self.device,
        );
        let out = self.embed_tokens.forward(ids);
        let [_, _, h] = out.dims();
        out.reshape([1, h])
    }

    /// Embed multiple token IDs. Returns [seq, hidden].
    pub fn embed_tokens_ids(&self, token_ids: &[u32]) -> Tensor<B, 2> {
        let ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let seq = token_ids.len();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids_i32, [1, seq]),
            &self.device,
        );
        let out = self.embed_tokens.forward(ids);
        let [_, _, h] = out.dims();
        out.reshape([seq, h])
    }

    fn causal_mask(&self, tgt: usize, offset: usize) -> Tensor<B, 4> {
        let minf = f32::NEG_INFINITY;
        let total = tgt + offset;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| {
                (0..total).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::<B, 4>::from_data(
            TensorData::new(mask, [1, 1, tgt, total]),
            &self.device,
        )
    }

    fn forward_hidden(&mut self, h: &Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_, l, _] = h.dims();
        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(l, offset))
        };

        let mut h = h.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rotary, offset);
        }
        self.norm.forward(h)
    }

    /// Forward pass with pre-built embeddings.
    /// embeds: [seq, hidden] → returns logits [1, vocab]
    pub fn forward_embed(&mut self, embeds: &Tensor<B, 2>, offset: usize) -> Tensor<B, 2> {
        let embeds_3d = embeds.clone().unsqueeze::<3>();
        let h = self.forward_hidden(&embeds_3d, offset);
        let seq_len = h.dims()[1];
        let hidden = h.dims()[2];
        let last: Tensor<B, 2> = h.narrow(1, seq_len - 1, 1).reshape([1, hidden]);
        last.matmul(self.lm_head_weight.clone().transpose())
    }

    /// Forward pass for a single token during autoregressive generation.
    pub fn forward_token(&mut self, token_id: u32, offset: usize) -> Tensor<B, 2> {
        let embed = self.embed_token(token_id);
        let embed_3d = embed.unsqueeze::<3>();
        let h = self.forward_hidden(&embed_3d, offset);
        let hidden = h.dims()[2];
        let h: Tensor<B, 2> = h.reshape([1, hidden]);
        h.matmul(self.lm_head_weight.clone().transpose())
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
