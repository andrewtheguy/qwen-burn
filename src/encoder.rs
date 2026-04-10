/// Qwen3-ASR Audio Encoder.
/// Conv2D stem (per-chunk) → sinusoidal PE → windowed Transformer → projector.
use burn::nn::{conv::Conv2d, LayerNorm, Linear};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::weights::Tensors;

// 0.6B encoder config
const D_MODEL: usize = 896;
const N_LAYERS: usize = 18;
const N_HEADS: usize = 14;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 64
#[allow(dead_code)]
const FFN_DIM: usize = 3584;
#[allow(dead_code)]
const OUTPUT_DIM: usize = 1024;
#[allow(dead_code)]
const DOWNSAMPLE_HIDDEN: usize = 480;
const N_WINDOW: usize = 50;
const N_WINDOW_INFER: usize = 800;
const CHUNK_SIZE: usize = N_WINDOW * 2; // 100 mel frames per chunk
const NUM_MEL_BINS: usize = 128;

struct EncoderAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> EncoderAttention<B> {
    fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            q_proj: tensors.load_linear::<B>(&format!("{prefix}.q_proj"), device)?,
            k_proj: tensors.load_linear::<B>(&format!("{prefix}.k_proj"), device)?,
            v_proj: tensors.load_linear::<B>(&format!("{prefix}.v_proj"), device)?,
            out_proj: tensors.load_linear::<B>(&format!("{prefix}.out_proj"), device)?,
        })
    }

    /// Bidirectional attention over a window slice. x: [seq, d_model]
    fn forward_window(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let seq_len = x.dims()[0];
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [1, n_heads, seq, head_dim]
        let q = q
            .reshape([seq_len, N_HEADS, HEAD_DIM])
            .swap_dims(0, 1)
            .unsqueeze::<4>();
        let k = k
            .reshape([seq_len, N_HEADS, HEAD_DIM])
            .swap_dims(0, 1)
            .unsqueeze::<4>();
        let v = v
            .reshape([seq_len, N_HEADS, HEAD_DIM])
            .swap_dims(0, 1)
            .unsqueeze::<4>();

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v); // [1, n_heads, seq, head_dim]

        let out = ctx
            .squeeze::<3>()
            .swap_dims(0, 1)
            .reshape([seq_len, D_MODEL]);
        self.out_proj.forward(out)
    }
}

struct EncoderLayer<B: Backend> {
    self_attn: EncoderAttention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            self_attn: EncoderAttention::load(tensors, &format!("{prefix}.self_attn"), device)?,
            self_attn_layer_norm: tensors.load_layer_norm::<B>(
                &format!("{prefix}.self_attn_layer_norm"),
                1e-5,
                device,
            )?,
            fc1: tensors.load_linear::<B>(&format!("{prefix}.fc1"), device)?,
            fc2: tensors.load_linear::<B>(&format!("{prefix}.fc2"), device)?,
            final_layer_norm: tensors.load_layer_norm::<B>(
                &format!("{prefix}.final_layer_norm"),
                1e-5,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 2>, cu_seqlens: &[usize]) -> Tensor<B, 2> {
        // Pre-attention layernorm
        let x_norm = self.self_attn_layer_norm.forward(x.clone());

        // Windowed attention
        let attn_out = if cu_seqlens.len() <= 2 {
            self.self_attn.forward_window(x_norm)
        } else {
            let mut outputs = Vec::new();
            for i in 0..cu_seqlens.len() - 1 {
                let start = cu_seqlens[i];
                let end = cu_seqlens[i + 1];
                let window = x_norm.clone().narrow(0, start, end - start);
                outputs.push(self.self_attn.forward_window(window));
            }
            Tensor::cat(outputs, 0)
        };

        let x = x + attn_out;

        // Pre-FFN layernorm
        let x_norm = self.final_layer_norm.forward(x.clone());
        let ffn = gelu(self.fc1.forward(x_norm));
        let ffn = self.fc2.forward(ffn);
        x + ffn
    }
}

pub struct AudioEncoder<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    conv_out_t: Tensor<B, 2>, // pre-transposed: [7680, d_model]
    layers: Vec<EncoderLayer<B>>,
    ln_post: LayerNorm<B>,
    proj1: Linear<B>,
    proj2: Linear<B>,
    device: B::Device,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        let conv2d1 = tensors.load_conv2d::<B>(
            &format!("{prefix}.conv2d1"),
            [2, 2],
            [3, 3],
            [1, 1],
            1,
            device,
        )?;
        let conv2d2 = tensors.load_conv2d::<B>(
            &format!("{prefix}.conv2d2"),
            [2, 2],
            [3, 3],
            [1, 1],
            1,
            device,
        )?;
        let conv2d3 = tensors.load_conv2d::<B>(
            &format!("{prefix}.conv2d3"),
            [2, 2],
            [3, 3],
            [1, 1],
            1,
            device,
        )?;
        let conv_out =
            tensors.load_tensor::<B, 2>(&format!("{prefix}.conv_out.weight"), device)?;
        let conv_out_t_view = conv_out.transpose();
        let [r, c] = conv_out_t_view.dims();
        let conv_out_t = conv_out_t_view.reshape([r, c]); // force contiguous

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(EncoderLayer::load(
                tensors,
                &format!("{prefix}.layers.{i}"),
                device,
            )?);
        }

        let ln_post = tensors.load_layer_norm::<B>(&format!("{prefix}.ln_post"), 1e-5, device)?;
        let proj1 = tensors.load_linear::<B>(&format!("{prefix}.proj1"), device)?;
        let proj2 = tensors.load_linear::<B>(&format!("{prefix}.proj2"), device)?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out_t,
            layers,
            ln_post,
            proj1,
            proj2,
            device: device.clone(),
        })
    }

    /// mel: flat [NUM_MEL_BINS, n_frames] row-major f32 slice
    pub fn forward(&self, mel: &[f32], n_frames: usize) -> Tensor<B, 2> {
        // ── Conv2D stem (per-chunk) ──
        let mut chunk_outputs = Vec::new();

        let mut start = 0;
        while start < n_frames {
            let end = std::cmp::min(start + CHUNK_SIZE, n_frames);
            let chunk_len = end - start;

            // Extract chunk: [mel_bins, chunk_len] → Tensor [1, 1, mel_bins, chunk_len]
            let mut chunk_data = vec![0.0f32; NUM_MEL_BINS * chunk_len];
            for m in 0..NUM_MEL_BINS {
                for t in 0..chunk_len {
                    chunk_data[m * chunk_len + t] = mel[m * n_frames + start + t];
                }
            }
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(chunk_data, [1, 1, NUM_MEL_BINS, chunk_len]),
                &self.device,
            );

            // 3 x Conv2D + GELU
            let x = gelu(self.conv2d1.forward(x));
            let x = gelu(self.conv2d2.forward(x));
            let x = gelu(self.conv2d3.forward(x));

            // x: [1, 480, freq, time] → [time, 480*freq]
            let [_, c, f, t] = x.dims();
            let x = x.permute([0, 3, 1, 2]).reshape([t, c * f]);
            chunk_outputs.push(x);

            start += CHUNK_SIZE;
        }

        // Extract per-chunk token counts before consuming chunk_outputs
        let chunk_token_counts: Vec<usize> =
            chunk_outputs.iter().map(|c| c.dims()[0]).collect();
        let tokens_per_chunk = chunk_token_counts[0];

        // Concatenate chunks → [total_tokens, 480*freq]
        let x = Tensor::cat(chunk_outputs, 0);
        let total_tokens = x.dims()[0];

        // Linear projection: [total_tokens, 7680] → [total_tokens, d_model]
        let x = x.matmul(self.conv_out_t.clone());

        // ── Per-chunk sinusoidal position embeddings ──
        let pos_emb = sinusoidal_position_embedding::<B>(tokens_per_chunk, D_MODEL, &self.device);

        let pe_chunks: Vec<Tensor<B, 2>> = chunk_token_counts
            .iter()
            .map(|&count| pos_emb.clone().narrow(0, 0, count))
            .collect();
        let full_pe = Tensor::cat(pe_chunks, 0); // [total_tokens, D_MODEL]
        let x = x + full_pe;

        // ── Compute cu_seqlens for windowed attention ──
        let tokens_per_infer_window = tokens_per_chunk * (N_WINDOW_INFER / CHUNK_SIZE);
        let mut cu_seqlens = vec![0usize];
        let mut pos = 0;
        while pos < total_tokens {
            let window_end = std::cmp::min(pos + tokens_per_infer_window, total_tokens);
            cu_seqlens.push(window_end);
            pos = window_end;
        }

        // ── Transformer layers ──
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, &cu_seqlens);
        }

        // ── Final LayerNorm + projection ──
        let h = self.ln_post.forward(h);
        let h = gelu(self.proj1.forward(h));
        self.proj2.forward(h)
    }
}

/// Sinusoidal position embedding: [length, channels]
fn sinusoidal_position_embedding<B: Backend>(
    length: usize,
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half = channels / 2;
    let log_timescale = (10000.0f64).ln() / (half - 1) as f64;
    let inv_timescales: Vec<f32> = (0..half)
        .map(|i| (-log_timescale * i as f64).exp() as f32)
        .collect();

    let mut data = vec![0.0f32; length * channels];
    for t in 0..length {
        for i in 0..half {
            let angle = t as f32 * inv_timescales[i];
            data[t * channels + i] = angle.sin();
            data[t * channels + half + i] = angle.cos();
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [length, channels]), device)
}
