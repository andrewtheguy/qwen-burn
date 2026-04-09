/// Qwen3-ASR Audio Encoder.
/// Conv2D stem (per-chunk) → sinusoidal PE → windowed Transformer → projector.
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder};

// 0.6B encoder config
const D_MODEL: usize = 896;
const N_LAYERS: usize = 18;
const N_HEADS: usize = 14;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 64
const FFN_DIM: usize = 3584;
const OUTPUT_DIM: usize = 1024;
const DOWNSAMPLE_HIDDEN: usize = 480;
const N_WINDOW: usize = 50;
const N_WINDOW_INFER: usize = 800;
const CHUNK_SIZE: usize = N_WINDOW * 2; // 100 mel frames per chunk
const NUM_MEL_BINS: usize = 128;

struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl EncoderAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear(D_MODEL, D_MODEL, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(D_MODEL, D_MODEL, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(D_MODEL, D_MODEL, vb.pp("v_proj"))?,
            out_proj: candle_nn::linear(D_MODEL, D_MODEL, vb.pp("out_proj"))?,
        })
    }

    /// Bidirectional attention over a window slice. x: [seq, d_model]
    fn forward_window(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [1, n_heads, seq, head_dim], ensure contiguous for Metal
        let q = q
            .reshape((seq_len, N_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let k = k
            .reshape((seq_len, N_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let v = v
            .reshape((seq_len, N_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [1, n_heads, seq, head_dim]

        let out = ctx
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, D_MODEL))?;
        self.out_proj.forward(&out)
    }
}

struct EncoderLayer {
    self_attn: EncoderAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        };
        Ok(Self {
            self_attn: EncoderAttention::load(vb.pp("self_attn"))?,
            self_attn_layer_norm: candle_nn::layer_norm(
                D_MODEL,
                ln_cfg,
                vb.pp("self_attn_layer_norm"),
            )?,
            fc1: candle_nn::linear(D_MODEL, FFN_DIM, vb.pp("fc1"))?,
            fc2: candle_nn::linear(FFN_DIM, D_MODEL, vb.pp("fc2"))?,
            final_layer_norm: candle_nn::layer_norm(D_MODEL, ln_cfg, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        // Pre-attention layernorm
        let x_norm = self.self_attn_layer_norm.forward(x)?;

        // Windowed attention
        let attn_out = if cu_seqlens.len() <= 2 {
            self.self_attn.forward_window(&x_norm)?
        } else {
            let mut outputs = Vec::new();
            for i in 0..cu_seqlens.len() - 1 {
                let start = cu_seqlens[i];
                let end = cu_seqlens[i + 1];
                let window = x_norm.narrow(0, start, end - start)?;
                outputs.push(self.self_attn.forward_window(&window)?);
            }
            Tensor::cat(&outputs, 0)?
        };

        let x = (x + attn_out)?;

        // Pre-FFN layernorm
        let x_norm = self.final_layer_norm.forward(&x)?;
        let ffn = self.fc1.forward(&x_norm)?.gelu()?;
        let ffn = self.fc2.forward(&ffn)?;
        x + ffn
    }
}

pub struct AudioEncoder {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Tensor, // weight only, no bias — [d_model, 7680]
    positional_embedding: Tensor,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    device: Device,
}

impl AudioEncoder {
    pub fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2d1 = candle_nn::conv2d(1, DOWNSAMPLE_HIDDEN, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = candle_nn::conv2d(
            DOWNSAMPLE_HIDDEN,
            DOWNSAMPLE_HIDDEN,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = candle_nn::conv2d(
            DOWNSAMPLE_HIDDEN,
            DOWNSAMPLE_HIDDEN,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;
        let conv_out = vb.get(
            (D_MODEL, DOWNSAMPLE_HIDDEN * NUM_MEL_BINS / 8),
            "conv_out.weight",
        )?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        let vb_l = vb.pp("layers");
        for i in 0..N_LAYERS {
            layers.push(EncoderLayer::load(vb_l.pp(i))?);
        }

        let ln_cfg = LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        };
        let ln_post = candle_nn::layer_norm(D_MODEL, ln_cfg, vb.pp("ln_post"))?;
        let proj1 = candle_nn::linear(D_MODEL, D_MODEL, vb.pp("proj1"))?;
        let proj2 = candle_nn::linear(D_MODEL, OUTPUT_DIM, vb.pp("proj2"))?;
        let positional_embedding =
            sinusoidal_position_embedding(feat_extract_output_length(CHUNK_SIZE), D_MODEL, device)?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj1,
            proj2,
            device: device.clone(),
        })
    }

    /// mel: flat [NUM_MEL_BINS, n_frames] row-major f32 slice
    pub fn forward(&self, mel: &[f32], n_frames: usize) -> Result<Tensor> {
        if n_frames == 0 {
            return Tensor::zeros((0, OUTPUT_DIM), DType::F32, &self.device);
        }

        let num_full_chunks = n_frames / CHUNK_SIZE;
        let tail_frames = n_frames % CHUNK_SIZE;
        let num_chunks = num_full_chunks + usize::from(tail_frames > 0);

        let mut chunk_valid_tokens = Vec::with_capacity(num_chunks);
        let mut chunk_data = vec![0.0f32; num_chunks * NUM_MEL_BINS * CHUNK_SIZE];

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * CHUNK_SIZE;
            let chunk_len = std::cmp::min(CHUNK_SIZE, n_frames - start);
            chunk_valid_tokens.push(feat_extract_output_length(chunk_len));

            let chunk_offset = chunk_idx * NUM_MEL_BINS * CHUNK_SIZE;
            for m in 0..NUM_MEL_BINS {
                let src_start = m * n_frames + start;
                let src_end = src_start + chunk_len;
                let dst_start = chunk_offset + m * CHUNK_SIZE;
                let dst_end = dst_start + chunk_len;
                chunk_data[dst_start..dst_end].copy_from_slice(&mel[src_start..src_end]);
            }
        }

        // ── Conv2D stem (batched across chunks) ──
        let x = Tensor::from_vec(
            chunk_data,
            (num_chunks, 1, NUM_MEL_BINS, CHUNK_SIZE),
            &self.device,
        )?;
        let x = self.conv2d1.forward(&x)?.gelu()?;
        let x = self.conv2d2.forward(&x)?.gelu()?;
        let x = self.conv2d3.forward(&x)?.gelu()?;

        // x: [chunks, 480, freq, time] → [chunks, time, 480*freq]
        let (_, c, f, t) = x.dims4()?;
        let x = x
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .reshape((num_chunks, t, c * f))?;

        // Candle's matmul fast path here expects 2D x 2D, so flatten the batch first.
        let x = x.reshape((num_chunks * t, c * f))?;
        let x = x.matmul(&self.conv_out.t()?)?;
        let x = x.reshape((num_chunks, t, D_MODEL))?;
        let pos_emb = self.positional_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let x = x.broadcast_add(&pos_emb)?;

        // Drop padded tail tokens and flatten back to a single sequence.
        let mut chunk_outputs = Vec::with_capacity(num_chunks);
        for (chunk_idx, &valid_tokens) in chunk_valid_tokens.iter().enumerate() {
            let chunk = x.narrow(0, chunk_idx, 1)?.squeeze(0)?;
            chunk_outputs.push(chunk.narrow(0, 0, valid_tokens)?);
        }

        let x = if chunk_outputs.len() == 1 {
            chunk_outputs.pop().unwrap()
        } else {
            Tensor::cat(&chunk_outputs, 0)?
        };
        let total_tokens = x.dim(0)?;

        // ── Compute cu_seqlens for windowed attention ──
        let tokens_per_chunk = self.positional_embedding.dim(0)?;
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
            h = layer.forward(&h, &cu_seqlens)?;
        }

        // ── Final LayerNorm + projection ──
        let h = self.ln_post.forward(&h)?;
        let h = self.proj1.forward(&h)?.gelu()?;
        self.proj2.forward(&h)
    }
}

fn feat_extract_output_length(input_frames: usize) -> usize {
    let after_conv = |len: usize| -> usize { (len - 1) / 2 + 1 };
    after_conv(after_conv(after_conv(input_frames)))
}

/// Sinusoidal position embedding: [length, channels]
fn sinusoidal_position_embedding(
    length: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
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
    Tensor::from_vec(data, (length, channels), device)
}
