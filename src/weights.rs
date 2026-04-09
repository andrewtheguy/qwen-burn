/// Weight loading helpers for safetensors → burn.
use anyhow::{bail, Context, Result};
use burn::module::{Ignored, Param, ParamId};
use burn::nn::{
    conv::Conv2d, Embedding, LayerNorm, LayerNormConfig, Linear, PaddingConfig2d, RmsNorm,
    RmsNormConfig,
};
use burn::tensor::{Tensor, TensorData};
use half::bf16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::PathBuf;

use crate::{B, Device};

/// Holds memory-mapped safetensors files and parsed headers.
pub struct TensorStore {
    mmaps: Vec<Mmap>,
}

impl TensorStore {
    /// Load and memory-map one or more safetensors files.
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        for path in paths {
            let file = File::open(path)
                .with_context(|| format!("Failed to open safetensors: {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };
            mmaps.push(mmap);
        }
        Ok(Self { mmaps })
    }

    /// Create a Tensors accessor borrowing from this store.
    pub fn tensors(&self) -> Result<Tensors<'_>> {
        let mut stores = Vec::with_capacity(self.mmaps.len());
        for mmap in &self.mmaps {
            stores.push(SafeTensors::deserialize(mmap)?);
        }
        Ok(Tensors { stores })
    }
}

/// Parsed safetensors with tensor access (borrows from TensorStore).
pub struct Tensors<'a> {
    stores: Vec<SafeTensors<'a>>,
}

impl<'a> Tensors<'a> {
    /// Get a tensor view by name, searching all shards.
    fn get_view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'a>> {
        for store in &self.stores {
            if let Ok(view) = store.tensor(name) {
                return Ok(view);
            }
        }
        bail!("Tensor not found: {name}");
    }

    /// Load a tensor as f32, converting from BF16/F16/F32 as needed.
    fn load_f32_data(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let view = self.get_view(name)?;
        let shape = view.shape().to_vec();
        let data = view.data();
        let dtype = view.dtype();

        let floats = match dtype {
            safetensors::Dtype::BF16 => data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F16 => data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => bail!("Unsupported dtype {other:?} for tensor {name}"),
        };

        Ok((floats, shape))
    }

    /// Load a tensor with the given const dimension.
    pub fn load_tensor<const D: usize>(
        &self,
        name: &str,
        device: &Device,
    ) -> Result<Tensor<B, D>> {
        let (floats, shape) = self.load_f32_data(name)?;
        let td = TensorData::new(floats, shape);
        Ok(Tensor::<B, D>::from_data(td, device))
    }

    /// Load a linear layer (with bias). Transposes weight from [d_out, d_in] to [d_in, d_out].
    pub fn load_linear(&self, prefix: &str, device: &Device) -> Result<Linear<B>> {
        let weight = self.load_tensor::<2>(&format!("{prefix}.weight"), device)?;
        let weight = weight.transpose(); // [d_out, d_in] → [d_in, d_out]
        let bias = self.load_tensor::<1>(&format!("{prefix}.bias"), device)?;
        Ok(Linear {
            weight: Param::initialized(ParamId::new(), weight),
            bias: Some(Param::initialized(ParamId::new(), bias)),
        })
    }

    /// Load a linear layer without bias. Transposes weight.
    pub fn load_linear_no_bias(&self, prefix: &str, device: &Device) -> Result<Linear<B>> {
        let weight = self.load_tensor::<2>(&format!("{prefix}.weight"), device)?;
        let weight = weight.transpose();
        Ok(Linear {
            weight: Param::initialized(ParamId::new(), weight),
            bias: None,
        })
    }

    /// Load a Conv2d layer. No transpose needed (both PyTorch and burn use [out, in, kH, kW]).
    pub fn load_conv2d(
        &self,
        prefix: &str,
        stride: [usize; 2],
        kernel_size: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        device: &Device,
    ) -> Result<Conv2d<B>> {
        let weight = self.load_tensor::<4>(&format!("{prefix}.weight"), device)?;
        let bias = self.load_tensor::<1>(&format!("{prefix}.bias"), device)?;
        Ok(Conv2d {
            weight: Param::initialized(ParamId::new(), weight),
            bias: Some(Param::initialized(ParamId::new(), bias)),
            stride,
            kernel_size,
            dilation,
            groups,
            padding: Ignored(PaddingConfig2d::Explicit(1, 1)),
        })
    }

    /// Load a LayerNorm. Maps weight→gamma, bias→beta.
    pub fn load_layer_norm(
        &self,
        prefix: &str,
        epsilon: f64,
        device: &Device,
    ) -> Result<LayerNorm<B>> {
        let gamma = self.load_tensor::<1>(&format!("{prefix}.weight"), device)?;
        let beta = self.load_tensor::<1>(&format!("{prefix}.bias"), device)?;
        let d_model = gamma.dims()[0];
        let mut ln = LayerNormConfig::new(d_model)
            .with_epsilon(epsilon)
            .init::<B>(device);
        ln.gamma = Param::initialized(ParamId::new(), gamma);
        ln.beta = Some(Param::initialized(ParamId::new(), beta));
        Ok(ln)
    }

    /// Load an RmsNorm. Maps weight→gamma.
    pub fn load_rms_norm(
        &self,
        prefix: &str,
        epsilon: f64,
        device: &Device,
    ) -> Result<RmsNorm<B>> {
        let gamma = self.load_tensor::<1>(&format!("{prefix}.weight"), device)?;
        let d_model = gamma.dims()[0];
        let mut rn = RmsNormConfig::new(d_model)
            .with_epsilon(epsilon)
            .init::<B>(device);
        rn.gamma = Param::initialized(ParamId::new(), gamma);
        Ok(rn)
    }

    /// Load an Embedding. No transpose needed.
    pub fn load_embedding(&self, prefix: &str, device: &Device) -> Result<Embedding<B>> {
        let weight = self.load_tensor::<2>(&format!("{prefix}.weight"), device)?;
        Ok(Embedding {
            weight: Param::initialized(ParamId::new(), weight),
        })
    }
}
