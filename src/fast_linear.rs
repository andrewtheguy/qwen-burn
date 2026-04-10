use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::weights::Tensors;

pub struct FastLinear<B: Backend> {
    weight_2d: Tensor<B, 2>, // [in, out], forced contiguous
    weight_3d: Tensor<B, 3>, // [1, in, out], forced contiguous
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> FastLinear<B> {
    pub fn load(tensors: &Tensors, prefix: &str, device: &B::Device) -> anyhow::Result<Self> {
        Self::load_inner(tensors, prefix, device, true)
    }

    pub fn load_no_bias(
        tensors: &Tensors,
        prefix: &str,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        Self::load_inner(tensors, prefix, device, false)
    }

    fn load_inner(
        tensors: &Tensors,
        prefix: &str,
        device: &B::Device,
        with_bias: bool,
    ) -> anyhow::Result<Self> {
        let weight = tensors.load_tensor::<B, 2>(&format!("{prefix}.weight"), device)?;
        let weight = weight.transpose();
        let [d_in, d_out] = weight.dims();
        let weight_2d = weight.reshape([d_in, d_out]);
        let weight_3d = weight_2d.clone().reshape([1, d_in, d_out]);
        let bias = if with_bias {
            Some(tensors.load_tensor::<B, 1>(&format!("{prefix}.bias"), device)?)
        } else {
            None
        };

        Ok(Self {
            weight_2d,
            weight_3d,
            bias,
        })
    }

    pub fn forward2d(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input.matmul(self.weight_2d.clone());
        if let Some(bias) = &self.bias {
            output = output + bias.clone().unsqueeze::<2>();
        }
        output
    }

    pub fn forward3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = input.matmul(self.weight_3d.clone());
        if let Some(bias) = &self.bias {
            output = output + bias.clone().reshape([1, 1, bias.dims()[0]]);
        }
        output
    }
}
