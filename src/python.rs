use crate::{QwenAsr as RustQwenAsr, DEFAULT_MODEL_ID, SUPPORTED_LANGUAGES};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::sync::Mutex;

enum Inner {
    Gpu(RustQwenAsr<burn_wgpu::Wgpu<f32, i32>>),
    Cpu(RustQwenAsr<burn_cpu::Cpu<f32, i32>>),
}

impl Inner {
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: Option<&str>,
    ) -> anyhow::Result<String> {
        match self {
            Inner::Gpu(m) => m.transcribe(samples, language, context),
            Inner::Cpu(m) => m.transcribe(samples, language, context),
        }
    }
}

#[pyclass]
struct QwenAsr {
    inner: Mutex<Inner>,
}

#[pymethods]
impl QwenAsr {
    #[new]
    #[pyo3(signature = (model_id=None, device="auto"))]
    fn new(model_id: Option<&str>, device: &str) -> PyResult<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL_ID);
        let inner = match device.to_lowercase().as_str() {
            "cpu" => {
                let dev = burn_cpu::CpuDevice::default();
                let model = RustQwenAsr::<burn_cpu::Cpu>::load_on(model_id, &dev)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Inner::Cpu(model)
            }
            "auto" | "metal" | "mps" | "gpu" => {
                let dev = burn_wgpu::WgpuDevice::DefaultDevice;
                let model = RustQwenAsr::<burn_wgpu::Wgpu>::load_on(model_id, &dev)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Inner::Gpu(model)
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown device: {}. Supported: auto, cpu, gpu/metal",
                    device
                )));
            }
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[pyo3(signature = (samples, *, language=None, context=None))]
    fn transcribe(
        &self,
        py: Python<'_>,
        samples: PyReadonlyArray1<'_, f32>,
        language: Option<&str>,
        context: Option<&str>,
    ) -> PyResult<String> {
        let samples = samples.as_slice()?.to_vec();
        let language = language.map(|s| s.to_string());
        let context = context.map(|s| s.to_string());

        py.detach(|| {
            self.inner
                .lock()
                .unwrap()
                .transcribe(
                    &samples,
                    language.as_deref(),
                    context.as_deref(),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

#[pymodule]
#[pyo3(name = "qwencandle")]
fn qwencandle(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<QwenAsr>()?;
    module.add("DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)?;
    module.add(
        "SUPPORTED_LANGUAGES",
        SUPPORTED_LANGUAGES.to_vec(),
    )?;
    Ok(())
}
