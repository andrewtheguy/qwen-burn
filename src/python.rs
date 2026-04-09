use crate::{QwenAsr as RustQwenAsr, DEFAULT_MODEL_ID, SUPPORTED_LANGUAGES, Device};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::sync::Mutex;

fn parse_device(device: &str) -> PyResult<Device> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" | "gpu" => Ok(Device::DiscreteGpu(0)),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown device: {}. Supported: cpu, metal/gpu",
            device
        ))),
    }
}

#[pyclass]
struct QwenAsr {
    inner: Mutex<RustQwenAsr>,
}

#[pymethods]
impl QwenAsr {
    #[new]
    #[pyo3(signature = (model_id=None, device="cpu"))]
    fn new(model_id: Option<&str>, device: &str) -> PyResult<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL_ID);
        let device = parse_device(device)?;
        let inner = RustQwenAsr::load_on(model_id, &device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        // Copy data to owned types before releasing the GIL
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
