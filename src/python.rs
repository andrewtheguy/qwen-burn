use crate::{QwenAsr as RustQwenAsr, DEFAULT_MODEL_ID, SUPPORTED_LANGUAGES};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::sync::Mutex;

#[pyclass]
struct QwenAsr {
    inner: Mutex<RustQwenAsr<burn_ndarray::NdArray<f32>>>,
}

#[pymethods]
impl QwenAsr {
    #[new]
    #[pyo3(signature = (model_id=None))]
    fn new(model_id: Option<&str>) -> PyResult<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL_ID);
        let dev = burn_ndarray::NdArrayDevice::Cpu;
        let model = RustQwenAsr::<burn_ndarray::NdArray>::load_on(model_id, &dev)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(model),
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
#[pyo3(name = "qwen_burn")]
fn qwen_burn(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<QwenAsr>()?;
    module.add("DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)?;
    module.add(
        "SUPPORTED_LANGUAGES",
        SUPPORTED_LANGUAGES.to_vec(),
    )?;
    Ok(())
}
