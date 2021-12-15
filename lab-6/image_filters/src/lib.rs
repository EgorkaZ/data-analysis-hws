
use numpy::{PyArray3, PyReadonlyArray3, PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyBool};

mod filter;
mod gaussian;
mod pub_utils;

#[pyfunction]
#[pyo3(text_signature = "gaussian_blur(images, inline, /)")]
fn gaussian_blur<'py>(
    gil: Python<'py>,
    images: &'py PyArray3<u8>,
    inline: &'py PyBool,
) -> PyResult<&'py PyArray3<u8>>
{
    filter::apply_filter_py(gil, images, inline, &gaussian::GaussianFilter{})
}

#[pyfunction]
#[pyo3(text_signature = "gaussian_blur(images, filter, inline, /)")]
fn apply_filter<'py>(
    gil: Python<'py>,
    images: &'py PyArray3<u8>,
    filter: &'py PyArray2<i64>,
    inline: &'py PyBool,
) -> PyResult<&'py PyArray3<u8>>
{
    let shape = filter.shape();
    assert_eq!(shape.len(), 2);

    let filter = unsafe { filter.as_slice_mut() }?.to_vec();
    let divisor = filter.iter().cloned().sum();

    let divisor = if divisor == 0 { 1 } else { divisor };
    let filter = filter::CustomFilter::new(filter, (shape[0], shape[1]), divisor);

    filter::apply_filter_py(gil, images, inline, &filter)
}

#[pyfunction]
#[pyo3(text_signature = "to_grayscale(image, /)")]
pub fn to_grayscale<'py>(
    gil: Python<'py>,
    rgb: PyReadonlyArray3<'py, u8>
) -> PyResult<&'py PyArray2<u8>>
{
    pub_utils::to_grayscale_py(gil, rgb)
}

#[pyfunction]
#[pyo3(text_signature = "threshold_to_bin(image, threshold, /)")]
pub fn threshold_to_bin<'py>(
    gil: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    threshold: i32
) -> PyResult<&'py PyArray2<u8>>
{
    pub_utils::threshold_to_bin_py(gil, image, threshold)
}

/// A Python module implemented in Rust.
#[pymodule]
fn image_filters(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(to_grayscale, m)?)?;
    m.add_function(wrap_pyfunction!(apply_filter, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_to_bin, m)?)?;
    Ok(())
}
