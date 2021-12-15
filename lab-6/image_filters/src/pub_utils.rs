use ndarray::{Array2};
use numpy::{PyReadonlyArray3, PyArray2, PyReadonlyArray2};
use pyo3::{Python, PyResult};

fn to_grayscale(
    rgb: &[u8],
) -> Vec<u8>
{
    rgb
        .chunks(3)
        .map(|p| {
            assert_eq!(p.len(), 3);
            let (r, g, b) = (p[0], p[1], p[2]);
            if r < g && g < b {
                g
            } else if g < r && r < b {
                r
            } else {
                b
            }

        })
        .collect()
}

pub fn to_grayscale_py<'py>(
    gil: Python<'py>,
    rgb: PyReadonlyArray3<'py, u8>
) -> PyResult<&'py PyArray2<u8>>
{
    let shape = rgb.shape();
    assert_eq!(shape.len(), 3);

    let rgb = rgb.as_slice()?;
    let gray = to_grayscale(rgb);

    let arr = Array2::from_shape_vec((shape[0], shape[1]), gray)
        .expect("Couldn't turn vec into arr");
    Ok(PyArray2::from_owned_array(gil, arr))
}

fn threshold_to_bin(
    image: &[u8],
    threshold: u8
) -> Vec<u8>
{
    image.iter()
        .cloned()
        .map(|pixel| if pixel > threshold { 255 } else { 0 })
        .collect()
}

pub fn threshold_to_bin_py<'py>(
    gil: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    threshold: i32
) -> PyResult<&'py PyArray2<u8>>
{
    let shape = image.shape();
    let pixels = image.as_slice()?;

    assert!(threshold >= 0);
    assert!(threshold < 256);
    let threshold = threshold as u8;
    let bin = threshold_to_bin(pixels, threshold);

    let np_arr = Array2::from_shape_vec((shape[0], shape[1]), bin)
        .expect("Couldn't turn vec into arr");
    Ok(PyArray2::from_owned_array(gil, np_arr))
}