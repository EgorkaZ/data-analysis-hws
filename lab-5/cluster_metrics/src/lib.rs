mod calcs;
mod helpers;

use helpers::Dist;

extern crate pyo3;

use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyTuple};

#[pyfunction]
fn count_metrics<'py>(
    gil: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    clusters: PyReadonlyArray1<'py, i32>) -> PyResult<&'py PyTuple>
{
    let xs = xs.as_slice()?;
    let ys = ys.as_slice()?;
    let clusters = clusters.as_slice()?;

    let (Dist(for_same), Dist(for_diff)) = calcs::count_metrics(xs, ys, clusters);
    Ok(PyTuple::new(gil, [for_same, for_diff]))
}

/// A Python module implemented in Rust.
#[pymodule]
fn cluster_metrics(_gil: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_metrics, m)?)?;
    Ok(())
}
