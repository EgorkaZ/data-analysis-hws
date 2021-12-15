use std::iter;

use ndarray::{ArrayViewMut2, Axis};
use numpy::PyArray3;
use pyo3::{Python, types::PyBool, PyResult};

pub trait Filter
{
    fn apply(&self, mut img: ArrayViewMut2<'_, u8>)
    {
        let filter = self.filter();


        let (cy, cx) = self.shape();
        assert_eq!(cy % 2, 1);
        assert_eq!(cx % 2, 1);
        let (cy, cx) = (cy as i64 / 2, cx as i64 / 2);

        let (h, w) = img.dim();

        let mut spreaded: Vec<i64> = Vec::new();

        for i in 0..h {
            for j in 0..w {
                let curr: i64 = (-cy..cy)
                    .flat_map(|sh_y| iter::repeat(sh_y).zip(-cx..cx))
                    .zip(filter.iter().cloned())
                    .filter_map(|((sh_y, sh_x), f_pix)| {
                        let y = (sh_y + i as i64) as usize;
                        let x = (sh_x + j as i64) as usize;
                        img.get((y, x))
                            .cloned()
                            .map(|pix| (pix as i64) * f_pix)
                    })
                    .sum();
                spreaded.push(curr / self.divisor());
            }
        }

        let min = *spreaded.iter().min().expect("Want a minimum");
        let max = *spreaded.iter().max().expect("Want a maximum");

        let spread = max - min;

        spreaded.iter_mut()
            .for_each(|curr| {
                *curr -= min;
                *curr *= 255;
                *curr /= spread;
            });

        let img_slice = img.as_slice_mut().expect("Image is not a slice");
        assert_eq!(img_slice.len(), spreaded.len());

        for (pixel, counted) in img_slice.iter_mut().zip(spreaded.iter().cloned()) {
            assert!(counted >= 0);
            assert!(counted < 256);
            *pixel = counted as u8;
        }
    }

    fn filter(&self) -> &[i64];

    fn shape(&self) -> (usize, usize);

    fn divisor(&self) -> i64;
}

pub struct CustomFilter
{
    filter: Vec<i64>,
    shape: (usize, usize),
    divisor: i64
}

impl CustomFilter
{
    pub fn new(filter: Vec<i64>, shape: (usize, usize), divisor: i64) -> Self
    { CustomFilter{ filter, shape, divisor } }
}

impl Filter for CustomFilter
{
    fn filter(&self) -> &[i64] {
        &self.filter
    }

    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn divisor(&self) -> i64 {
        self.divisor
    }
}

pub fn apply_filter_py<'py, F: Filter>(
    gil: Python<'py>,
    images: &'py PyArray3<u8>,
    inline: &'py PyBool,
    filter: &F
) -> PyResult<&'py PyArray3<u8>>
{
    let images_ret;
    if !inline.is_true() {
        println!("{:?}", images.dims());
        let cloned = PyArray3::new(gil, images.dims(), images.is_fortran_contiguous());
        images.copy_to(cloned)?;

        images_ret = cloned;
    } else {
        images_ret = images;
    }

    {
        let mut images_ref = unsafe { images_ret.as_array_mut() };

        images_ref.axis_iter_mut(Axis(0))
            .for_each(|img| filter.apply(img))
    }

    Ok(images_ret)
}
