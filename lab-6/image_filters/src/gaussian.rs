use crate::filter::Filter as MyFilter;

pub struct GaussianFilter
{
}

impl MyFilter for GaussianFilter
{
    fn filter(&self) -> &[i64]
    {
        &[
            1,  4,  7,  4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1,  4,  7,  4, 1,
        ]
    }

    fn divisor(&self) -> i64
    { 273 }

    fn shape(&self) -> (usize, usize) {
        (5, 5)
    }
}
