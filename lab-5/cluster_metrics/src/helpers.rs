use std::ops::AddAssign;

#[derive(Clone, Copy)]
pub struct Point(f64, f64, i32);

impl Point
{
    pub fn new(x: f64, y: f64, cluster: i32) -> Self
    { Point(x, y, cluster) }

    pub fn cluster(&self) -> i32
    { self.2 }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Dist(pub f64);

impl AddAssign for Dist
{
    fn add_assign(&mut self, Dist(rhs): Self)
    {
        self.0 += rhs
    }
}

pub fn distance(Point(x1, y1,..): Point, Point(x2, y2,..): Point) -> Dist
{
    let x = x2 - x1;
    let y = y2 - y1;
    Dist((x * x + y * y).sqrt())
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AvgDistAccum
{
    dist_sum: Dist,
    count: usize,
}

impl AvgDistAccum
{
    pub fn get_avg(&self) -> Dist
    {
        let Dist(dist) = self.dist_sum;
        Dist(dist / self.count as f64)
    }
}

impl AddAssign<Dist> for AvgDistAccum
{
    fn add_assign(&mut self, rhs: Dist)
    {
        self.dist_sum += rhs;
        self.count += 1;
    }
}

impl AddAssign for AvgDistAccum
{
    fn add_assign(&mut self, Self{ dist_sum: r_sum, count: r_count }: Self)
    {
        self.dist_sum += r_sum;
        self.count += r_count;
    }
}
