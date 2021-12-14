use std::sync::Mutex;

use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSlice};

use crate::helpers::{AvgDistAccum, Dist, Point, distance};

pub fn count_metrics(xs: &[f64], ys: &[f64], clusters: &[i32]) -> (Dist, Dist)
{
    let points: Vec<_> = xs.iter().zip(ys.iter()).zip(clusters.iter())
        .map(|((x, y), c)| Point::new(*x, *y, *c))
        .enumerate()
        .collect();

    let in_same_acc = Mutex::new(AvgDistAccum::default());
    let in_diff_acc = Mutex::new(AvgDistAccum::default());

    let points = points.as_parallel_slice();
    points.par_iter().for_each(|(i, p1)| {
        let mut in_same_curr = AvgDistAccum::default();
        let mut in_diff_curr = AvgDistAccum::default();
        points.iter()
            .take(*i)
            .map(|(_j, p2)| *p2)
            .for_each(|p2| {
                if p1.cluster() == p2.cluster() {
                    in_same_curr += distance(*p1, p2);
                } else {
                    in_diff_curr += distance(*p1, p2);
                }
            });
        *in_same_acc.lock().unwrap() += in_same_curr;
        *in_diff_acc.lock().unwrap() += in_diff_curr;
    });

    let same_acc = in_same_acc.into_inner().unwrap();
    let diff_acc = in_diff_acc.into_inner().unwrap();

    (same_acc.get_avg(), diff_acc.get_avg())
}
