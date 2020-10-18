use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
extern crate matrixmultiply;
pub use matrixmultiply::dgemm;
pub use matrixmultiply::ref_dgemm;
use rand::Rng;

#[macro_use]
extern crate blas;

use std::os::raw::c_int;

#[allow(non_camel_case_types)]
type blas_index = c_int; // blas index type

fn bench_dgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("DGEMM");
    let mut rng = rand::thread_rng();

    for i in [
        31, 32, 96, 97, 127, 128, 129, 229, 255, 256, 257, 319, 320, 321, 512, 767, 768, 769,
    ]
    .iter()
    {
        let mat_a = vec![rng.gen::<f64>(); i * i];
        let mat_b = vec![rng.gen::<f64>(); i * i];
        let mut mat_c = vec![rng.gen::<f64>(); i * i];
        group.bench_function(BenchmarkId::new("naive", i), |b| {
            b.iter(|| ref_dgemm(*i as usize, &mat_a, &mat_b, &mut mat_c))
        });

        group.bench_function(BenchmarkId::new("block-w-optimized-algo", i), |b| {
            b.iter(|| unsafe {
                dgemm(
                    *i as usize,
                    mat_a.as_ptr(),
                    mat_b.as_ptr(),
                    mat_c.as_mut_ptr(),
                )
            })
        });

        group.bench_function(BenchmarkId::new("blas", i), |b| {
            b.iter(|| unsafe {
                blas::dgemm(
                    b'N',
                    b'N',
                    *i as blas_index, // m, rows of Op(a)
                    *i as blas_index, // n, cols of Op(b)
                    *i as blas_index, // k, cols of Op(a)
                    1.,
                    &mat_a,
                    *i as i32, // lda
                    &mat_b,
                    *i as i32, // ldb
                    0.,        // beta
                    &mut mat_c,
                    *i as i32, // ldc
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dgemm);
criterion_main!(benches);
