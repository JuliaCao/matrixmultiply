use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
extern crate matrixmultiply;
pub use matrixmultiply::dgemm;
pub use matrixmultiply::ref_dgemm;
use rand::Rng;

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
        group.bench_function(BenchmarkId::new("Naive", i), |b| {
            b.iter(|| ref_dgemm(*i as usize, &mat_a, &mat_b, &mut mat_c))
        });

        group.bench_function(BenchmarkId::new("Block", i), |b| {
            b.iter(|| unsafe {
                dgemm(
                    *i as usize,
                    mat_a.as_ptr(),
                    mat_b.as_ptr(),
                    mat_c.as_mut_ptr(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dgemm);
criterion_main!(benches);
