use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
extern crate matrixmultiply;
pub use matrixmultiply::dgemm;

// naive 3 loop dgemm
fn ref_dgemm(lda: usize, a: &Vec<f64>, vec_b: &Vec<f64>, c: &mut Vec<f64>) {
    for i in 0..lda {
        for j in 0..lda {
            let mut cij = c[i + j * lda];
            for k in 0..lda {
                cij += a[k + i * lda] * vec_b[k + j * lda];
            }
            c[i + j * lda] = cij;
        }
    }
}

fn bench_dgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("DGEMM");

    for i in [16, 256].iter() {
        let mat_a = vec![0.; i * i];
        let mat_b = vec![0.; i * i];
        let mut mat_c = vec![0.; i * i];
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
