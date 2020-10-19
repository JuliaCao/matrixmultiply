pub use matrixmultiply::dgemm;
pub use matrixmultiply::ref_dgemm;
use rand::Rng;
use std::time::Instant;

use std::os::raw::c_int;

#[allow(non_camel_case_types)]
type blas_index = c_int; // blas index type

// 2.3 GHz * 8 vector width * 2 flops for FMA = 36.8 GF/s
static MAX_SPEED: f64 = 36.8;

fn main() {
    let n_sizes = [
        31, 32, 96, 97, 127, 128, 129, 229, 255, 256, 257, 319, 320, 321, 512, 767, 768, 769,
    ];
    let nsizes = n_sizes.len();
    let mut rng = rand::thread_rng();

    let mut Mflops_s = vec![0.; nsizes];
    let mut per = vec![0.; nsizes];

    for (i, n) in n_sizes.iter().enumerate() {
        let mut Gflops_s: f64 = -1.0;
        let mut seconds: f64 = -1.0;
        let timeout: f64 = 5.;

        let mat_a = vec![rng.gen::<f64>(); n * n];
        let mat_b = vec![rng.gen::<f64>(); n * n];
        let mut mat_c = vec![rng.gen::<f64>(); n * n];
        let mut mat_c_dgemm = mat_c.clone();

        let ptr_a = mat_a.as_ptr();
        let ptr_b = mat_b.as_ptr();
        let ptr_c = mat_c_dgemm.as_mut_ptr();

        // warm up
        unsafe {
            dgemm(*n, ptr_a, ptr_b, ptr_c);
            ref_dgemm(*n as usize, &mat_a, &mat_b, &mut mat_c);
            check_vec_eq(&mat_c, ptr_c, *n);
        }

        let mut n_iter: f64 = 1.;
        while seconds < timeout {
            n_iter *= 2.;

            let now = Instant::now();
            for _ in 0..n_iter as usize {
                unsafe {
                    dgemm(*n, ptr_a, ptr_b, ptr_c);
                    //ref_dgemm(*n as usize, &mat_a, &mat_b, &mut mat_c);

                    /* blas::dgemm(
                        b'N',
                        b'N',
                        *n as blas_index, // m, rows of Op(a)
                        *n as blas_index, // n, cols of Op(b)
                        *n as blas_index, // k, cols of Op(a)
                        1.,
                        &mat_a,
                        *n as i32, // lda
                        &mat_b,
                        *n as i32, // ldb
                        0.,        // beta
                        &mut mat_c,
                        *n as i32, // ldc
                    ) */
                }
            }
            seconds = Instant::now().duration_since(now).as_secs_f64();
            //println!("n_iter: {}, elapsed time: {}", n_iter, seconds);
            let n = *n as f64;
            Gflops_s = 2e-9 * n_iter * n * n * n / seconds;
        }
        Mflops_s[i] = Gflops_s * 1000.;
        per[i] = Gflops_s * 100. / MAX_SPEED;
        println!(
            "Size: {}\tMflop/s:{}\tPercentage:{}\n",
            n, Mflops_s[i], per[i]
        );

        let mut averper = 0.;
        for (i, _) in n_sizes.iter().enumerate() {
            averper += per[i];
        }
        averper /= nsizes as f64 * 1.;
        println!("Average percentage of Peak = {}\n", averper);
    }
}

unsafe fn check_vec_eq(v1: &Vec<f64>, v2: *const f64, i: usize) {
    for i in 0..i * i {
        assert_eq!(v1[i], *v2.offset(i as isize));
    }
}

/* print a col major matrix */
fn print_matrx(mat: &Vec<f64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            print!("{} ", mat[i + j * n]);
        }
        println!("")
    }
}

#[cfg(test)]
mod tests {
    pub use matrixmultiply::dgemm;
    pub use matrixmultiply::ref_dgemm;
    use rand::Rng;
}
