pub use matrixmultiply::dgemm;
pub use matrixmultiply::ref_dgemm;
use rand::Rng;

static I: usize = 256;

fn main() {
    let mut rng = rand::thread_rng();

    let mat_a = vec![rng.gen::<f64>(); I * I];
    let mat_b = vec![rng.gen::<f64>(); I * I];
    let mut mat_c = vec![rng.gen::<f64>(); I * I];

    //println!("Matrix A:");
    //print_matrx(&mat_a, I);
    //println!("Matrix B:");
    //print_matrx(&mat_b, I);
    //println!("Matrix C:");
    //print_matrx(&mat_c, I);

    let mut mat_c_dgemm = mat_c.clone();
    assert_eq!(mat_c_dgemm[0], mat_c[0]);

    let len = mat_c_dgemm.len();
    let cap = mat_c_dgemm.capacity();
    assert_eq!(len, I * I);
    assert_eq!(cap, I * I);

    ref_dgemm(I as usize, &mat_a, &mat_b, &mut mat_c);

    let ptr_a = mat_a.as_ptr();
    let ptr_b = mat_b.as_ptr();
    let ptr_c = mat_c_dgemm.as_mut_ptr();

    unsafe {
        dgemm(I, ptr_a, ptr_b, ptr_c);
        correctness_check(&mat_c, ptr_c);
    }
}

unsafe fn correctness_check(v1: &Vec<f64>, v2: *const f64) {
    for i in 0..I * I {
        assert_eq!(v1[i], *v2.offset(i as isize));
    }
}

/* print a col major matrix */
fn print_matrx(mat: &Vec<f64>, n: usize) {
    for i in 0..I {
        for j in 0..I {
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
