pub use matrixmultiply::dgemm;

fn main() {
    let i = 8;
    let mat_a = vec![0.; i * i];
    let mat_b = vec![0.; i * i];
    let mut mat_c = vec![0.; i * i];

    let ptr_a = mat_a.as_ptr();
    let ptr_b = mat_b.as_ptr();
    let ptr_c = mat_c.as_mut_ptr();

    unsafe {
        dgemm(
            i,
            ptr_a,
            ptr_b,
            ptr_c,
        )
    }
}