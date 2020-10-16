// naive 3 loop dgemm
pub fn ref_dgemm(lda: usize, a: &Vec<f64>, b: &Vec<f64>, c: &mut Vec<f64>) {
    /* For each row i of A */
    for i in 0..lda {
        /* For each column j of B */
        for j in 0..lda {
            let mut cij = c[i + j * lda];
            for k in 0..lda {
                /* Compute C(i,j) */
                cij += a[i + k * lda] * b[k + j * lda];
            }
            c[i + j * lda] = cij;
        }
    }
}
