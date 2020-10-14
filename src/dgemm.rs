use maligned::{align_first, A64};
use std::cmp::min;
use std::ptr;

static BLOCK_SIZE: usize = 8;

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda*lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
pub unsafe fn dgemm(lda: usize, a: *const f64, b: *const f64, c: *mut f64) {
    let mut block_a: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE);
    let mut block_b: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE);
    let mut block_c: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE);

    for i in (0..lda).step_by(BLOCK_SIZE) {
        for j in (0..lda).step_by(BLOCK_SIZE) {
            let m = min(BLOCK_SIZE, lda - i);
            let n = min(BLOCK_SIZE, lda - j);
            let offset = i + j * lda;
            copy_col_major(
                c.offset(offset as isize),
                BLOCK_SIZE,
                n,
                m,
                lda,
                block_c.as_mut_ptr(),
            );

            for k in (0..lda).step_by(BLOCK_SIZE) {
                let k = min(BLOCK_SIZE, lda - k);
                let a_offset = i + k * lda;
                let b_offset = k + j * lda;
                copy_col_major(
                    a.offset(a_offset as isize),
                    BLOCK_SIZE,
                    k,
                    m,
                    lda,
                    block_a.as_mut_ptr(),
                );
                copy_col_major(
                    b.offset(b_offset as isize),
                    BLOCK_SIZE,
                    n,
                    k,
                    lda,
                    block_b.as_mut_ptr(),
                );
            }
        }
    }
}

unsafe fn copy_col_major(
    x: *const f64,
    n: usize,
    right: usize,
    down: usize,
    l: usize,
    temp: *mut f64,
) {
    // for each column of x
    if down < n || right < n {
        for i in 0..n {
            for j in 0..n {
                if j >= down {
                    let offset = i * n + j;
                    ptr::write(temp.offset(offset as isize), 0f64);
                } else if i >= right {
                    //temp[i * n + j] = 0;
                    let offset = i * n + j;
                    ptr::write(temp.offset(offset as isize), 0f64);
                } else {
                    //temp[i * n + j] = x[i * l + j];
                    let offset = i * l + j;
                    ptr::write(temp.offset(offset as isize), 0f64);
                }
            }
        }
    } else {
        for i in 0..n {
            for j in (0..n).step_by(16) {
                //use avx
            }
        }
    }
}
