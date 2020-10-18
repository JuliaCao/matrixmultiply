use maligned::{align_first, A64};
use std::cmp::min;
use std::ptr;

static BLOCK_SIZE: usize = 8;

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda*lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
pub unsafe fn dgemm(lda: usize, a: *const f64, b: *const f64, c: *mut f64) {
    let mut block_a: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE * BLOCK_SIZE);
    let mut block_b: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE * BLOCK_SIZE);
    let mut block_c: Vec<f64> = align_first::<f64, A64>(BLOCK_SIZE * BLOCK_SIZE);

    //println!("passed in val for a {}", *(a.offset(0)));
    //println!("passed val for b {}", *(b.offset(0)));
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
                let p = min(BLOCK_SIZE, lda - k);
                let a_offset = i + k * lda;
                let b_offset = k + j * lda;
                copy_col_major(
                    a.offset(a_offset as isize),
                    BLOCK_SIZE,
                    p,
                    m,
                    lda,
                    block_a.as_mut_ptr(),
                );
                copy_col_major(
                    b.offset(b_offset as isize),
                    BLOCK_SIZE,
                    n,
                    p,
                    lda,
                    block_b.as_mut_ptr(),
                );
                do_block_with_optimized_algorithm(
                    BLOCK_SIZE,
                    block_a.as_mut_ptr(),
                    block_b.as_mut_ptr(),
                    block_c.as_mut_ptr(),
                );
                let offset = i + j * lda;
                unpad_col_to_col_dst(
                    block_c.as_ptr(),
                    BLOCK_SIZE,
                    n,
                    m,
                    c.offset(offset as isize),
                    lda,
                );
            }
        }
    }
}

unsafe fn unpad_col_to_col_dst(
    x: *const f64,
    n: usize,
    right: usize,
    down: usize,
    dst: *mut f64,
    col: usize,
) {
    for i in 0..n {
        for j in 0..n {
            if !(i >= down || j >= right) {
                let dst_offset = j * col + i;
                let x_offset = i + n * j;
                ptr::write(
                    dst.offset(dst_offset as isize),
                    *(x.offset(x_offset as isize)),
                );
            }
        }
    }
}

unsafe fn copy_row_major(
    x: *const f64,
    n: usize,
    right: usize,
    down: usize,
    l: usize,
    temp: *mut f64,
) {
    // for each col of x
    for i in 0..n {
        // for each row of x
        for j in 0..n {
            let offset = j * n + i;
            if j >= down {
                ptr::write(temp.offset(offset as isize), 0f64);
            } else if i >= right {
                ptr::write(temp.offset(offset as isize), 0f64);
            } else {
                let x_offset = i * l + j;
                //println!("offset: {}", offset);
                //println!("x_offset: {}", x_offset);
                //println!("x value: {}", *(x.offset(x_offset as isize)));
                ptr::write(temp.offset(offset as isize), *(x.offset(x_offset as isize)));
            }
        }
    }
}

unsafe fn do_block(lda: usize, a: *mut f64, b: *mut f64, c: *mut f64) {
    // for each row i of A
    for i in 0..lda {
        // for each col j of b
        for j in 0..lda {
            let offset = i + j * lda;
            let mut cij = *(c.offset(offset as isize));
            for k in 0..lda {
                let a_offset = k + i * lda;
                let b_offset = k + j * lda;

                /*                if i == 0 && j == 0 && k == 0 {
                //println!("aik={}", *(a.offset(a_offset as isize)));
                //println!("bkj={}", *b.offset(b_offset as isize));
                //println!("cij={}", *c.offset(offset as isize));
                 */
                cij += *(a.offset(a_offset as isize)) * *(b.offset(b_offset as isize));

                //println!("cij after addtion = {}", cij);
                ptr::write(c.offset(offset as isize), cij);
                //}
            }
        }
    }
}

unsafe fn do_block_with_optimized_algorithm(lda: usize, a: *mut f64, b: *mut f64, c: *mut f64) {
    // for each col i of B
    for i in 0..lda {
        // for each col j of A / row j of B
        for j in 0..lda {
            let b_offset = j + i * lda;
            let bji = *(b.offset(b_offset as isize));
            // for each row k of A
            for k in 0..lda {
                let a_offset = k + j * lda;
                let c_offset = k + i * lda;
                let mut cki = *(c.offset(c_offset as isize));
                cki += *(a.offset(a_offset as isize)) * bji;

                ptr::write(c.offset(c_offset as isize), cki);
                //}
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
    for i in 0..n {
        for j in 0..n {
            let offset = i * n + j;
            if j >= down {
                ptr::write(temp.offset(offset as isize), 0f64);
            } else if i >= right {
                ptr::write(temp.offset(offset as isize), 0f64);
            } else {
                let x_offset = i * l + j;
                let value = *(x.offset(x_offset as isize));
                ptr::write(temp.offset(offset as isize), value);
            }
        }
    }
}
