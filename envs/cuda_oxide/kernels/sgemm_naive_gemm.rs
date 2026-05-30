// Starter kernel for the cuda-oxide naive GEMM env (the editable surface).
// This is the VERBATIM sgemm_naive kernel from the cuda-oxide `gemm` example
// (crates/rustc-codegen-cuda/examples/gemm/src/main.rs) — real NVIDIA shipped code.
//
// CONTRACT (fixed — host launches it; do NOT change the signature):
//   sgemm_naive(m, n, k, alpha, a, b, beta, c) — host launches on 16x16 blocks.
//   Optimize the BODY (tiling, shared mem, register blocking, vectorization).
#[cuda_module]
mod kernels {
    use super::*;

    /// Naive GEMM kernel: C = alpha * A * B + beta * C
    ///
    /// Each thread computes ONE element of C.
    /// Matrix layout: Row-major
    /// - A is M x K
    /// - B is K x N
    /// - C is M x N
    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < m as usize {
                let n_size = n as usize;
                let k_size = k as usize;

                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_size {
                    sum += a[row * k_size + i] * b[i * n_size + col];
                    i += 1;
                }

                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }
}
