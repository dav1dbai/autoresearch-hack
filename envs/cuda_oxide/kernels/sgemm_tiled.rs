// Starter kernel for the cuda-oxide GEMM env (the editable surface).
// Contract: this module MUST expose `sgemm_tiled(m,n,k,alpha,a,b,beta,c)` with this
// exact signature — the fixed host harness launches it on a 16x16 grid. Optimize the
// BODY (tiling, unrolling, register blocking, vectorization); keep the signature.
const TILE_SIZE: usize = 16;
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn sgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;
        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;
        let num_tiles = k_size.div_ceil(TILE_SIZE);

        let mut sum = 0.0f32;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;
            let smem_idx = ty * TILE_SIZE + tx;
            unsafe {
                let a_col = tile_start + tx;
                if row < m_size && a_col < k_size {
                    TILE_A[smem_idx] = a[row * k_size + a_col];
                } else {
                    TILE_A[smem_idx] = 0.0;
                }
                let b_row = tile_start + ty;
                if b_row < k_size && col < n_size {
                    TILE_B[smem_idx] = b[b_row * n_size + col];
                } else {
                    TILE_B[smem_idx] = 0.0;
                }
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    sum += TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx];
                    i += 1;
                }
            }
            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size
                && let Some(c_elem) = c.get_mut(c_idx)
            {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
