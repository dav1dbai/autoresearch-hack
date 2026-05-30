// Naive baseline kernel — correct but slow (global-memory reads, no shared-memory
// tiling). Same `sgemm_tiled` signature/launch as the starter; used to confirm the
// env discriminates a poor kernel from a good one (the optimization gradient).
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
        let row = thread::blockIdx_y() as usize * TILE_SIZE + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + thread::threadIdx_x() as usize;
        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let mut sum = 0.0f32;
        if row < m_size && col < n_size {
            let mut kk = 0usize;
            while kk < k_size {
                sum += a[row * k_size + kk] * b[kk * n_size + col];
                kk += 1;
            }
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
