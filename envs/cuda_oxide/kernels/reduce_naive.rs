// Naive parallel reduction baseline — correct but deliberately slow.
// Pathology: each thread reads a CONTIGUOUS chunk, so adjacent threads in a warp
// touch addresses `chunk` apart — fully UNCOALESCED global loads. Since reduction is
// memory-bound, this (not the tree) is what kills bandwidth; the optimized kernel's
// coalesced grid-stride load is the real win the agent must keep/improve.
//
// CONTRACT (fixed — host launches it; do NOT change the signature):
//   reduce_sum(n, data, out) — `out` has exactly 1024 slots (one partial per block).
//   The host accumulates the partials on CPU. Launch: 1024 blocks × 256 threads.

const BLOCK_SIZE: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn reduce_sum(n: u32, data: &[f32], mut out: DisjointSlice<f32>) {
        static mut SDATA: SharedArray<f32, BLOCK_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let n_usize = n as usize;

        // Uncoalesced: thread gtid owns the contiguous block [start, start+chunk).
        let total_threads = 1024usize * BLOCK_SIZE;
        let chunk = (n_usize + total_threads - 1) / total_threads;
        let gtid = bid * BLOCK_SIZE + tid;
        let start = gtid * chunk;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < chunk {
            let idx = start + j;
            if idx < n_usize {
                acc += data[idx];
            }
            j += 1;
        }
        unsafe { SDATA[tid] = acc; }
        thread::sync_threads();

        // Correct sequential-addressing tree (negligible vs the load cost).
        let mut stride = BLOCK_SIZE / 2;
        while stride > 0 {
            if tid < stride {
                unsafe { SDATA[tid] = SDATA[tid] + SDATA[tid + stride]; }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe { *out.get_unchecked_mut(bid) = SDATA[0]; }
        }
    }
}
