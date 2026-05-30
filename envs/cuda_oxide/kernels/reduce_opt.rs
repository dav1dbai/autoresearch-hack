// Optimized parallel reduction starter — the agent begins here and climbs further.
// Techniques applied: grid-stride accumulation (each thread sums multiple elements
// before entering shared memory), sequential addressing (no bank conflicts, no warp
// divergence in the tree), and warp-shuffle final reduction (last 32 threads bypass
// __syncthreads for the last five steps). Deliberate headroom left: no vectorized
// loads (float4), no instruction-level unrolling beyond the warp-shuffle section,
// and the grid-stride factor is modest (4 elements per thread instead of 8+).
//
// CONTRACT (fixed — host launches it; do NOT change the signature):
//   reduce_sum(n, data, out) — `out` has exactly `num_blocks` slots (1024).
//   The host accumulates the partials on CPU after the kernel returns.
//   Launch: 1024 blocks × 256 threads. Grid-stride covers all 32M elements.

const BLOCK_SIZE: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn reduce_sum(n: u32, data: &[f32], mut out: DisjointSlice<f32>) {
        static mut SDATA: SharedArray<f32, BLOCK_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let block_size = BLOCK_SIZE;            // fixed launch: blockDim.x
        let grid_size = 1024usize * BLOCK_SIZE; // fixed launch: gridDim.x(1024) * blockDim.x
        let n_usize = n as usize;

        // Grid-stride accumulation: each thread covers multiple elements.
        let mut acc = 0.0f32;
        let mut i = bid * block_size + tid;
        while i < n_usize {
            acc += data[i];
            i += grid_size;
        }

        // Store per-thread partial into shared memory.
        unsafe { SDATA[tid] = acc; }
        thread::sync_threads();

        // Sequential-addressing tree reduction (no bank conflicts, no divergence).
        // Stop at stride=32: the last warp will finish with shuffle intrinsics.
        let mut stride = block_size / 2;
        while stride > 32 {
            if tid < stride {
                unsafe { SDATA[tid] = SDATA[tid] + SDATA[tid + stride]; }
            }
            thread::sync_threads();
            stride /= 2;
        }

        // Warp-level reduction for the last 32 active threads (tid < 32).
        // warp::shuffle_down_f32 replaces __shfl_down_sync — no syncthreads needed
        // within a warp. We read the shared-mem partial first to seed the warp sum.
        if tid < 32 {
            // Tree loop exited at stride=32, so SDATA[0..64] hold 64 live partials.
            // Fold the stride-32 step manually (SDATA[tid] += SDATA[tid+32]) to get
            // down to 32 lane values, then warp-shuffle the final 5 steps.
            let mut val = unsafe { SDATA[tid] + SDATA[tid + 32] };

            // 5-step warp butterfly using shuffle_down (only lane 0 gets the full sum).
            val = val + warp::shuffle_down_f32(val, 16);
            val = val + warp::shuffle_down_f32(val, 8);
            val = val + warp::shuffle_down_f32(val, 4);
            val = val + warp::shuffle_down_f32(val, 2);
            val = val + warp::shuffle_down_f32(val, 1);

            if tid == 0 {
                unsafe {
                    *out.get_unchecked_mut(bid) = val;
                }
            }
        }
    }
}
