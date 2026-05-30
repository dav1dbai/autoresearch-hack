// Starter kernel for the cuda-oxide TMA copy bandwidth env (the editable surface).
// This is the VERBATIM tma_copy_2d_test kernel pattern from the cuda-oxide `tma_copy`
// example (crates/rustc-codegen-cuda/examples/tma_copy/src/main.rs) — real NVIDIA
// shipped code, scaled to NUM_TILES blocks so the benchmark is bandwidth-meaningful.
//
// CONTRACT (fixed — host launches it; do NOT change the signature):
//   tma_copy_bench(tensor_map, tiles_x, tiles_y, out)
//   Launched: NUM_TILES blocks × 256 threads.
//   tensor_map: TMA descriptor for TENSOR_H × TENSOR_W f32, tile TILE_H × TILE_W.
//   tiles_x, tiles_y: grid of tiles (tiles_x * tiles_y == NUM_TILES blocks).
//   out: flat f32 buffer of size NUM_TILES * TILE_H * TILE_W.
//   Each block copies one tile to out[blockIdx.x * TILE_SIZE ..].
//
// Optimize: double-buffer (two SharedArrays + alternating barriers), process
// multiple tiles per block with software pipeline, tune TILE_SIZE (must match
// the TMA descriptor which the host holds fixed at 64x64).
// Do NOT change the kernel name or its four-argument signature.
#[cuda_module]
mod kernels {
    use super::*;

    /// TMA bulk copy benchmark: each block copies one 64x64 tile from global memory
    /// to shared memory via the Hopper TMA engine, then scatters to the output buffer.
    ///
    /// The Hopper-canonical pattern (from the real tma_copy example):
    ///   thread 0: mbarrier_init → fence_proxy_async_shared_cta → cp_async_bulk_tensor_2d_g2s
    ///   all threads: arrive at barrier (thread 0 with expect_tx, others plain arrive)
    ///   all threads: spin on mbarrier_try_wait until TMA completes
    ///   sync_threads + scatter tile to output
    #[kernel]
    pub fn tma_copy_bench(
        tensor_map: *const TmaDescriptor,
        tiles_x: i32,
        tiles_y: i32,
        mut out: DisjointSlice<f32>,
    ) {
        const TILE_W: usize = 64;
        const TILE_H: usize = 64;
        const TILE_SIZE: usize = TILE_W * TILE_H;
        const TILE_BYTES: u32 = (TILE_SIZE * core::mem::size_of::<f32>()) as u32;
        // TMA destination MUST be 128-byte aligned — use 3-arg SharedArray form.
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let bid = thread::blockIdx_x() as i32;

        // Map flat block index to 2D tile coordinates.
        let tile_x = bid % tiles_x;
        let tile_y = bid / tiles_x;

        // Thread 0: initialize barrier for all threads in the block, then fence so
        // the TMA async proxy sees the barrier initialization before the copy starts.
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        // Thread 0: issue the asynchronous TMA 2D bulk copy global→shared.
        if tid == 0 {
            unsafe {
                // coord0/coord1 are element offsets (not tile indices) into the tensor.
                // TMA 2D: coord0 = column element offset, coord1 = row element offset.
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE as *mut u8,
                    tensor_map,
                    tile_x * (TILE_W as i32),
                    tile_y * (TILE_H as i32),
                    &raw mut BAR,
                );
            }
        }

        // All threads arrive at the barrier.
        // Thread 0: arrive_expect_tx (announces how many bytes TMA will produce).
        // Other threads: plain arrive.
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };

        // All threads spin until the TMA copy is complete.
        unsafe {
            while !mbarrier_try_wait(&raw const BAR, token) {}
        }
        thread::sync_threads();

        // Scatter this block's tile to the flat output buffer (grid-stride over tile).
        let out_base = (bid as usize) * TILE_SIZE;
        let mut elem = tid as usize;
        while elem < TILE_SIZE {
            let val = unsafe { TILE[elem] };
            unsafe {
                *out.get_unchecked_mut(out_base + elem) = val;
            }
            elem += block_size as usize;
        }
    }
}
