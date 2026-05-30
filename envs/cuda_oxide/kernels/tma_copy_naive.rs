// Naive baseline for the TMA copy bandwidth env — correct but slower.
// Uses TMA correctly (required; the raw data pointer is not passed to the kernel)
// but scatters the tile to output sequentially with thread 0 only, leaving the
// other 255 threads idle after the barrier. This is a genuine slowdown vs the
// starter which uses all 256 threads in a grid-stride scatter.
//
// CONTRACT: same signature as tma_copy_tiled.rs.
//   tma_copy_bench(tensor_map, tiles_x, tiles_y, out) — launched NUM_TILES blocks x 256 threads.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn tma_copy_bench(
        tensor_map: *const TmaDescriptor,
        tiles_x: i32,
        tiles_y: i32,
        mut out: DisjointSlice<f32>,
    ) {
        let _ = tiles_y;

        const TILE_W: usize = 64;
        const TILE_H: usize = 64;
        const TILE_SIZE: usize = TILE_W * TILE_H;
        const TILE_BYTES: u32 = (TILE_SIZE * core::mem::size_of::<f32>()) as u32;
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let bid = thread::blockIdx_x() as i32;

        let tile_x = bid % tiles_x;
        let tile_y = bid / tiles_x;

        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        if tid == 0 {
            unsafe {
                // coord0/coord1 are element offsets (not tile indices) into the tensor.
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE as *mut u8,
                    tensor_map,
                    tile_x * (TILE_W as i32),
                    tile_y * (TILE_H as i32),
                    &raw mut BAR,
                );
            }
        }

        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };

        unsafe {
            while !mbarrier_try_wait(&raw const BAR, token) {}
        }
        thread::sync_threads();

        // Naive: only thread 0 scatters all TILE_SIZE elements (sequential, wastes 255 threads).
        if tid == 0 {
            let out_base = (bid as usize) * TILE_SIZE;
            let mut elem = 0usize;
            while elem < TILE_SIZE {
                let val = unsafe { TILE[elem] };
                unsafe {
                    *out.get_unchecked_mut(out_base + elem) = val;
                }
                elem += 1;
            }
        }
    }
}
