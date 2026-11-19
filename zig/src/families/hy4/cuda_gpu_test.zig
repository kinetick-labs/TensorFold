//! The Hy4 GPU test binary (hy4-gpu-test): kernels against the CPU references beside a serving model (HY4_HC_PTX), and
//! the window-only whole-pack load (HY4_LOAD_PACK). Each test skips without its variable.

test {
    _ = @import("cuda_hc_test.zig");
    _ = @import("cuda_load_test.zig");
    _ = @import("cuda_fwd_test.zig");
}
