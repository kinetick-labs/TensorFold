//! The runner's and compute lane's libc calls linked for -Dtarget (`zig build runner-link-check`); never run.
const std = @import("std");

pub fn main() void {
    std.mem.doNotOptimizeAway(&@import("runner_frontend_forkserver.zig").run);
    std.mem.doNotOptimizeAway(&@import("runner_frontend_forkserver.zig").start);
    std.mem.doNotOptimizeAway(&@import("runner_frontend_spawn.zig").run);
    std.mem.doNotOptimizeAway(&@import("compute_lane.zig").Pool.run);
    std.mem.doNotOptimizeAway(&@import("compute_sandbox.zig").main);
}
