//! The in-model runner's view of each committed token (core/runner.zig): true holds the stream for a result here.
pub const RunnerHook = struct {
    ptr: *anyopaque,
    token: *const fn (ptr: *anyopaque, token: u32, think_open: bool, can_hold: bool, think_end: bool) bool,
    settled: ?*const fn (ptr: *anyopaque) ?[]const u32 = null, // at a hold: a result forced next instead, or null
    eos: ?*const fn (ptr: *anyopaque, think_open: bool, can_hold: bool) bool = null,
};
