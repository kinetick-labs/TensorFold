//! Prompt-path switches (environment, read once per process); the defaults are the fast paths.
const std = @import("std");

/// TF_FLASHNEXT_PROMPT_SYNC=1: drain the stream after every non-final prompt chunk, as v1.1.4 did. The drain leaves the
/// GPU idle while the host rebuilds the next chunk's launches and its n-gram gather (about 40 ms a 1,536-row chunk).
pub fn chunkSync() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_SYNC") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_PROMPT_MX128=N: MXFP8 projections of N columns or more take the relaxed 128 x 128 prompt tile in prompt
/// passes (default 1024); 0 keeps the exact kernels everywhere.
pub fn mx128MinN() usize {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_MX128") orelse return 1024;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1024;
}
