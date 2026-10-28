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

/// TF_FLASHNEXT_GATHER_EARLY=1: a shared pass gathers every stream's n-gram rows in one pool call (not one call a stream).
pub fn gatherEarly() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GATHER_EARLY") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_GATHER_LOG=1: the gather's and the PLE layer's join wait of shared passes, averaged over 100 passes.
pub fn gatherLog() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GATHER_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_GATHER_LOCK_SCALES=1: lock the n-gram table's scale arrays in memory at load (see Ngram.lockScales).
pub fn gatherLockScales() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GATHER_LOCK_SCALES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_GDN_LANES=1: a shared round's DeltaNet layers take every lane's rows in three launches (front, late chain, window
/// shift) instead of five a lane. Each lane's kernel program is the per-lane one's on its own pointers: the same bytes out.
pub fn gdnLanes() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GDN_LANES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_SHARED_DIRECT=1: a prompt chunk reads the shared expert's output where it lies instead of scattering it into y.
/// Off by default: the scatter is what 1.1.8 does, and the direct read is the recipe series' shortcut.
pub fn sharedDirect() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SHARED_DIRECT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
