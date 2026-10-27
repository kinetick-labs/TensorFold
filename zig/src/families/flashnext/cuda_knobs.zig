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

/// TF_FLASHNEXT_PROMPT_HC2=1: the native hyper-connection read-out of prompt passes (TF_FLASHNEXT_HC2_MIN rows or more) runs on two fused
/// kernels (flashnext_hcf.cu hcn: the norm formed in the down tile and in the up epilogue, the up rows mixed in place)
/// instead of five launches; every byte of its outputs is the five-launch chain's.
pub fn promptHc2() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_HC2") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_HC2_DN / TF_FLASHNEXT_HC2_UP: the CTA row tile (16 * N rows) of those two kernels; 0: the default.
pub fn hc2Tile(name: [:0]const u8) usize {
    const v = std.c.getenv(name) orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
}

/// TF_FLASHNEXT_HC2_MIN: the fewest rows of a prompt pass that take the fused read-out (default 768; its CTAs need rows to fill the GPU).
pub fn hc2MinRows() usize {
    const v = std.c.getenv("TF_FLASHNEXT_HC2_MIN") orelse return 768;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 768;
}

/// TF_FLASHNEXT_HEAD_PIECE: the rows a prompt chunk's MTP head pass takes at most (default 256, the decode rounds' piece).
/// From 1024 rows the head's experts take the shared-memory row tiles (the decode kernel's bits) and read each weight once.
pub fn headPiece() usize {
    const v = std.c.getenv("TF_FLASHNEXT_HEAD_PIECE") orelse return 256;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return 256;
    return if (n == 0) 256 else n;
}

/// TF_FLASHNEXT_PROMPT_SHARED_DIRECT=1: a prompt pass's MoE write-back reads the shared expert's fp32 output in place instead of
/// a copy of it in y's last slot (the same value in the same fused-multiply-add order; one launch and 62 MB per layer less).
pub fn sharedDirect() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_SHARED_DIRECT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_PROMPT_GEMM2=1: the MX128 prompt tile (GDN in-projection, attention qkv, out projections) runs with two pipeline
/// stages and two blocks an SM: one block output stores and pipeline fill overlap the other K loop. Same arithmetic, same bits.
pub fn promptGemm2() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_GEMM2") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_GDN_LANES=1: a shared round's DeltaNet layers take every lane's rows in three launches (front, late chain, window
/// shift) instead of five a lane. Each lane's kernel program is the per-lane one's on its own pointers: the same bytes out.
pub fn gdnLanes() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GDN_LANES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
