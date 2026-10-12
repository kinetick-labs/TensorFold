//! EXL3: the MTP draft head built from the trellis lm_head (the pack keeps no bf16 head, so the drafts used to run
//! the whole 248k-row head, ~2 ms a step). The head is read back 128 input columns at a time (identity rows through
//! the trellis matmul: output row r is column c0 + r of the head in bf16), its draft rows gathered and packed 4-bit
//! (groups of 32) like the bf16 packs' draft head. Only the drafts read it: the output is unchanged, acceptance can move.
const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const lay = @import("cuda_layouts.zig");

pub fn build(gpa: std.mem.Allocator, d: anytype, f: anytype, w: *W.Weights, stream: anytype, vocab: u32, rank: u32, world: u32) !void {
    const h = w.head3 orelse return;
    const ids = try W.draftIds(gpa, W.draft_vocab, vocab, rank, world);
    defer gpa.free(ids);
    const k: usize = h.k;
    const nh: usize = h.n;
    const n = ids.len;
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    var xb = try cuda.DeviceBuffer.alloc(d, 128 * k * 2);
    defer xb.free();
    var ob = try cuda.DeviceBuffer.alloc(d, 128 * nh * 2);
    defer ob.free();
    const eye = try gpa.alloc(u16, 128 * k);
    defer gpa.free(eye);
    const out = try gpa.alloc(u16, 128 * nh);
    defer gpa.free(out);
    var c0: usize = 0;
    while (c0 < k) : (c0 += 128) {
        @memset(eye, 0);
        for (0..128) |r| eye[r * k + c0 + r] = 0x3F80; // bf16 1.0
        try xb.upload(0, std.mem.sliceAsBytes(eye));
        try f.x3mm(null, xb.ptr, k, 1, h, ob.ptr, 1, 128);
        try stream.synchronize();
        try ob.download(0, std.mem.sliceAsBytes(out));
        for (ids, 0..) |id, j| for (0..128) |r| {
            rows[j * k + c0 + r] = out[r * nh + (id - w.vocab_offset)];
        };
    }
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const sc = try gpa.alloc(u16, n * k / 32);
    defer gpa.free(sc);
    const bi = try gpa.alloc(u16, n * k / 32);
    defer gpa.free(bi);
    lay.quantize4(rows, n, k, words, sc, bi);
    const npad = (n + 127) / 128 * 128;
    const kg = k / 32;
    const frag = try gpa.alloc(u32, npad / 64 * kg * 8 * 32);
    defer gpa.free(frag);
    @memset(frag, 0);
    lay.packQ4(words, n, k, frag);
    const major = try gpa.alloc(u16, kg * npad);
    defer gpa.free(major);
    const wb = frag.len * 4;
    const sb = major.len * 2;
    var buf = try cuda.DeviceBuffer.alloc(d, wb + 2 * sb + n * 8);
    errdefer buf.free();
    try buf.upload(0, std.mem.sliceAsBytes(frag));
    @memset(major, 0);
    lay.majorQ4(sc, n, kg, major);
    try buf.upload(wb, std.mem.sliceAsBytes(major));
    @memset(major, 0);
    lay.majorQ4(bi, n, kg, major);
    try buf.upload(wb + sb, std.mem.sliceAsBytes(major));
    const wide = try gpa.alloc(i64, n);
    defer gpa.free(wide);
    for (ids, wide) |id, *x| x.* = id;
    try buf.upload(wb + 2 * sb, std.mem.sliceAsBytes(wide));
    try w.buffers.append(gpa, buf);
    w.bytes += buf.len;
    w.draft_head = .{ .weight = buf.ptr, .scales = buf.ptr + wb, .biases = buf.ptr + wb + sb, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
    w.draft_ids = buf.ptr + wb + 2 * sb;
    w.draft_count = @intCast(n);
}
