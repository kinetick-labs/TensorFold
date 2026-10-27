//! --kv-cache fp8 on the device: the quantizing write (fn_attn_prep_kv8) against cuda_kv.zig's model of the format, and the
//! widening reads (fn_attn_chunks_kv8, fn_attn_fused_kv8) against the bf16 kernels run on the dequantized rows, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const kvf = flash.kv_format;
const Rig = rig.Rig;
const bf = rig.bf;

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const eps: f32 = 1e-6;
const cap = 3072;
const idw = 2052;
const key_stride = kvf.key_stride;

fn inverse(r: *Rig) ![]f32 {
    const inv = try r.a.alloc(f32, half);
    for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
    return inv;
}

fn f32of(h: u16) f32 {
    return @floatCast(bf(h));
}

/// The write: the bf16 kernel's key and value rows through the model must be the fp8 kernel's bytes, and everything else it writes
/// (queries, indexer queries and keys) the bf16 kernel's bytes.
fn prep(r: *Rig, g: *const flash.Glue) !void {
    const rows = 17;
    const pos0 = 700;
    const p = r.bfs(rows * proj_n, 2);
    // row 1's values are all zero (a zero row: scale 2^-126, zero codes)
    const v_at = q_heads * 2 * head_dim + kv_heads * head_dim;
    @memset(p[proj_n + v_at ..][0 .. kv_heads * head_dim], 0);
    const qw = r.f32s(head_dim, 0.5, 1.5);
    const kw = r.f32s(head_dim, 0.5, 1.5);
    const iw = r.f32s(index_dim, 0.5, 1.5);
    const inv = try inverse(r);
    const dp = try r.dev(std.mem.sliceAsBytes(p));
    const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
    const dqw = try r.dev(std.mem.sliceAsBytes(qw));
    const dkw = try r.dev(std.mem.sliceAsBytes(kw));
    const diw = try r.dev(std.mem.sliceAsBytes(iw));
    const dinv = try r.dev(std.mem.sliceAsBytes(inv));
    // bf16 caches, then fp8 ones
    const q_b = try r.out(rows * q_heads * head_dim * 2);
    const iq_b = try r.out(rows * index_heads * index_dim * 2);
    const kc_b = try r.out(cap * kv_heads * head_dim * 2);
    const vc_b = try r.out(cap * kv_heads * head_dim * 2);
    const ikc_b = try r.out(cap * index_dim * 2);
    try g.attnPrep(r.s, dp, dpos, dqw, dkw, diw, dinv, q_b, kc_b, vc_b, iq_b, ikc_b, eps, rows);
    const q_8 = try r.out(rows * q_heads * head_dim * 2);
    const iq_8 = try r.out(rows * index_heads * index_dim * 2);
    const kc_8 = try r.out(cap * kvf.Kv.fp8.keyRow());
    const vc_8 = try r.out(cap * kvf.Kv.fp8.valueRow());
    const ikc_8 = try r.out(cap * index_dim * 2);
    try g.attnPrepKv8(r.s, dp, dpos, dqw, dkw, diw, dinv, q_8, kc_8, vc_8, iq_8, ikc_8, eps, rows);

    try r.sameRows("fp8 prep queries", u16, q_8, try r.back(u16, q_b, rows * q_heads * head_dim));
    try r.sameRows("fp8 prep indexer queries", u16, iq_8, try r.back(u16, iq_b, rows * index_heads * index_dim));
    try r.sameRows("fp8 prep indexer keys", u16, ikc_8 + pos0 * index_dim * 2, (try r.back(u16, ikc_b, cap * index_dim))[pos0 * index_dim ..][0 .. rows * index_dim]);

    const kb = try r.back(u16, kc_b, cap * kv_heads * head_dim);
    const vb = try r.back(u16, vc_b, cap * kv_heads * head_dim);
    const k8 = try r.back(u8, kc_8, cap * kvf.Kv.fp8.keyRow());
    const v8 = try r.back(u8, vc_8, cap * kvf.Kv.fp8.valueRow());
    var k: [head_dim]f32 = undefined;
    var v: [head_dim]f32 = undefined;
    var want_k: [key_stride]u8 = undefined;
    var want_v: [head_dim]u8 = undefined;
    for (0..rows) |row| for (0..kv_heads) |h| {
        const slot = kvf.slotOf(pos0 + row, h);
        for (&k, &v, 0..) |*a, *b, d| {
            a.* = f32of(kb[slot * head_dim + d]);
            b.* = f32of(vb[slot * head_dim + d]);
        }
        kvf.packRows(&k, &v, &want_k, &want_v);
        try check.sameBytes("fp8 prep key row (codes, scales, zero tail)", k8[kvf.keyRowAt(slot)..][0..key_stride], &want_k);
        try check.sameBytes("fp8 prep value row", v8[kvf.valueRowAt(slot)..][0..head_dim], &want_v);
    };
    // positions the launch did not write stay as filled
    try check.expect(k8[kvf.keyRowAt(kvf.slotOf(pos0 - 1, 1))] == 0xff and v8[kvf.valueRowAt(kvf.slotOf(pos0 + rows, 0))] == 0xff, "fp8 prep writes only its rows' positions", .{});
    // the zero value row of row 1: zero codes and the smallest scale
    const zero_slot = kvf.slotOf(pos0 + 1, 0);
    for (v8[kvf.valueRowAt(zero_slot)..][0..head_dim]) |c| try check.expect(c == 0, "a zero value row has zero codes", .{});
    const sv: f32 = @bitCast(std.mem.readInt(u32, k8[kvf.keyRowAt(zero_slot) + kvf.value_scale_at ..][0..4], .little));
    try check.expect(sv == 0x1p-126, "a zero value row's scale is 2^-126", .{});
}

/// A cache's rows of bf16 values, each row of a different size, as the format's bytes (kc8, vc8) and the bf16 rows they dequantize to.
const Cache = struct {
    kc8: []u8,
    vc8: []u8,
    kcd: []u16,
    vcd: []u16,
};

fn cache(r: *Rig) !Cache {
    const slots = cap * kv_heads;
    const c: Cache = .{
        .kc8 = try r.a.alloc(u8, slots * key_stride),
        .vc8 = try r.a.alloc(u8, slots * head_dim),
        .kcd = try r.a.alloc(u16, slots * head_dim),
        .vcd = try r.a.alloc(u16, slots * head_dim),
    };
    var k: [head_dim]f32 = undefined;
    var v: [head_dim]f32 = undefined;
    var back: [head_dim]f32 = undefined;
    for (0..slots) |slot| {
        const size = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast(slot * 7 % 9)) - 4);
        for (&k, &v) |*a, *b| {
            a.* = @floatCast(bf(rig.bf16(@floatCast((r.rng.float(f32) * 2 - 1) * size))));
            b.* = @floatCast(bf(rig.bf16(@floatCast((r.rng.float(f32) * 2 - 1) * size * 0.5))));
        }
        const kr = c.kc8[slot * key_stride ..][0..key_stride];
        const vr = c.vc8[slot * head_dim ..][0..head_dim];
        kvf.packRows(&k, &v, kr, vr);
        kvf.dequantizeRow(kr[0..head_dim], @bitCast(std.mem.readInt(u32, kr[kvf.key_scale_at..][0..4], .little)), &back);
        for (back, c.kcd[slot * head_dim ..][0..head_dim]) |a, *o| o.* = rig.bf16(a);
        kvf.dequantizeRow(vr, @bitCast(std.mem.readInt(u32, kr[kvf.value_scale_at..][0..4], .little)), &back);
        for (back, c.vcd[slot * head_dim ..][0..head_dim]) |a, *o| o.* = rig.bf16(a);
    }
    return c;
}

/// The read: the fp8 kernels over the codes against the bf16 kernels over the dequantized rows, dense and sparse, 1, 3 and 20 rows
/// (the bf16 side reads ahead for up to 16 rows, one tile at a time past that), and the fused prompt launch.
fn attention(r: *Rig, g: *const flash.Glue, c: Cache, sparse: bool) !void {
    const counts = [_]usize{ 1, 3, 20 };
    const rows = counts[counts.len - 1];
    const pos0: usize = if (sparse) 2600 else 1100;
    const q = r.bfs(rows * q_heads * head_dim, 1);
    const ids = try r.a.alloc(i32, rows * idw);
    const nk = try r.a.alloc(i32, rows);
    const spr = try r.a.alloc(i32, rows);
    for (0..rows) |row| {
        nk[row] = @intCast(if (sparse) idw - 1 - row % 3 else pos0 + row + 1);
        spr[row] = @intFromBool(sparse);
        for (0..idw) |i| ids[row * idw + i] = @intCast((i * 1237 + row * 31) % (pos0 + row + 1));
    }
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const dkc8 = try r.dev(c.kc8);
    const dvc8 = try r.dev(c.vc8);
    const dkcd = try r.dev(std.mem.sliceAsBytes(c.kcd));
    const dvcd = try r.dev(std.mem.sliceAsBytes(c.vcd));
    const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ @intCast(pos0), 0, 0, 0 }));
    const dids = try r.dev(std.mem.sliceAsBytes(ids));
    const dnk = try r.dev(std.mem.sliceAsBytes(nk));
    const dspr = try r.dev(std.mem.sliceAsBytes(spr));
    const nch = 5;
    const po = try r.out(rows * nch * q_heads * head_dim * 4);
    const pm = try r.out(rows * nch * q_heads * 4);
    const pl = try r.out(rows * nch * q_heads * 4);
    const name = if (sparse) "fp8 attention sparse" else "fp8 attention dense";
    for (counts) |m| {
        const chunks = if (sparse) (idw + 511) / 512 else (pos0 + m + 511) / 512;
        const want = try r.out(m * q_heads * head_dim * 2);
        try g.attnChunks(r.s, dq, dkcd, dvcd, dpos, po, pm, pl, dids, dnk, dspr, m, kv_heads, chunks, true);
        try g.attnMerge(r.s, po, pm, pl, dpos, want, dnk, dspr, m, kv_heads, true);
        const got = try r.out(m * q_heads * head_dim * 2);
        try g.attnChunksKv8(r.s, dq, dkc8, dvc8, dpos, po, pm, pl, dids, dnk, dspr, m, kv_heads, chunks, true);
        try g.attnMerge(r.s, po, pm, pl, dpos, got, dnk, dspr, m, kv_heads, true);
        try r.sameRows(name, u16, got, try r.back(u16, want, m * q_heads * head_dim));
    }
    const want = try r.out(rows * q_heads * head_dim * 2);
    try g.attnFused(r.s, dq, dkcd, dvcd, dpos, want, dids, dnk, dspr, rows, kv_heads);
    const got = try r.out(rows * q_heads * head_dim * 2);
    try g.attnFusedKv8(r.s, dq, dkc8, dvc8, dpos, got, dids, dnk, dspr, rows, kv_heads);
    try r.sameRows(if (sparse) "fp8 attention sparse fused" else "fp8 attention dense fused", u16, got, try r.back(u16, want, rows * q_heads * head_dim));
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(61);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    try prep(&r, &g);
    const c = try cache(&r);
    try attention(&r, &g, c, false);
    try attention(&r, &g, c, true);
    check.pass("flashnext-kv8: the quantizing write equals the format's model byte for byte; the widening reads equal the bf16 kernels on the dequantized rows, dense and sparse, chunked and fused", .{});
}
