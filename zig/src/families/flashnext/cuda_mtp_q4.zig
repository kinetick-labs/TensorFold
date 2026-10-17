//! Decode plan D5 (work/research/R1-decode.md 3.5): the MTP head's bf16 matrices in affine 4-bit, groups of 32 (the
//! draft head's own format: cuda_layouts.quantize4 / packQ4 / majorQ4, qmm.Q4), for the drafts only. The target never
//! reads them, so its output is unchanged; only the drafts (and so the rounds' acceptance) can move.
//! TF_FLASHNEXT_MTP_Q4=1 builds the table at load from the loaded bf16 weights (this rank's slices at TP=2):
//! fc_embedding, fc_hidden, the MTP layer's attention projection and o_proj, its two hyper-connections' down and up
//! projections and the head's mixer. Forward.mmx then runs a matrix found here through qmm (K slices in a cluster).
//! The value may name parts instead of all ("fc,hc,mixer", "attn"); TF_FLASHNEXT_MTP_Q4_MSE=1 picks each group's
//! range by the least squared error over shrunken min/max ranges instead of min/max itself.
const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const lay = @import("cuda_layouts.zig");
const fp8 = @import("cuda_fp8.zig");

const Allocator = std.mem.Allocator;

pub const max_entries = 12;

pub const Table = struct {
    keys: [max_entries]u64 = @splat(0),
    qs: [max_entries]kern.Q4 = undefined,
    /// TF_FLASHNEXT_MTP_MX8: the entry as MXFP8 (fp8.Linear mx) instead of 4-bit
    mx: [max_entries]?fp8.Linear = @splat(null),
    bufs: [max_entries]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    bytes_bf16: usize = 0,
    bytes_q4: usize = 0,

    pub fn deinit(t: *Table) void {
        for (t.bufs[0..t.n]) |*b| b.free();
        t.n = 0;
    }

    /// The 4-bit copy of the bf16 matrix at `weight`, if it is one of the head's.
    pub fn find(t: *const Table, weight: u64) ?kern.Q4 {
        for (t.keys[0..t.n], t.qs[0..t.n], t.mx[0..t.n]) |k, q, m| if (k == weight and m == null) return q;
        return null;
    }

    /// The MXFP8 copy of the bf16 matrix at `weight`.
    pub fn findMx(t: *const Table, weight: u64) ?fp8.Linear {
        for (t.keys[0..t.n], t.mx[0..t.n]) |k, m| if (k == weight) return m;
        return null;
    }

    /// Quantizes every MTP bf16 matrix of `w` (downloaded from the device) on `threads` host threads.
    /// `parts`: "1" or "all", else a comma list of fc, attn, hc, mixer.
    pub fn build(gpa: Allocator, d: *const cuda.Driver, w: *const W.Weights, parts: []const u8, mse: bool) !Table {
        return buildAs(gpa, d, w, parts, mse, false);
    }

    /// `build` with `mx8`: MXFP8 (e4m3, a UE8M0 power of two per 32 inputs) instead of 4-bit.
    pub fn buildAs(gpa: Allocator, d: *const cuda.Driver, w: *const W.Weights, parts: []const u8, mse: bool, mx8: bool) !Table {
        var t: Table = .{};
        errdefer t.deinit();
        const m = w.mtp orelse return error.NoMtpHead;
        const l = &m.layer;
        const a = l.attn orelse return error.NoMtpAttention;
        const Group = struct { name: []const u8, rows: []const W.Rows };
        const groups = [_]Group{
            .{ .name = "fc", .rows = &.{ m.fc_e, m.fc_h } },
            .{ .name = "attn", .rows = &.{ a.proj, a.o } },
            .{ .name = "hc", .rows = &.{ l.attn_hc.down, l.attn_hc.up, l.mlp_hc.down, l.mlp_hc.up } },
            .{ .name = "mixer", .rows = &.{ m.mixer.down, m.mixer.up } },
        };
        const every = std.mem.eql(u8, parts, "1") or std.mem.eql(u8, parts, "all");
        for (groups) |g| {
            if (!every and !listed(parts, g.name)) continue;
            for (g.rows) |r| {
                if (r.weight == 0 or t.find(r.weight) != null) continue;
                if (mx8) try t.addMx(gpa, d, r) else try t.add(gpa, d, r, mse);
            }
        }
        return t;
    }

    fn listed(parts: []const u8, name: []const u8) bool {
        var it = std.mem.tokenizeScalar(u8, parts, ',');
        while (it.next()) |p| if (std.mem.eql(u8, p, name)) return true;
        return false;
    }

    fn addMx(t: *Table, gpa: Allocator, d: *const cuda.Driver, r: W.Rows) !void {
        if (t.n == max_entries) return error.TooMany;
        const n: usize = r.n;
        const k: usize = r.k;
        if (k % 64 != 0) return error.UnsupportedShape;
        const rows = try gpa.alloc(u16, n * k);
        defer gpa.free(rows);
        const src: cuda.DeviceBuffer = .{ .d = d, .ptr = r.weight, .len = n * k * 2 };
        try src.download(0, std.mem.sliceAsBytes(rows));
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        const sc = try gpa.alloc(u8, n * (k / 32));
        defer gpa.free(sc);
        quantizeMx(rows, n, k, codes, sc);
        const npad = fp8.npadOf(n);
        const frag = try gpa.alloc(u8, npad * k);
        defer gpa.free(frag);
        try fp8.fragmentOrder(codes, n, k, npad, frag);
        const tiled = try gpa.alloc(u8, npad * (k / 64) * 2);
        defer gpa.free(tiled);
        try fp8.tileMxScales(sc, n, k, npad, tiled);
        var buf = try cuda.DeviceBuffer.alloc(d, frag.len + tiled.len);
        errdefer buf.free();
        try buf.upload(0, frag);
        try buf.upload(frag.len, tiled);
        t.bufs[t.n] = buf;
        t.keys[t.n] = r.weight;
        t.mx[t.n] = .{ .w8 = buf.ptr, .bs = buf.ptr + frag.len, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad), .mx = true };
        t.n += 1;
        t.bytes_bf16 += n * k * 2;
        t.bytes_q4 += frag.len + tiled.len;
    }

    fn add(t: *Table, gpa: Allocator, d: *const cuda.Driver, r: W.Rows, mse: bool) !void {
        if (t.n == max_entries) return error.TooMany;
        const n: usize = r.n;
        const k: usize = r.k;
        if (k % 32 != 0) return error.UnsupportedShape;
        const rows = try gpa.alloc(u16, n * k);
        defer gpa.free(rows);
        const src: cuda.DeviceBuffer = .{ .d = d, .ptr = r.weight, .len = n * k * 2 };
        try src.download(0, std.mem.sliceAsBytes(rows));
        const kg = k / 32;
        const words = try gpa.alloc(u32, n * k / 8);
        defer gpa.free(words);
        const sc = try gpa.alloc(u16, n * kg);
        defer gpa.free(sc);
        const bi = try gpa.alloc(u16, n * kg);
        defer gpa.free(bi);
        if (mse) quantize4Mse(rows, n, k, words, sc, bi) else lay.quantize4(rows, n, k, words, sc, bi);
        const npad = (n + 127) / 128 * 128;
        const frag = try gpa.alloc(u32, npad / 64 * kg * 8 * 32);
        defer gpa.free(frag);
        lay.packQ4(words, n, k, frag);
        const major = try gpa.alloc(u16, kg * npad);
        defer gpa.free(major);
        // one device buffer: words, then scales, then biases
        const wb = frag.len * 4;
        const sb = major.len * 2;
        var buf = try cuda.DeviceBuffer.alloc(d, wb + 2 * sb);
        errdefer buf.free();
        try buf.upload(0, std.mem.sliceAsBytes(frag));
        lay.majorQ4(sc, n, kg, major);
        try buf.upload(wb, std.mem.sliceAsBytes(major));
        lay.majorQ4(bi, n, kg, major);
        try buf.upload(wb + sb, std.mem.sliceAsBytes(major));
        t.bufs[t.n] = buf;
        t.keys[t.n] = r.weight;
        t.qs[t.n] = .{ .w = buf.ptr, .s = buf.ptr + wb, .b = buf.ptr + wb + sb, .n = n, .k = k, .npad = npad };
        t.n += 1;
        t.bytes_bf16 += n * k * 2;
        t.bytes_q4 += wb + 2 * sb;
    }
};

/// bf16 rows -> MXFP8: each 32 inputs' power of two 2^e, e the least with amax / 2^e <= 448 (UE8M0 byte e + 127),
/// the codes e4m3 RNE of x / 2^e.
pub fn quantizeMx(rows: []const u16, n: usize, k: usize, codes: []u8, scales: []u8) void {
    for (0..n) |r| for (0..k / 32) |g| {
        const v = rows[r * k + g * 32 ..][0..32];
        var amax: f32 = 0;
        for (v) |b| amax = @max(amax, @abs(lay.bf16ToF32(b)));
        var e: i32 = -127;
        if (amax > 0) {
            e = @intFromFloat(@ceil(std.math.log2(amax / 448.0)));
            while (std.math.ldexp(@as(f32, 448.0), e) < amax) e += 1;
        }
        e = std.math.clamp(e, -126, 127);
        scales[r * (k / 32) + g] = @intCast(e + 127);
        const inv = std.math.ldexp(@as(f32, 1.0), -e);
        for (v, 0..) |b, i| codes[r * k + g * 32 + i] = lay.f32ToE4m3(lay.bf16ToF32(b) * inv);
    };
}

test "MXFP8 quantization stays in range and round-trips powers of two" {
    var rows: [64]u16 = undefined;
    for (&rows, 0..) |*b, i| b.* = lay.bf16Rne(@as(f32, @floatFromInt(i)) - 20.0);
    var codes: [64]u8 = undefined;
    var sc: [2]u8 = undefined;
    quantizeMx(&rows, 1, 64, &codes, &sc);
    for (codes, 0..) |c, i| {
        const s = std.math.ldexp(@as(f32, 1.0), @as(i32, sc[i / 32]) - 127);
        const back = lay.e4m3ToF32(c) * s;
        const want = lay.bf16ToF32(rows[i]);
        try std.testing.expect(@abs(back - want) <= @abs(want) / 16.0 + 1e-6);
    }
}

/// cuda_layouts.quantize4's format (scale and bias bf16, codes 0..15), each group's range chosen among the min/max range
/// shrunk toward its centre by 0..30% (the bias and scale as stored, rounded to bf16) by the least squared error.
pub fn quantize4Mse(rows: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const kg = k / 32;
    for (0..n) |r| for (0..kg) |g| {
        const v = rows[r * k + g * 32 ..][0..32];
        var x: [32]f32 = undefined;
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (v, &x) |b, *o| {
            o.* = lay.bf16ToF32(b);
            mn = @min(mn, o.*);
            mx = @max(mx, o.*);
        }
        var best_err: f32 = std.math.inf(f32);
        var best_s: u16 = 0;
        var best_b: u16 = 0;
        var best_q: [32]u32 = @splat(0);
        var step: usize = 0;
        while (step <= 12) : (step += 1) {
            const shrink: f32 = 1.0 - 0.025 * @as(f32, @floatFromInt(step));
            const c = 0.5 * (mn + mx);
            const lo = c - (c - mn) * shrink;
            const hi = c + (mx - c) * shrink;
            const sb = lay.bf16Rne(@max((hi - lo) / 15.0, 1e-8));
            const bb = lay.bf16Rne(lo);
            const s = lay.bf16ToF32(sb);
            const b0 = lay.bf16ToF32(bb);
            var q: [32]u32 = undefined;
            var err: f32 = 0;
            for (x, &q) |xi, *qi| {
                const qq = std.math.clamp(lay.roundEven((xi - b0) / s), 0, 15);
                qi.* = @intFromFloat(qq);
                const e = qq * s + b0 - xi;
                err += e * e;
            }
            if (err < best_err) {
                best_err = err;
                best_s = sb;
                best_b = bb;
                best_q = q;
            }
        }
        for (0..4) |w| {
            var word: u32 = 0;
            for (0..8) |p| word |= best_q[w * 8 + p] << @intCast(4 * p);
            words[r * (k / 8) + g * 4 + w] = word;
        }
        scales[r * kg + g] = best_s;
        biases[r * kg + g] = best_b;
    };
}

test "the least-error range is never worse than min/max" {
    var rows: [64]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&rows, 0..) |*b, i| b.* = lay.bf16Rne((prng.random().float(f32) - 0.5) * (if (i == 5) @as(f32, 20) else 1));
    var w1: [8]u32 = undefined;
    var s1: [2]u16 = undefined;
    var b1: [2]u16 = undefined;
    var w2: [8]u32 = undefined;
    var s2: [2]u16 = undefined;
    var b2: [2]u16 = undefined;
    lay.quantize4(&rows, 1, 64, &w1, &s1, &b1);
    quantize4Mse(&rows, 1, 64, &w2, &s2, &b2);
    const errOf = struct {
        fn f(r: []const u16, w: []const u32, s: []const u16, b: []const u16) f32 {
            var e: f32 = 0;
            for (r, 0..) |x, i| {
                const q: f32 = @floatFromInt((w[i / 8] >> @intCast(4 * (i % 8))) & 0xF);
                const d = q * lay.bf16ToF32(s[i / 32]) + lay.bf16ToF32(b[i / 32]) - lay.bf16ToF32(x);
                e += d * d;
            }
            return e;
        }
    }.f;
    try std.testing.expect(errOf(&rows, &w2, &s2, &b2) <= errOf(&rows, &w1, &s1, &b1) + 1e-6);
}

test "the table finds only what it holds" {
    var t: Table = .{};
    t.keys[0] = 0x1000;
    t.qs[0] = .{ .w = 1, .s = 2, .b = 3, .n = 64, .k = 32, .npad = 128 };
    t.n = 1;
    try std.testing.expect(t.find(0x1000) != null);
    try std.testing.expect(t.find(0x2000) == null);
}
