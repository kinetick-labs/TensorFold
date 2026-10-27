//! Flash Next's attention caches on the device (--kv-cache): bf16 rows, or FP8 e4m3 rows with a power-of-two scale a
//! (position, KV head) row. The indexer's raw keys and pooled blocks stay bf16 in both; so do the DeltaNet states.
//!
//! FP8 layout, position t and KV head h (slot = t * kv_heads + h):
//!   keys   [cap, kv_heads, head_dim + 16] bytes: head_dim e4m3 codes, then s_k and s_v (fp32), then 8 zero bytes;
//!   values [cap, kv_heads, head_dim] bytes: head_dim e4m3 codes (their scale s_v is in the key row's trailer).
//! A row's scale is 2^(ceil(log2 amax) - 8): its largest value lands in (128, 256], under e4m3's 448, so nothing
//! saturates. A code times a power of two is a bf16 value, so reading a row (dequantize) is exact and the attention
//! sums are those of the bf16 kernels on the dequantized rows. The kernels are flashnext_attn.cu's fn_attn_prep_kv8
//! (quantize on write) and fn_attn_chunks_kv8 and fn_attn_fused_kv8 (dequantize on read). This file holds the sizes and
//! a CPU model of the arithmetic that the tests hold the CUDA formulas to.

const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");

/// The attention caches' format of an engine; every sequence, draft cache and kept state of it uses the same one.
pub const Kv = enum {
    bf16,
    fp8,

    /// Bytes one position's keys take in one attention layer (all KV heads).
    pub fn keyRow(kv: Kv) usize {
        return switch (kv) {
            .bf16 => wt.kv_heads * wt.head_dim * 2,
            .fp8 => wt.kv_heads * key_stride,
        };
    }

    /// Bytes one position's values take in one attention layer.
    pub fn valueRow(kv: Kv) usize {
        return switch (kv) {
            .bf16 => wt.kv_heads * wt.head_dim * 2,
            .fp8 => wt.kv_heads * wt.head_dim,
        };
    }

    /// Bytes one position's keys and values take in one attention layer: 2,048 bf16, 1,056 fp8.
    pub fn kvRow(kv: Kv) usize {
        return kv.keyRow() + kv.valueRow();
    }

    /// Bytes one position takes in one attention layer, the indexer's raw key included (the pooled blocks not).
    pub fn position(kv: Kv) usize {
        return kv.kvRow() + index_row;
    }
};

/// Bytes after an FP8 key row's codes: s_k and s_v (fp32) and 8 zero bytes, so rows stay 16-byte aligned.
pub const key_trailer = 16;
/// An FP8 key row: head_dim codes and the trailer.
pub const key_stride = wt.head_dim + key_trailer;
/// Where the scales sit in an FP8 key row: s_k, then s_v.
pub const key_scale_at = wt.head_dim;
pub const value_scale_at = wt.head_dim + 4;
/// One position's raw indexer key, bf16 in both formats.
pub const index_row = wt.index_dim * 2;
/// A row's scale is 2^(ceil(log2 amax) - scale_shift).
pub const scale_shift = 8;
/// The largest e4m3 magnitude (code 0x7E); 0x7F is NaN.
pub const e4m3_max = 448.0;

/// Byte offset of slot `slot`'s FP8 key row in a layer's key buffer, and of its value row in the value buffer.
pub fn keyRowAt(slot: usize) usize {
    return slot * key_stride;
}

pub fn valueRowAt(slot: usize) usize {
    return slot * wt.head_dim;
}

/// The slot of position `pos`, KV head `head`.
pub fn slotOf(pos: usize, head: usize) usize {
    return pos * wt.kv_heads + head;
}

// ---- the arithmetic on the CPU ---------------------------------------------------------------------------------

/// The scale's biased exponent for a row whose largest magnitude is `amax`: ceil(log2 amax) + 127 - 8, held in 1..254
/// (a zero row gets 2^-126 and zero codes). The CUDA kernels compute it on the float's bits just so.
pub fn scaleExponent(amax: f32) u32 {
    const bits: u32 = @bitCast(amax);
    const e: i32 = @intCast(((bits >> 23) & 0xFF) + @intFromBool(bits & 0x7FFFFF != 0));
    return @intCast(std.math.clamp(e - scale_shift, 1, 254));
}

/// 2^(ceil(log2 amax) - 8) and its reciprocal, both exact powers of two.
pub fn scaleOf(amax: f32) f32 {
    return @bitCast(scaleExponent(amax) << 23);
}

pub fn scaleInverse(amax: f32) f32 {
    return @bitCast((254 - scaleExponent(amax)) << 23);
}

/// The e4m3 code of `x`: round to nearest, ties to the even code, saturating at +-448 (infinities too); NaN is 0x7F.
/// cvt.rn.satfinite.e4m3x2.f32's result, which cuda_fp8.h's __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) is.
pub fn encode(x: f32) u8 {
    if (std.math.isNan(x)) return 0x7F;
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @abs(x);
    if (a >= e4m3_max) return sign | 0x7E;
    const bits: u32 = @bitCast(a);
    const e32: i32 = @intCast(bits >> 23);
    if (e32 >= 121) { // a normal e4m3 value: 2^-6 and up
        var e: u32 = @intCast(e32 - 120);
        var m: u32 = (bits >> 20) & 7;
        const rest = bits & 0xFFFFF;
        if (rest > 0x80000 or (rest == 0x80000 and m & 1 == 1)) m += 1;
        if (m == 8) {
            m = 0;
            e += 1;
        }
        return sign | @as(u8, @intCast(@min(e << 3 | m, 0x7E)));
    }
    // below 2^-6: multiples of 2^-9 (a code of 8 is the first normal value, 2^-6)
    const t = a * 512.0;
    const fl = @floor(t);
    var m: u32 = @intFromFloat(fl);
    const d = t - fl;
    if (d > 0.5 or (d == 0.5 and m & 1 == 1)) m += 1;
    return sign | @as(u8, @intCast(m));
}

/// The value of an e4m3 code (exact in fp32); 0x7F and 0xFF are NaN.
pub fn decode(code: u8) f32 {
    const e: i32 = (code >> 3) & 15;
    const m: u32 = code & 7;
    const mag: f32 = if (e == 15 and m == 7)
        std.math.nan(f32)
    else if (e == 0)
        @as(f32, @floatFromInt(m)) * 0x1p-9
    else
        (1.0 + @as(f32, @floatFromInt(m)) / 8.0) * std.math.ldexp(@as(f32, 1.0), e - 7);
    return if (code & 0x80 != 0) -mag else mag;
}

/// A bf16 value rounded to nearest even, as fp32 (what __float2bfloat16_rn then bf() give).
pub fn roundBf16(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return x;
    bits += 0x7FFF + ((bits >> 16) & 1);
    bits &= 0xFFFF0000;
    return @bitCast(bits);
}

/// A row of bf16 values (as fp32) to codes and its scale: the kernel's `quantize`, one (position, head) row.
pub fn quantizeRow(x: []const f32, codes: []u8) f32 {
    std.debug.assert(codes.len == x.len);
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(v)); // fmaxf drops a NaN, so does @max
    const inv = scaleInverse(amax);
    for (x, codes) |v, *c| c.* = encode(v * inv);
    return scaleOf(amax);
}

/// A row's codes and scale to the bf16 values the attention tiles hold: exact, e4m3 times a power of two.
pub fn dequantizeRow(codes: []const u8, scale: f32, out: []f32) void {
    for (codes, out) |c, *o| o.* = roundBf16(decode(c) * scale);
}

/// One position's FP8 key row for KV head values `k` and `v` (fp32 holding bf16 values): the codes, both scales and
/// the zero bytes, and the value row's codes.
pub fn packRows(k: []const f32, v: []const f32, key_row: []u8, value_row: []u8) void {
    std.debug.assert(key_row.len == key_stride and value_row.len == wt.head_dim);
    const sk = quantizeRow(k, key_row[0..wt.head_dim]);
    const sv = quantizeRow(v, value_row);
    std.mem.writeInt(u32, key_row[key_scale_at..][0..4], @bitCast(sk), .little);
    std.mem.writeInt(u32, key_row[value_scale_at..][0..4], @bitCast(sv), .little);
    @memset(key_row[value_scale_at + 4 ..], 0);
}

// ---- tests -----------------------------------------------------------------------------------------------------

/// The code of the e4m3 value nearest `a` (a >= 0, below 448), the even code on a tie, by looking at every code.
fn nearest(a: f32) u8 {
    var best: u8 = 0;
    var best_gap: f64 = std.math.inf(f64);
    var code: u8 = 0;
    while (code < 0x7F) : (code += 1) {
        const gap = @abs(@as(f64, decode(code)) - @as(f64, a));
        if (gap < best_gap or (gap == best_gap and code & 1 == 0)) {
            best_gap = gap;
            best = code;
        }
    }
    return best;
}

test "a position's FP8 rows: 1,056 bytes a layer, 272-byte key rows and 256-byte value rows" {
    try std.testing.expectEqual(@as(usize, 2048), Kv.bf16.kvRow());
    try std.testing.expectEqual(@as(usize, 1056), Kv.fp8.kvRow());
    try std.testing.expectEqual(@as(usize, 272), key_stride);
    try std.testing.expectEqual(@as(usize, 0), key_stride % 16);
    try std.testing.expectEqual(@as(usize, 2 * 272), Kv.fp8.keyRow());
    try std.testing.expectEqual(@as(usize, 2 * 256), Kv.fp8.valueRow());
    try std.testing.expectEqual(@as(usize, 2048 + 256), Kv.bf16.position());
    try std.testing.expectEqual(@as(usize, 1056 + 256), Kv.fp8.position());
    // the trailer holds both scales and 8 zero bytes
    try std.testing.expectEqual(@as(usize, key_stride), value_scale_at + 4 + 8);
    // slots are position-major, head-minor: the rows of one position are adjacent
    try std.testing.expectEqual(@as(usize, 2 * 272), keyRowAt(slotOf(1, 0)));
    try std.testing.expectEqual(@as(usize, 2 * 272 + 272), keyRowAt(slotOf(1, 1)));
    try std.testing.expectEqual(@as(usize, 3 * 256), valueRowAt(slotOf(1, 1)));
}

test "the README's per-token figures: 13 layers of 1,376 bytes, 17,888 fp8 against 30,784 bf16" {
    const pooled = wt.index_dim * 2 / wt.index_ratio; // a block of 4 positions pools to one 256-byte row
    try std.testing.expectEqual(@as(usize, 64), pooled);
    try std.testing.expectEqual(@as(usize, 1376), Kv.fp8.position() + pooled);
    try std.testing.expectEqual(@as(usize, 17888), 13 * (Kv.fp8.position() + pooled));
    try std.testing.expectEqual(@as(usize, 30784), 13 * (Kv.bf16.position() + pooled));
}

test "the e4m3 table: 126 finite magnitudes, 448 at the top, 2^-9 at the bottom, NaN at 0x7F" {
    try std.testing.expectEqual(@as(f32, 448.0), decode(0x7E));
    try std.testing.expectEqual(@as(f32, 0x1p-9), decode(1));
    try std.testing.expectEqual(@as(f32, 0x1p-6), decode(8));
    try std.testing.expectEqual(@as(f32, 1.0), decode(0x38));
    try std.testing.expectEqual(@as(f32, -2.0), decode(0xC0));
    try std.testing.expect(std.math.isNan(decode(0x7F)) and std.math.isNan(decode(0xFF)));
    var prev: f32 = -1;
    var code: u8 = 0;
    while (code < 0x7F) : (code += 1) { // strictly increasing: every code a distinct value
        try std.testing.expect(decode(code) > prev);
        prev = decode(code);
    }
}

test "encode is round to nearest even over every code, saturating, and decode inverts it" {
    var code: u8 = 0;
    while (code < 0x7F) : (code += 1) {
        try std.testing.expectEqual(code, encode(decode(code)));
        try std.testing.expectEqual(code | 0x80, encode(-decode(code))); // -0.0 is 0x80
    }
    // a grid of magnitudes over the whole range, ties included, against the brute-force nearest code
    var checked: usize = 0;
    var bits: u32 = 0x3A800000; // 2^-10
    while (bits < 0x43E00000) : (bits += 0x4801) { // to 448
        const a: f32 = @bitCast(bits);
        if (a >= e4m3_max) break;
        try std.testing.expectEqual(nearest(a), encode(a));
        checked += 1;
    }
    try std.testing.expect(checked > 2000);
    // midpoints between neighbours go to the even code
    code = 0;
    while (code < 0x7E) : (code += 1) {
        const mid = (decode(code) + decode(code + 1)) / 2;
        try std.testing.expectEqual(if (code & 1 == 0) code else code + 1, encode(mid));
    }
    // saturation and the special values
    try std.testing.expectEqual(@as(u8, 0x7E), encode(448.0));
    try std.testing.expectEqual(@as(u8, 0x7E), encode(470.0));
    try std.testing.expectEqual(@as(u8, 0x7E), encode(1.0e30));
    try std.testing.expectEqual(@as(u8, 0xFE), encode(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0x7F), encode(std.math.nan(f32)));
    try std.testing.expectEqual(@as(u8, 0), encode(0.0));
    try std.testing.expectEqual(@as(u8, 0x80), encode(-0.0));
    try std.testing.expectEqual(@as(u8, 0), encode(0x1p-11)); // under half of the smallest subnormal
    try std.testing.expectEqual(@as(u8, 1), encode(0x1p-10 * 1.0001));
    try std.testing.expectEqual(@as(u8, 0), encode(0x1p-10)); // the tie goes to the even code, zero
}

test "the scale is 2^(ceil(log2 amax) - 8) by the float's bits" {
    // the definition in floating point, for every amax the cache can see (a bf16 magnitude from 2^-100 to 2^100)
    var k: i32 = -100;
    while (k <= 100) : (k += 1) {
        const p = std.math.pow(f64, 2.0, @floatFromInt(k));
        for ([_]f64{ 1.0, 1.0078125, 1.5, 1.9921875 }) |frac| {
            const amax: f32 = @floatCast(p * frac);
            const want: f64 = std.math.pow(f64, 2.0, std.math.ceil(std.math.log2(@as(f64, amax))) - 8);
            try std.testing.expectEqual(@as(f32, @floatCast(want)), scaleOf(amax));
            try std.testing.expectEqual(@as(f32, @floatCast(1.0 / want)), scaleInverse(amax));
            // the row's largest value quantizes into (128, 256]
            const top = amax * scaleInverse(amax);
            try std.testing.expect(top > 128.0 and top <= 256.0);
        }
    }
    // a zero row: scale 2^-126, inverse 2^126, zero codes
    try std.testing.expectEqual(@as(f32, 0x1p-126), scaleOf(0));
    try std.testing.expectEqual(@as(f32, 0x1p126), scaleInverse(0));
    // an exact power of two keeps its exponent: 1.0 / 2^-8 = 256 is the top of the range
    try std.testing.expectEqual(@as(f32, 0x1p-8), scaleOf(1.0));
    try std.testing.expectEqual(@as(f32, 0x1p-7), scaleOf(1.0000001));
}

test "a row survives quantize and dequantize to within an e4m3 half step of its largest value" {
    var prng = std.Random.DefaultPrng.init(0x4b5638);
    const r = prng.random();
    var x: [wt.head_dim]f32 = undefined;
    var codes: [wt.head_dim]u8 = undefined;
    var back: [wt.head_dim]f32 = undefined;
    for (0..64) |trial| {
        const spread: f32 = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast(trial % 12)) - 6);
        for (&x) |*v| v.* = roundBf16((r.float(f32) * 2 - 1) * spread * (if (r.boolean()) @as(f32, 1.0) else 0.01));
        const s = quantizeRow(&x, &codes);
        dequantizeRow(&codes, s, &back);
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));
        for (x, back) |a, b| {
            // an e4m3 value has 3 mantissa bits: half a step is 2^-4 of the value, and a tiny value under 2^-6 of the
            // row's largest rounds on the subnormal grid, 2^-9 of the scaled row (2^-9 * s)
            const tol = @max(@abs(a) * 0x1p-4, 0x1p-10 * s);
            try std.testing.expect(@abs(a - b) <= tol);
        }
        // the largest value quantizes to a code of 128 or more: the row uses the format's top binade
        try std.testing.expect(@abs(back[std.mem.indexOfScalar(f32, &x, amax) orelse std.mem.indexOfScalar(f32, &x, -amax).?]) > 0.9 * amax);
    }
    // a zero row stays zero
    @memset(&x, 0);
    const s0 = quantizeRow(&x, &codes);
    dequantizeRow(&codes, s0, &back);
    for (back) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "dequantizing is exact: every code times every scale of a row above 2^-109 is a bf16 value" {
    // (below that the smallest codes fall under bf16's denormals: no model row is that small)
    var e: u32 = 10;
    while (e <= 247) : (e += 17) {
        const s: f32 = @bitCast(e << 23);
        var code: u8 = 0;
        while (code < 0x7F) : (code += 1) {
            const v = decode(code) * s;
            try std.testing.expectEqual(v, roundBf16(v));
        }
    }
}

test "a packed position holds its codes, both scales as little-endian fp32 and a zero tail" {
    var k: [wt.head_dim]f32 = undefined;
    var v: [wt.head_dim]f32 = undefined;
    for (&k, &v, 0..) |*a, *b, i| {
        a.* = roundBf16(@as(f32, @floatFromInt(i)) - 100); // -100 .. 155
        b.* = roundBf16(0.01 * @as(f32, @floatFromInt(i))); // 0 .. 2.55
    }
    var key_row: [key_stride]u8 = @splat(0xAA);
    var value_row: [wt.head_dim]u8 = undefined;
    packRows(&k, &v, &key_row, &value_row);
    const sk: f32 = @bitCast(std.mem.readInt(u32, key_row[key_scale_at..][0..4], .little));
    const sv: f32 = @bitCast(std.mem.readInt(u32, key_row[value_scale_at..][0..4], .little));
    try std.testing.expectEqual(@as(f32, 1.0), sk); // amax 155: ceil(log2) is 8, and 2^(8 - 8) is 1
    try std.testing.expectEqual(@as(f32, 0x1p-6), sv); // amax 2.55: ceil(log2) is 2
    for (key_row[key_stride - 8 ..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    var back: [wt.head_dim]f32 = undefined;
    dequantizeRow(key_row[0..wt.head_dim], sk, &back);
    try std.testing.expectEqual(@as(f32, 160.0), back[255]); // 155 sits between 144 and 160 on the grid of 16
    try std.testing.expectEqual(@as(f32, -96.0), back[0]); // 100 is the tie of 96 and 104: the even code
    dequantizeRow(&value_row, sv, &back);
    try std.testing.expect(@abs(back[255] - v[255]) <= v[255] * 0x1p-4);
}

/// Float bits and the code cuda_fp8.h's __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) gives them (compiled with g++ on the
/// host, where the header's software conversion runs; the SM 89 and later instruction rounds the same way): tie neighbours, the
/// subnormal edge, saturation, infinities and NaN.
const nvidia_codes = [_][2]u32{
    .{ 0x3a7ffffe, 0x00 },
    .{ 0x3a7fffff, 0x00 },
    .{ 0x3a800000, 0x00 },
    .{ 0x3bdffffe, 0x03 },
    .{ 0x3bdfffff, 0x03 },
    .{ 0x3be00000, 0x04 },
    .{ 0x3c4ffffe, 0x06 },
    .{ 0x3c4fffff, 0x06 },
    .{ 0x3c500000, 0x06 },
    .{ 0x3c6ffffe, 0x07 },
    .{ 0x3c6fffff, 0x07 },
    .{ 0x3c700000, 0x08 },
    .{ 0x3c87fffe, 0x08 },
    .{ 0x3c87ffff, 0x08 },
    .{ 0x3c880000, 0x08 },
    .{ 0x3cf7fffe, 0x0f },
    .{ 0x3cf7ffff, 0x0f },
    .{ 0x3cf80000, 0x10 },
    .{ 0x3d07fffe, 0x10 },
    .{ 0x3d07ffff, 0x10 },
    .{ 0x3d080000, 0x10 },
    .{ 0x3e87fffe, 0x28 },
    .{ 0x3e87ffff, 0x28 },
    .{ 0x3e880000, 0x28 },
    .{ 0x3fc7fffe, 0x3c },
    .{ 0x3fc7ffff, 0x3c },
    .{ 0x3fc80000, 0x3c },
    .{ 0x4067fffe, 0x46 },
    .{ 0x4067ffff, 0x46 },
    .{ 0x40680000, 0x46 },
    .{ 0x4247fffe, 0x64 },
    .{ 0x4247ffff, 0x64 },
    .{ 0x42480000, 0x64 },
    .{ 0x4337fffe, 0x73 },
    .{ 0x4337ffff, 0x73 },
    .{ 0x43380000, 0x74 },
    .{ 0x43d7fffe, 0x7d },
    .{ 0x43d7ffff, 0x7d },
    .{ 0x43d80000, 0x7e },
    .{ 0x00000000, 0x00 },
    .{ 0x80000000, 0x80 },
    .{ 0x43e00000, 0x7e },
    .{ 0x43e08000, 0x7e },
    .{ 0x43e80000, 0x7e },
    .{ 0x7149f2ca, 0x7e },
    .{ 0x7f800000, 0x7e },
    .{ 0xff800000, 0xfe },
    .{ 0x3a800000, 0x00 },
    .{ 0x3b000000, 0x01 },
    .{ 0x3c800000, 0x08 },
    .{ 0x0da24260, 0x00 },
    .{ 0x7fc00000, 0x7f },
};

test "encode gives the codes cuda_fp8.h gives" {
    for (nvidia_codes) |pair| {
        try std.testing.expectEqual(@as(u8, @intCast(pair[1])), encode(@bitCast(pair[0])));
    }
}
