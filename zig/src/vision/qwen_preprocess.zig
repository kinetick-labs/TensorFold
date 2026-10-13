//! Flash-Next (Qwen3.5 / qwen4_exp) image preprocessing in Zig, the CPU half of the native vision path: transformers
//! 5.17's Qwen2VLImageProcessor (TorchvisionBackend) step for step, so the patches match Python's bytes.
//!   smart_resize (Python's round: half to even) -> torch's uint8 antialiased bicubic (Keys a = -0.5, int16 weights,
//!   width pass then height pass, as ATen's separable path and PIL) -> (x - 127.5) / 127.5 in f32 (tvF.normalize with
//!   mean and std fused with 1/255) -> patchify [gh*gw, C*T*p*p] (T = 2: the frame repeated), merge-block order.
//! And the rope index (qwen_processing.media_positions): text counts up on all three axes, an image block sits at the
//! next position with rows / columns on the h / w axes, text resumes past its larger side.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Geometry = struct {
    patch: u32 = 16,
    merge: u32 = 2,
    temporal: u32 = 2,
    channels: u32 = 3,
    pub fn factor(g: Geometry) u32 {
        return g.patch * g.merge;
    }
    /// one patch row's width: C * T * p * p (1536)
    pub fn rowWidth(g: Geometry) usize {
        return @as(usize, g.channels) * g.temporal * g.patch * g.patch;
    }
};

/// Python 3's round(): to nearest, ties to even.
fn pyRound(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return r;
}

pub const Size = struct { h: u32, w: u32 };

pub const ResizeError = error{AspectRatio};

/// transformers' smart_resize (qwen2_vl): both sides multiples of `factor`, area within [min_pixels, max_pixels].
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) ResizeError!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const f: f64 = @floatFromInt(factor);
    if (@max(h, w) / @min(h, w) > 200) return error.AspectRatio;
    var hb = pyRound(h / f) * f;
    var wb = pyRound(w / f) * f;
    const minp: f64 = @floatFromInt(min_pixels);
    const maxp: f64 = @floatFromInt(max_pixels);
    if (hb * wb > maxp) {
        const beta = @sqrt((h * w) / maxp);
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < minp) {
        const beta = @sqrt(minp / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
}

/// One axis' taps, as ATen's _compute_index_ranges_int16_weights (antialias, bicubic, align_corners false).
const Taps = struct {
    min: []u32,
    size: []u32,
    weights: []i16, // [out * stride]
    stride: usize,
    precision: u5,

    fn deinit(t: *Taps, a: Allocator) void {
        a.free(t.min);
        a.free(t.size);
        a.free(t.weights);
    }
};

fn cubic(x0: f64) f64 {
    const a: f64 = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
}

fn taps(a: Allocator, in_size: u32, out_size: u32) !Taps {
    const scale: f64 = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const support: f64 = if (scale >= 1.0) 2.0 * scale else 2.0;
    const max_size: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const invscale: f64 = if (scale >= 1.0) 1.0 / scale else 1.0;
    var t: Taps = .{ .min = try a.alloc(u32, out_size), .size = try a.alloc(u32, out_size), .weights = try a.alloc(i16, out_size * max_size), .stride = max_size, .precision = 0 };
    errdefer t.deinit(a);
    const wf = try a.alloc(f64, out_size * max_size);
    defer a.free(wf);
    var wt_max: f64 = 0;
    for (0..out_size) |i| {
        const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
        // static_cast<int64_t>: truncation toward zero
        const xmin: i64 = @max(@as(i64, @intFromFloat(@trunc(center - support + 0.5))), 0);
        var xsize: i64 = @min(@as(i64, @intFromFloat(@trunc(center + support + 0.5))), @as(i64, in_size)) - xmin;
        xsize = std.math.clamp(xsize, 0, @as(i64, @intCast(max_size)));
        const row = wf[i * max_size ..][0..max_size];
        var total: f64 = 0;
        var j: usize = 0;
        while (j < @as(usize, @intCast(xsize))) : (j += 1) {
            const w = cubic((@as(f64, @floatFromInt(@as(i64, @intCast(j)) + xmin)) - center + 0.5) * invscale);
            row[j] = w;
            total += w;
        }
        var m: f64 = 0;
        if (total != 0.0) for (row[0..j]) |*w| {
            w.* /= total;
            m = @max(m, w.*);
        };
        while (j < max_size) : (j += 1) row[j] = 0;
        wt_max = @max(wt_max, m);
        t.min[i] = @intCast(xmin);
        t.size[i] = @intCast(xsize);
    }
    var p: u5 = 0;
    while (p < 22) : (p += 1) {
        const next: i64 = @intFromFloat(@trunc(0.5 + wt_max * @as(f64, @floatFromInt(@as(i64, 1) << (p + 1)))));
        if (next >= (1 << 15)) break;
    }
    t.precision = p;
    const mul: f64 = @floatFromInt(@as(i64, 1) << p);
    for (wf, t.weights) |w, *o| {
        const v = w * mul;
        o.* = @intCast(if (v < 0) @as(i64, @intFromFloat(@trunc(-0.5 + v))) else @as(i64, @intFromFloat(@trunc(0.5 + v))));
    }
    return t;
}

/// One 1-d pass over interleaved u8 rows: `count` lines of `n` pixels (C channels), `line`/`step` strides in bytes.
fn pass(t: *const Taps, src: []const u8, dst: []u8, lines: usize, src_line: usize, dst_line: usize, src_step: usize, dst_step: usize, out_n: usize, channels: usize) void {
    const half: i32 = @as(i32, 1) << (t.precision - 1);
    for (0..lines) |l| {
        for (0..out_n) |o| {
            const w = t.weights[o * t.stride ..][0..t.size[o]];
            for (0..channels) |c| {
                var acc: i32 = half;
                for (w, 0..) |wt, j| acc += @as(i32, src[l * src_line + (t.min[o] + j) * src_step + c]) * wt;
                dst[l * dst_line + o * dst_step + c] = @intCast(std.math.clamp(acc >> t.precision, 0, 255));
            }
        }
    }
}

/// torch's antialiased bicubic resize of an RGB(-like) u8 image [h, w, c] to [oh, ow, c].
pub fn resizeBicubicAA(a: Allocator, src: []const u8, h: u32, w: u32, c: u32, oh: u32, ow: u32) ![]u8 {
    const C: usize = c;
    var cur: []u8 = try a.dupe(u8, src);
    var cw: usize = w;
    if (ow != w) {
        var t = try taps(a, w, ow);
        defer t.deinit(a);
        const out = try a.alloc(u8, @as(usize, h) * ow * C);
        pass(&t, cur, out, h, cw * C, @as(usize, ow) * C, C, C, ow, C);
        a.free(cur);
        cur = out;
        cw = ow;
    }
    if (oh != h) {
        var t = try taps(a, h, oh);
        defer t.deinit(a);
        const out = try a.alloc(u8, @as(usize, oh) * cw * C);
        // a vertical pass: each column (a "line" of cw*C bytes stepping one byte) over the rows
        pass(&t, cur, out, 1, 0, 0, cw * C, cw * C, oh, cw * C);
        a.free(cur);
        cur = out;
    }
    return cur;
}

/// (u8 - 127.5) / 127.5 in f32, then the patch rows [gh*gw, C*T*p*p] in the tower's (merge-block) order, f32.
pub fn patchify(a: Allocator, img: []const u8, h: u32, w: u32, g: Geometry) ![]f32 {
    const p: usize = g.patch;
    const m: usize = g.merge;
    const C: usize = g.channels;
    const T: usize = g.temporal;
    const gh = h / p;
    const gw = w / p;
    const width = g.rowWidth();
    const out = try a.alloc(f32, gh * gw * width);
    const mean: f32 = 127.5;
    const stdv: f32 = 127.5;
    var r: usize = 0;
    for (0..gh / m) |bh| for (0..gw / m) |bw| for (0..m) |ih| for (0..m) |iw| {
        const py0 = (bh * m + ih) * p;
        const px0 = (bw * m + iw) * p;
        const row = out[r * width ..][0..width];
        var k: usize = 0;
        for (0..C) |c| for (0..T) |_| for (0..p) |y| for (0..p) |x| {
            const v: f32 = @floatFromInt(img[((py0 + y) * w + px0 + x) * C + c]);
            row[k] = (v - mean) / stdv;
            k += 1;
        };
        r += 1;
    };
    return out;
}

/// f32 -> bf16, round to nearest even (torch's .to(bfloat16)); NaN kept quiet.
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub const Positions = struct {
    /// [3][n] (t, h, w axes)
    pos: []i32,
    delta: i64,
};

pub const Ids = struct { image: u32, start: u32, end: u32 };

/// qwen_processing.media_positions for images only; `grids`: each image's (t=1, h, w) in patches.
pub fn mediaPositions(a: Allocator, tokens: []const u32, grids: []const [3]u32, merge: u32, ids: Ids) !Positions {
    const n = tokens.len;
    const pos = try a.alloc(i32, 3 * n);
    errdefer a.free(pos);
    var cursor: usize = 0;
    var next: i32 = 0;
    var used: usize = 0;
    while (true) {
        const begin = for (tokens[cursor..], cursor..) |t, i| {
            if (t == ids.image) break i;
        } else break;
        if (used >= grids.len) return error.BadMedia;
        const gr = grids[used];
        used += 1;
        if (gr[0] != 1 or gr[1] % merge != 0 or gr[2] % merge != 0 or gr[1] == 0 or gr[2] == 0) return error.BadMedia;
        const hh = gr[1] / merge;
        const ww = gr[2] / merge;
        const end = begin + @as(usize, hh) * ww;
        if (begin == 0 or tokens[begin - 1] != ids.start or end >= n or tokens[end] != ids.end) return error.BadMedia;
        for (tokens[begin..end]) |t| if (t != ids.image) return error.BadMedia;
        for (cursor..begin, 0..) |i, k| for (0..3) |ax| {
            pos[ax * n + i] = next + @as(i32, @intCast(k));
        };
        const base = next + @as(i32, @intCast(begin - cursor));
        for (0..hh) |y| for (0..ww) |x| {
            const i = begin + y * ww + x;
            pos[i] = base;
            pos[n + i] = base + @as(i32, @intCast(y));
            pos[2 * n + i] = base + @as(i32, @intCast(x));
        };
        next = base + @as(i32, @intCast(@max(hh, ww)));
        cursor = end;
    }
    if (used < grids.len) return error.BadMedia;
    for (cursor..n, 0..) |i, k| for (0..3) |ax| {
        pos[ax * n + i] = next + @as(i32, @intCast(k));
    };
    return .{ .pos = pos, .delta = @as(i64, next) - @as(i64, @intCast(cursor)) };
}

test "smart_resize follows Python's rounding" {
    // 1000x750: round(31.25)=31, round(23.4375)=23 -> 992x736; 48 = 1.5*32 rounds to even (2*32=64), 80 -> 2.5 -> 64
    try std.testing.expectEqual(Size{ .h = 992, .w = 736 }, try smartResize(1000, 750, 32, 65536, 16777216));
    const s = try smartResize(48, 80, 32, 1024, 16777216);
    try std.testing.expectEqual(Size{ .h = 64, .w = 64 }, s);
}

test "positions: text, an image block, text" {
    const a = std.testing.allocator;
    const ids: Ids = .{ .image = 9, .start = 7, .end = 8 };
    const toks = [_]u32{ 1, 7, 9, 9, 9, 9, 9, 9, 8, 2 }; // 2x3 merged block (grid 1,4,6)
    const p = try mediaPositions(a, &toks, &.{.{ 1, 4, 6 }}, 2, ids);
    defer a.free(p.pos);
    const n = toks.len;
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, p.pos[0..n]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, p.pos[n .. 2 * n]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, p.pos[2 * n ..]);
    try std.testing.expectEqual(@as(i64, 5 - 8), p.delta);
}

test "resize identity and bf16 rounding" {
    const a = std.testing.allocator;
    const img = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const r = try resizeBicubicAA(a, &img, 1, 2, 3, 1, 2);
    defer a.free(r);
    try std.testing.expectEqualSlices(u8, &img, r);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // tie -> even
}
