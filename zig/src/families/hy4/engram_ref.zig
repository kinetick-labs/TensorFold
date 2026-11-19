//! A plain f32 CPU reference of Hy4-flash's Engram memory (transformers HYV4FlashEngramLayer, PR #50903): n-gram
//! windows that stop at a document start (the time position plane restarting), the normalized-vocabulary hash, the
//! table rows read through an FP8 E4M3 round trip, per-head RMSNorm, and the value and per-stream gated keys whose
//! residual is added to the hyper-connection streams before attention. torch's nn.RMSNorm without an eps uses
//! float32's machine epsilon, which is what these norms use.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const f32_eps: f32 = std.math.floatEps(f32);

pub const Engram = struct {
    hidden: u32,
    hc: u32,
    order: u32, // engram_max_ngram_size
    heads_per_order: u32,
    head_dim: u32,
    pad: u32,
    fp8: bool = true,
    map: []const i32, // lookup_table
    mult: []const i64, // [order]
    sizes: []const u64, // [(order - 1) * heads_per_order], order 2's heads first
    table: []const f32, // [sum(sizes), head_dim]
    head_norms: []const f32, // [hash heads, head_dim]
    key_w: []const f32, // [hc, hidden, E]
    key_b: []const f32, // [hc, hidden]
    norm_key: []const f32, // [hc, hidden]
    norm_query: []const f32, // [hc, hidden]
    value_w: []const f32, // [hidden, E]
    value_b: []const f32, // [hidden]
    pre_gamma: []const f32, // [hidden]
    scales: []const f32, // [hc, hidden]

    pub fn hashHeads(e: *const Engram) u32 {
        return (e.order - 1) * e.heads_per_order;
    }

    pub fn width(e: *const Engram) u32 {
        return e.hashHeads() * e.head_dim;
    }
};

/// `[t][order]` windows: the token, then the ones before it; slots before the document start (time position < offset)
/// read as the pad, as build_ngram_tokens makes them.
pub fn windows(ids: []const u32, time: []const i32, order: u32, pad: u32, out: []u32) void {
    for (ids, 0..) |id, i| {
        out[i * order] = id;
        for (1..order) |k| out[i * order + k] = if (time[i] < k) pad else ids[i - k];
    }
}

/// The row of each hash head for one window (offsets into the flat table included).
pub fn rows(e: *const Engram, window: []const u32, out: []usize) void {
    var mixed: [8]i64 = undefined;
    for (window, 0..) |tok, k| {
        const t = if (tok < e.map.len) tok else e.pad;
        mixed[k] = @as(i64, e.map[t]) *% e.mult[k];
    }
    var offset: u64 = 0;
    var h: usize = 0;
    for (2..e.order + 1) |n| {
        var x = mixed[0];
        for (1..n) |k| x ^= mixed[k];
        for (0..e.heads_per_order) |_| {
            const size = e.sizes[h];
            out[h] = @intCast(offset + @as(u64, @intCast(@mod(x, @as(i64, @intCast(size))))));
            offset += size;
            h += 1;
        }
    }
    // later heads' offsets continue past this order's
}

/// `x` through float8_e4m3fn and back (round to nearest even; finite inputs are clamped to +-448 first).
pub fn e4m3(x: f32) f32 {
    const v = @abs(x);
    if (v == 0) return x;
    const min_normal: f32 = 0.015625; // 2^-6
    const step: f32 = if (v < min_normal) 0.001953125 else std.math.ldexp(@as(f32, 1), std.math.ilogb(v) - 3);
    const q = @round(v / step);
    // ties to even
    const r = if (@abs(v / step - (q - 0.5)) == 0 and @mod(q, 2) != 0) q - 1 else q;
    return std.math.copysign(r * step, x);
}

fn fp8RoundTrip(row: []f32) void {
    var amax: f32 = 0;
    for (row) |v| amax = @max(amax, @abs(v));
    const scale = @max(amax, 1e-12) / 448.0;
    for (row) |*v| v.* = e4m3(std.math.clamp(v.* / scale, -448.0, 448.0)) * scale;
}

fn rmsNorm(out: []f32, x: []const f32, w: ?[]const f32, eps: f32) void {
    var s: f32 = 0;
    for (x) |e| s += e * e;
    const r = 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, 0..) |*o, e, i| o.* = e * r * (if (w) |g| g[i] else 1);
}

fn affine(out: []f32, w: []const f32, b: []const f32, x: []const f32) void {
    for (out, 0..) |*o, r| {
        var acc: f32 = 0;
        for (w[r * x.len ..][0..x.len], x) |p, q| acc += p * q;
        o.* = acc + b[r];
    }
}

/// The residual for `streams` `[t][hc][hidden]` into `out` (same shape); masked tokens (pictures, the pad) get zeros.
pub fn residual(a: Allocator, e: *const Engram, wins: []const u32, mask: []const bool, streams: []const f32, out: []f32) !void {
    const H = e.hidden;
    const t = mask.len;
    const heads = e.hashHeads();
    const emb = try a.alloc(f32, e.width());
    defer a.free(emb);
    const at = try a.alloc(usize, heads);
    defer a.free(at);
    const key = try a.alloc(f32, H);
    defer a.free(key);
    const nk = try a.alloc(f32, H);
    defer a.free(nk);
    const nq = try a.alloc(f32, H);
    defer a.free(nq);
    const main = try a.alloc(f32, H);
    defer a.free(main);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
    for (0..t) |i| {
        const o = out[i * e.hc * H ..][0 .. e.hc * H];
        if (!mask[i]) {
            @memset(o, 0);
            continue;
        }
        rows(e, wins[i * e.order ..][0..e.order], at);
        for (0..heads) |h| {
            const dst = emb[h * e.head_dim ..][0..e.head_dim];
            @memcpy(dst, e.table[at[h] * e.head_dim ..][0..e.head_dim]);
            if (e.fp8) fp8RoundTrip(dst);
            rmsNorm(dst, dst, e.head_norms[h * e.head_dim ..][0..e.head_dim], f32_eps);
        }
        affine(main, e.value_w, e.value_b, emb);
        rmsNorm(main, main, null, f32_eps);
        for (main, e.pre_gamma) |*m, g| m.* *= g;
        for (0..e.hc) |s| {
            affine(key, e.key_w[s * H * e.width() ..][0 .. H * e.width()], e.key_b[s * H ..][0..H], emb);
            rmsNorm(nk, key, e.norm_key[s * H ..][0..H], f32_eps);
            rmsNorm(nq, streams[(i * e.hc + s) * H ..][0..H], e.norm_query[s * H ..][0..H], f32_eps);
            var g: f32 = 0;
            for (nk, nq) |p, q| g += p * q;
            g *= scale;
            g = @sqrt(@max(@abs(g), 1e-6)) * std.math.sign(g);
            const gate = 1.0 / (1.0 + @exp(-g));
            for (o[s * H ..][0..H], main, e.scales[s * H ..][0..H]) |*r, m, sc| r.* = gate * m + sc * m;
        }
    }
}
