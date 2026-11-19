//! An f32 CPU reference of the Hy4 vision tower (transformers' HYV4VLVisionModel): the 16 x 16 patch embedding plus
//! the learned position grid resampled to the picture, pre-norm blocks of LayerNorm, full attention within each
//! picture and a GELU MLP, then the merger: RMSNorm, a 2 x 2 stride-2 convolution, GELU, a 1 x 1 convolution to the
//! text width, and `image_newline` after every merged row. Slow and plain, for checking the GPU path.

const std = @import("std");
const Allocator = std.mem.Allocator;
const image_grid = @import("image_grid.zig");
const erf = @import("act_ref").erf;

pub const Dims = struct {
    hidden: usize,
    inter: usize,
    heads: usize,
    patch: usize,
    merge: usize = 2,
    out: usize,
    /// side of the learned position grid, `max_image_size / patch`
    edge: u32,
    eps: f32 = 1e-5,
};

pub const Block = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    q_w: []const f32,
    q_b: []const f32,
    k_w: []const f32,
    k_b: []const f32,
    v_w: []const f32,
    v_b: []const f32,
    o_w: []const f32,
    o_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    fc1_w: []const f32,
    fc1_b: []const f32,
    fc2_w: []const f32,
    fc2_b: []const f32,
};

pub const Weights = struct {
    patch_w: []const f32, // [hidden, 3 * patch * patch]
    patch_b: []const f32,
    /// `position_embedding.weight`, `[edge * edge + 1, hidden]`; row 0 (the unused cls slot) is skipped
    positions: []const f32,
    blocks: []const Block,
    merge_norm: []const f32,
    conv_w: []const f32, // [hidden * merge^2, hidden, merge, merge]
    conv_b: []const f32,
    out_w: []const f32, // [out, hidden * merge^2]
    out_b: []const f32,
    newline: []const f32,
};

/// The checkpoint's names for block `i` (`vit.layers.i.*`), for loaders that read by name.
pub fn blockNames(a: Allocator, i: usize) ![16][]const u8 {
    const parts = [16][]const u8{ "input_layernorm.weight", "input_layernorm.bias", "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight", "self_attn.o_proj.bias", "post_attention_layernorm.weight", "post_attention_layernorm.bias", "mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias", "mlp.dense_4h_to_h.weight", "mlp.dense_4h_to_h.bias" };
    var out: [16][]const u8 = undefined;
    for (parts, &out) |p, *o| o.* = try std.fmt.allocPrint(a, "vit.layers.{d}.{s}", .{ i, p });
    return out;
}

pub fn gelu(x: f32) f32 {
    const v: f64 = x;
    return @floatCast(0.5 * v * (1 + erf(v / std.math.sqrt2)));
}

/// y[t, o] = b[o] + sum_i x[t, i] w[o, i]
fn linear(x: []const f32, t: usize, in: usize, w: []const f32, b: ?[]const f32, out: usize, y: []f32) void {
    for (0..t) |r| {
        const xr = x[r * in ..][0..in];
        for (0..out) |o| {
            const wr = w[o * in ..][0..in];
            var s: f32 = if (b) |bb| bb[o] else 0;
            for (xr, wr) |xv, wv| s += xv * wv;
            y[r * out + o] = s;
        }
    }
}

fn layerNorm(x: []const f32, w: []const f32, b: []const f32, eps: f32, y: []f32) void {
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(x.len);
    var vari: f32 = 0;
    for (x) |v| vari += (v - mean) * (v - mean);
    vari /= @floatFromInt(x.len);
    const r = 1 / @sqrt(vari + eps);
    for (y, x, w, b) |*o, v, wv, bv| o.* = (v - mean) * r * wv + bv;
}

/// Merged features `[grid.tokens(merge), out]` for each picture of `grids`, in order, from row-major patches
/// `pixels` (`[patches, 3 * patch * patch]`, as `image_grid.patchify` lays them out).
pub fn forward(a: Allocator, d: *const Dims, w: *const Weights, pixels: []const f32, grids: []const image_grid.Grid) ![]f32 {
    var total: usize = 0;
    var tokens: usize = 0;
    for (grids) |g| {
        total += g.patches();
        tokens += g.tokens(@intCast(d.merge));
    }
    const pp = 3 * d.patch * d.patch;
    std.debug.assert(pixels.len == total * pp);
    const h = d.hidden;
    const x = try a.alloc(f32, total * h);
    defer a.free(x);
    linear(pixels, total, pp, w.patch_w, w.patch_b, h, x);
    {
        var at: usize = 0;
        const grid = w.positions[h..];
        for (grids) |g| {
            const pos = try a.alloc(f32, g.patches() * h);
            defer a.free(pos);
            image_grid.resamplePositions(grid, d.edge, h, g, pos);
            for (x[at * h ..][0..pos.len], pos) |*v, p| v.* += p;
            at += g.patches();
        }
    }
    const n = try a.alloc(f32, total * h);
    defer a.free(n);
    const q = try a.alloc(f32, total * h);
    defer a.free(q);
    const k = try a.alloc(f32, total * h);
    defer a.free(k);
    const v = try a.alloc(f32, total * h);
    defer a.free(v);
    const att = try a.alloc(f32, total * h);
    defer a.free(att);
    const mid = try a.alloc(f32, total * d.inter);
    defer a.free(mid);
    var max_len: usize = 0;
    for (grids) |g| max_len = @max(max_len, g.patches());
    const scores = try a.alloc(f32, max_len);
    defer a.free(scores);
    const hd = h / d.heads;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (w.blocks) |*blk| {
        for (0..total) |t| layerNorm(x[t * h ..][0..h], blk.norm1_w, blk.norm1_b, d.eps, n[t * h ..][0..h]);
        linear(n, total, h, blk.q_w, blk.q_b, h, q);
        linear(n, total, h, blk.k_w, blk.k_b, h, k);
        linear(n, total, h, blk.v_w, blk.v_b, h, v);
        var start: usize = 0;
        for (grids) |g| {
            const len = g.patches();
            for (start..start + len) |i| for (0..d.heads) |hh| {
                const qi = q[i * h + hh * hd ..][0..hd];
                var mx: f32 = -std.math.inf(f32);
                for (0..len) |j| {
                    const kj = k[(start + j) * h + hh * hd ..][0..hd];
                    var s: f32 = 0;
                    for (qi, kj) |a1, b1| s += a1 * b1;
                    scores[j] = s * scale;
                    mx = @max(mx, scores[j]);
                }
                var sum: f32 = 0;
                for (scores[0..len]) |*s| {
                    s.* = @exp(s.* - mx);
                    sum += s.*;
                }
                const o = att[i * h + hh * hd ..][0..hd];
                @memset(o, 0);
                for (0..len) |j| {
                    const p = scores[j] / sum;
                    for (o, v[(start + j) * h + hh * hd ..][0..hd]) |*ov, vv| ov.* += p * vv;
                }
            };
            start += len;
        }
        linear(att, total, h, blk.o_w, blk.o_b, h, n);
        for (x, n) |*xv, nv| xv.* += nv;
        for (0..total) |t| layerNorm(x[t * h ..][0..h], blk.norm2_w, blk.norm2_b, d.eps, n[t * h ..][0..h]);
        linear(n, total, h, blk.fc1_w, blk.fc1_b, d.inter, mid);
        for (mid) |*m| m.* = gelu(m.*);
        linear(mid, total, d.inter, blk.fc2_w, blk.fc2_b, h, n);
        for (x, n) |*xv, nv| xv.* += nv;
    }
    // the merger, per picture
    for (0..total) |t| {
        const xr = x[t * h ..][0..h];
        var ss: f32 = 0;
        for (xr) |e| ss += e * e;
        const r = 1 / @sqrt(ss / @as(f32, @floatFromInt(h)) + d.eps);
        for (n[t * h ..][0..h], xr, w.merge_norm) |*o, e, wv| o.* = wv * (e * r);
    }
    const m = d.merge;
    const wide = h * m * m;
    const out = try a.alloc(f32, tokens * d.out);
    const cell = try a.alloc(f32, wide);
    defer a.free(cell);
    var at: usize = 0;
    var row_out: usize = 0;
    for (grids) |g| {
        const gh = g.h / m;
        const gw = g.w / m;
        for (0..gh) |r| {
            for (0..gw) |c| {
                for (0..wide) |o| {
                    var s: f32 = w.conv_b[o];
                    for (0..m) |ky| for (0..m) |kx| {
                        const src = n[(at + (r * m + ky) * g.w + c * m + kx) * h ..][0..h];
                        for (src, 0..) |sv, ci| s += sv * w.conv_w[((o * h + ci) * m + ky) * m + kx];
                    };
                    cell[o] = gelu(s);
                }
                linear(cell, 1, wide, w.out_w, w.out_b, d.out, out[row_out * d.out ..][0..d.out]);
                row_out += 1;
            }
            @memcpy(out[row_out * d.out ..][0..d.out], w.newline);
            row_out += 1;
        }
        at += g.patches();
    }
    std.debug.assert(row_out == tokens);
    return out;
}
