//! Hy4-flash pictures before the tower (transformers HYV4VLImageProcessor and HYV4FlashModel, PR #50903):
//! the resize to a multiple of patch x merge inside [min_pixels, max_pixels], the row-major patch grid, the prompt's
//! `(H / merge) x (W / merge + 1)` image tokens (each merged row ends with a newline token), their 3D M-RoPE
//! positions, and the learned 128 x 128 position grid's bilinear resampling at `(n + 0.1) / 128`.

const std = @import("std");

/// processor_config.json's image limits: min_pixels 4096, max_pixels 2048 x 2048; patch 16, merge 2.
pub const Limits = struct {
    patch: u32 = 16,
    merge: u32 = 2,
    min_pixels: u64 = 4096,
    max_pixels: u64 = 4194304,

    pub fn factor(l: Limits) u32 {
        return l.patch * l.merge;
    }
};

pub const Size = struct { height: u32, width: u32 };

/// Python's round(): halves to even.
fn roundHalfEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2 * @round(x / 2);
    return r;
}

/// transformers' `smart_resize(height, width, factor, min_pixels, max_pixels)`.
pub fn smartResize(height: u32, width: u32, l: Limits) !Size {
    if (height == 0 or width == 0) return error.EmptyPicture;
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    if (@max(h, w) / @min(h, w) > 200) return error.AspectRatio;
    const f: f64 = @floatFromInt(l.factor());
    var hb = roundHalfEven(h / f) * f;
    var wb = roundHalfEven(w / f) * f;
    const most: f64 = @floatFromInt(l.max_pixels);
    const least: f64 = @floatFromInt(l.min_pixels);
    if (hb * wb > most) {
        const beta = @sqrt(h * w / most);
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < least) {
        const beta = @sqrt(least / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .height = @intFromFloat(hb), .width = @intFromFloat(wb) };
}

/// A resized picture's patch grid (`image_grid_thw`'s h and w, t = 1).
pub const Grid = struct {
    h: u32,
    w: u32,

    pub fn of(s: Size, l: Limits) Grid {
        return .{ .h = s.height / l.patch, .w = s.width / l.patch };
    }

    pub fn patches(g: Grid) u32 {
        return g.h * g.w;
    }

    /// Image tokens the prompt holds for it: merged rows of merged columns plus a newline each.
    pub fn tokens(g: Grid, merge: u32) u32 {
        return (g.h / merge) * (g.w / merge + 1);
    }

    /// `get_vision_position_ids` for one frame from `start`: (t, h, w) per token, rows of `w / merge + 1`.
    /// Returns the next text position (`max + 1`).
    pub fn positions(g: Grid, merge: u32, start: u32, t: []u32, h: []u32, w: []u32) u32 {
        const rows = g.h / merge;
        const cols = g.w / merge + 1;
        std.debug.assert(t.len == rows * cols and h.len == t.len and w.len == t.len);
        var i: usize = 0;
        for (0..rows) |r| for (0..cols) |c| {
            t[i] = start;
            h[i] = start + @as(u32, @intCast(r));
            w[i] = start + @as(u32, @intCast(c));
            i += 1;
        };
        return start + @max(rows, cols);
    }
};

/// One output coordinate of `F.interpolate(edge, scale_factor=(n + 0.1) / edge, mode="bilinear",
/// align_corners=False)`: torch samples at `(i + 0.5) / scale - 0.5` (the given scale, not edge / n), clamped at 0,
/// and copies the grid when n equals the edge.
pub const Tap = struct { lo: u32, hi: u32, frac: f32 };

pub fn positionTap(edge: u32, n: u32, i: u32) Tap {
    // torch copies when the output size equals the input's, whatever the scale
    if (n == edge) return .{ .lo = i, .hi = i, .frac = 0 };
    const scale = (@as(f64, @floatFromInt(n)) + 0.1) / @as(f64, @floatFromInt(edge));
    var src = (@as(f64, @floatFromInt(i)) + 0.5) / scale - 0.5;
    if (src < 0) src = 0;
    const lo: u32 = @intFromFloat(@floor(src));
    const lo_c = @min(lo, edge - 1);
    const hi = @min(lo_c + 1, edge - 1);
    return .{ .lo = lo_c, .hi = hi, .frac = @floatCast(src - @as(f64, @floatFromInt(lo_c))) };
}

/// The patch grid's position embeddings, `[g.h * g.w, dim]` fp32, from the learned `[edge * edge, dim]` grid
/// (`position_embedding.weight[1:]`, row-major), as `interpolate_pos_encoding` resamples it.
pub fn resamplePositions(grid: []const f32, edge: u32, dim: usize, g: Grid, out: []f32) void {
    std.debug.assert(grid.len == @as(usize, edge) * edge * dim and out.len == @as(usize, g.h) * g.w * dim);
    for (0..g.h) |y| {
        const ty = positionTap(edge, g.h, @intCast(y));
        for (0..g.w) |x| {
            const tx = positionTap(edge, g.w, @intCast(x));
            const o = out[(y * g.w + x) * dim ..][0..dim];
            const a = grid[(@as(usize, ty.lo) * edge + tx.lo) * dim ..][0..dim];
            const b = grid[(@as(usize, ty.lo) * edge + tx.hi) * dim ..][0..dim];
            const c = grid[(@as(usize, ty.hi) * edge + tx.lo) * dim ..][0..dim];
            const d = grid[(@as(usize, ty.hi) * edge + tx.hi) * dim ..][0..dim];
            for (o, a, b, c, d) |*v, va, vb, vc, vd| {
                const top = va + (vb - va) * tx.frac;
                const bot = vc + (vd - vc) * tx.frac;
                v.* = top + (bot - top) * ty.frac;
            }
        }
    }
}

/// Row-major patches (`patchify`): patch `(y, x)` holds channel-major 16 x 16 pixels of the normalized picture
/// `[3, H, W]` ((v / 255 - 0.5) / 0.5 already applied).
pub fn patchify(pixels: []const f32, s: Size, patch: u32, out: []f32) void {
    const gh = s.height / patch;
    const gw = s.width / patch;
    const plane = @as(usize, s.height) * s.width;
    std.debug.assert(pixels.len == 3 * plane and out.len == pixels.len);
    var i: usize = 0;
    for (0..gh) |py| for (0..gw) |px| for (0..3) |ch| for (0..patch) |dy| for (0..patch) |dx| {
        out[i] = pixels[ch * plane + (py * patch + dy) * s.width + px * patch + dx];
        i += 1;
    };
}
