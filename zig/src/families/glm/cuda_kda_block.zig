//! GLM-5.3-Flash's KDA block on CUDA, a rank's share of the Python family's kda_block: projection, chain, o_proj.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const kda = cuda.kda;
const b16 = @import("cuda_b16.zig");
const prof = @import("cuda_prof.zig");
const l2pf = @import("cuda_l2pf.zig");
const projection = @import("cuda_proj.zig");

/// The input projection [q | k | v | f_a | g_a | b] [3 heads 128 + 256 + heads][dims]: one bf16 stack, or two parts.
pub const In = union(enum) {
    bf16: u64,
    split: struct { qkv: projection.Proj, rest: u64 }, // [q | k | v] MXFP8 (FP4-Local's) or EXL3 (REXL3), bf16 the rest
};

/// One KDA layer, `heads` heads: its input projection, f_b, g_b, o (bf16 or MXFP8), the chain's taps and norm.
pub const Layer = struct {
    proj: In,
    fb: u64,
    gb: u64,
    o: projection.Proj,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
    dims: usize,

    pub fn width(l: Layer) usize {
        return 3 * l.heads * kda.dk + 2 * kda.dk + l.heads;
    }

    /// The width the input stack's rows actually carry (TF_GLM_KDA_PAD): the COMMON width, so every rank agrees.
    pub fn matWidth(l: Layer) usize {
        return rowWidth(l.heads);
    }

    /// The input stack and its two views need glm_mm's eight-column output and row stride.
    pub fn tiledInput(l: Layer) bool {
        return l.matWidth() % 8 == 0;
    }
};

/// TF_GLM_KDA_PAD's one eight-aligned row width for every rank, set once at load (Engine.open); 0: no pad.
pub var pad_to: usize = 0;

/// The pad width for `kda_heads` split over `world` ranks: the widest rank's row, rounded up to eight columns.
pub fn padWidth(kda_heads: usize, world: usize) usize {
    const heads = (kda_heads + world - 1) / world;
    return std.mem.alignForward(usize, 3 * heads * kda.dk + 2 * kda.dk + heads, 8);
}

/// The fused input's row width on a rank of `own_heads` heads: pad_to when the pad is on.
pub fn rowWidth(own_heads: usize) usize {
    if (pad_to != 0) return pad_to;
    return 3 * own_heads * kda.dk + 2 * kda.dk + own_heads;
}

/// The block's scratch for `rows` rows: projection, decay and gate rows, chain saves, B16 partials, a split's tmp;
/// `chunk` on a prompt chunk: its chain in chunks (no replay saves).
pub const Scratch = struct { p: u64, ka: u64, kg: u64, saves: kda.Saves, wide: kda.Wide, mm_part: u64, tmp: u64 = 0, chunk: ?Chunk = null };

/// glm_kda_chunk.cuh's dynamic shared memory (kch::SMEM).
pub const chunk_smem: u32 = 101120;

/// A prompt chunk's prep + step in chunks of 32 rows (tf_glm_kda_chunk: the chunked delta rule on tf32 tensor cores).
pub const Chunk = struct {
    f: cuda.Function,

    /// TF_GLM_KDA_CHUNK: unset or 1 one-term tf32 products, 3 three-term (fp32-grade), 0 none (prep + step).
    pub fn load(m: cuda.Module) !?Chunk {
        const v: []const u8 = if (std.c.getenv("TF_GLM_KDA_CHUNK")) |e| std.mem.span(e) else "1";
        if (std.mem.eql(u8, v, "0")) return null;
        const f = try m.function(if (std.mem.eql(u8, v, "3")) "tf_glm_kda_chunk3" else "tf_glm_kda_chunk");
        try f.allowDynamicShared(chunk_smem);
        return .{ .f = f };
    }

    /// The chain of `rows` rows through the chunk kernel and kda's out (false: a shape it does not take).
    pub fn run(ch: Chunk, ks: *const kda.Kernels, stream: cuda.Stream, c: kda.Chain, rows: usize, s: kda.Saves, w: kda.Wide) !bool {
        // the decay factors through row 16 of a chunk need lower >= -5; 16-byte async copies of the rows
        if (c.lower < -5.0 or c.lower >= 0 or c.p_stride % 8 != 0 or c.a_stride % 8 != 0) return false;
        if (c.heads == 0 or c.heads > 65535 or rows > std.math.maxInt(c_int) or c.p_stride > std.math.maxInt(c_int)) return false;
        const h: c_int = @intCast(c.heads);
        var a: cuda.Args = .{};
        a.add(h);
        a.add(c.p);
        a.add(@as(c_int, @intCast(c.p_stride)));
        a.add(@as(c_int, @intCast(c.b_off)));
        a.add(c.a);
        a.add(@as(c_int, @intCast(c.a_stride)));
        for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias }) |v| a.add(v);
        a.add(c.lower);
        a.add(@as(c_int, @intCast(rows)));
        a.add(w.y);
        a.add(s.state_out);
        try cuda.launch.launch(ch.f, .{ .grid = .{ .x = 2, .y = @intCast(c.heads) }, .block = .{ .x = 256 }, .shared = chunk_smem }, stream, &a);
        var o: cuda.Args = .{};
        o.add(h);
        o.add(w.y);
        o.add(c.g);
        o.add(@as(c_int, @intCast(c.g_stride)));
        o.add(c.norm_w);
        o.add(c.eps);
        o.add(s.out);
        try cuda.launch.launch(ks.out, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(c.heads) }, .block = .{ .x = kda.dv } }, stream, &o);
        return true;
    }
};

/// TF_GLM_KDA_PDL: unset or 1, a decode window's wide chain (prep, step, out) and its column copies launch dependent
/// (each waits for the kernel before it, so only the launch gaps go); 0 the plain launches.
pub fn pdlWanted() bool {
    const v: []const u8 = if (std.c.getenv("TF_GLM_KDA_PDL")) |e| std.mem.span(e) else "1";
    return !std.mem.eql(u8, v, "0");
}

/// Set at load from pdlWanted: the column copies' launches.
pub var copy_pdl = false;

/// forward.kda_block: out [rows][dims] fp32 partial; `wide`: three-kernel chain; `prompt`: commits the conv window.
pub fn block(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, out: u64) !void {
    try body(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt);
    try fill(set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the KDA block: o_proj of rows lo..hi (s.saves.out) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const n = l.heads * kda.dk;
    try projection.matmul(set, stream, l.o, s.saves.out + lo * n * 2, n, out, s.mm_part, hi - lo, l.dims, n, true);
}

/// The block up to its out_proj: the projections, the chain, and a prompt chunk's conv window commit.
pub fn body(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool) !void {
    return bodyProfile(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt, null, null);
}

pub fn bodyProfile(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, profile: ?*prof.Prof, o_site: ?l2pf.At) !void {
    var stamp = prof.span(profile, stream, .projections);
    try project(set, stream, l, s, x, x_stride, rows);
    stamp.end();
    if (o_site) |at| try at.fork(stream); // TF_GLM_L2PF: o read into L2 while the chain runs
    stamp = prof.span(profile, stream, .kda);
    defer stamp.end();
    try chainRows(ks, stream, l, s, rows, conv_state, state_in, wide);
    if (prompt) try glue.convShift(set, stream, conv_state, s.p, rows, rows * l.matWidth(), l.matWidth(), 1, 3 * l.heads * kda.dk);
}

/// The projections of `rows` rows into s.p, s.ka (decay) and s.kg (gate), row by row so segments share one call.
pub fn project(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize) !void {
    var selected = set;
    if (!l.tiledInput()) selected.mm = null;
    const n = l.heads * kda.dk;
    const w = l.matWidth();
    const fa = 3 * n;
    switch (l.proj) {
        .bf16 => |p| try b16.matmul(selected, stream, x, x_stride, p, s.p, s.mm_part, rows, w, l.dims, false),
        .split => |sp| { // each part on its own kernel into its columns of s.p: one row straight there, more through tmp
            // cc-defer: two row copies a window; an output row stride on the lane and glm_mm if a profile flags them
            const one = rows == 1;
            // a prompt chunk on the fused EXL3 GEMM writes its rows straight into s.p's columns: no copy
            if (one or !try projection.matmulStrided(set, stream, sp.qkv, x, x_stride, s.p, w, rows, fa, l.dims, false)) {
                try projection.matmul(set, stream, sp.qkv, x, x_stride, if (one) s.p else s.tmp, s.mm_part, rows, fa, l.dims, false);
                if (!one) try columns(set, stream, s.tmp, fa, s.p, w, rows);
            }
            // the bf16 rest (2 * dk + heads wide) tiles with the stack, or stays row-major with it when uneven
            try b16.matmul(selected, stream, x, x_stride, sp.rest, if (one) s.p + fa * 2 else s.tmp, s.mm_part, rows, w - fa, l.dims, false);
            if (!one) try columns(set, stream, s.tmp, w - fa, s.p + fa * 2, w, rows);
        },
    }
    try b16.matmul(selected, stream, s.p + fa * 2, w, l.fb, s.ka, s.mm_part, rows, n, kda.dk, false);
    try b16.matmul(selected, stream, s.p + (fa + kda.dk) * 2, w, l.gb, s.kg, s.mm_part, rows, n, kda.dk, false);
}

/// `rows` rows of `cols` bf16 packed at `src` into `dst`, rows `stride` apart (tf_glm_copy_rows).
fn columns(set: glue.Set, stream: cuda.Stream, src: u64, cols: usize, dst: u64, stride: usize, rows: usize) !void {
    const f = set.copy_rows orelse return error.FormatNotLoaded;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_longlong, @intCast(cols * 2)));
    a.add(dst);
    a.add(@as(c_longlong, @intCast(stride * 2)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(cols * 2)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * cols * 2 + 255) / 256) }, .block = .{ .x = 256 }, .pdl = copy_pdl }, stream, &a);
}

/// The chain over `rows` projected rows (s.p, s.ka, s.kg) on one sequence's conv window and recurrent state.
pub fn chainRows(ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, rows: usize, conv_state: u64, state_in: u64, wide: bool) !void {
    const n = l.heads * kda.dk;
    const fa = 3 * n;
    const c: kda.Chain = .{ .p = s.p, .p_stride = l.matWidth(), .b_off = fa + 2 * kda.dk, .a = s.ka, .a_stride = n, .g = s.kg, .g_stride = n, .conv_state = conv_state, .conv_w = l.conv_w, .state_in = state_in, .a_log = l.a_log, .dt_bias = l.dt_bias, .norm_w = l.norm_w, .eps = l.eps, .lower = l.lower, .heads = l.heads };
    if (s.chunk) |ch| if (rows >= kda.wide_rows and try ch.run(ks, stream, c, rows, s.saves, s.wide)) return;
    try kda.chain(ks, stream, c, rows, s.saves, if (wide or rows >= kda.wide_rows) s.wide else null);
}

test {
    _ = @import("cuda_kda_tp3_test.zig");
}
