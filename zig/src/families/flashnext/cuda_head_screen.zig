//! TF_FLASHNEXT_HEAD_SCREEN=1 (one GPU, bf16 head): the main head's greedy verify rows in two stages with the same draws.
//! An MXFP8 copy of the head (cuda_mtp_q4.quantizeMx) gives approximate logits; fn_head_screen.cu bounds each one's
//! distance to the bf16 head's logit rigorously (per 32-input block norms of the copy's error, rounded up here, and
//! the kernels' fp32 accumulation) and recomputes exactly (the bf16 head's own chain, bit-identical) only the ids that
//! could be some row's argmax; the rest are -inf. The argmax of the result is the full head's argmax (ties included).
//! Only rows drawn greedily may read it: the engine screens a verify round when every stream in it drew greedily
//! last round, and recomputes the full head before any other draw from a screened round.
const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const fp8 = @import("cuda_fp8.zig");
const lay = @import("cuda_layouts.zig");
const q4 = @import("cuda_mtp_q4.zig");

pub const max_rows = 16;
/// both kernels' fp32 accumulation, relative to ||x|| (||W_v|| + ||Q_v||)
const accum_g: f64 = 1e-3;

pub const HeadScreen = struct {
    mod: cuda.Module,
    f_init: cuda.Function,
    f_bound: cuda.Function,
    f_select: cuda.Function,
    f_exact: cuda.Function,
    lin: fp8.Linear,
    head: W.Rows,
    buf: cuda.DeviceBuffer, // fragments + tiled scales
    side: cuda.DeviceBuffer, // dnb [KB, V] f32, wq [V] f32, a [16, V] f32, list [V] i32, tmax [16] i32, count
    dnb: u64,
    wq: u64,
    a: u64,
    list: u64,
    tmax: u64,
    count: u64,
    rounds: u64 = 0,

    pub fn build(gpa: std.mem.Allocator, d: *const cuda.Driver, head: W.Rows) !HeadScreen {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const n: usize = head.n;
        const k: usize = head.k;
        if (k % 64 != 0 or k / 32 > 128) return error.UnsupportedShape;
        var m = try cuda.Module.load(d, cuda.kernels.fn_head_screen);
        errdefer m.unload();
        const rows = try gpa.alloc(u16, n * k);
        defer gpa.free(rows);
        const src: cuda.DeviceBuffer = .{ .d = d, .ptr = head.weight, .len = n * k * 2 };
        try src.download(0, std.mem.sliceAsBytes(rows));
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        const sc = try gpa.alloc(u8, n * (k / 32));
        defer gpa.free(sc);
        q4.quantizeMx(rows, n, k, codes, sc);
        const kb = k / 32;
        const dnb = try gpa.alloc(f32, kb * n);
        defer gpa.free(dnb);
        const wq = try gpa.alloc(f32, n);
        defer gpa.free(wq);
        for (0..n) |r| {
            var ww: f64 = 0;
            var qq: f64 = 0;
            for (0..kb) |b| {
                const s = std.math.ldexp(@as(f64, 1.0), @as(i32, sc[r * kb + b]) - 127);
                var dd: f64 = 0;
                for (0..32) |j| {
                    const i = r * k + b * 32 + j;
                    const w: f64 = lay.bf16ToF32(rows[i]);
                    const q: f64 = @as(f64, lay.e4m3ToF32(codes[i])) * s;
                    dd += (w - q) * (w - q);
                    ww += w * w;
                    qq += q * q;
                }
                dnb[b * n + r] = upF32(@sqrt(dd));
            }
            wq[r] = upF32(accum_g * (@sqrt(ww) + @sqrt(qq)));
        }
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
        const al = struct {
            fn f(x: usize) usize {
                return std.mem.alignForward(usize, x, 256);
            }
        }.f;
        const o_wq = al(kb * n * 4);
        const o_a = o_wq + al(n * 4);
        const o_list = o_a + al(max_rows * n * 4);
        const o_tmax = o_list + al(n * 4);
        const o_count = o_tmax + 256;
        var side = try cuda.DeviceBuffer.alloc(d, o_count + 256);
        errdefer side.free();
        try side.upload(0, std.mem.sliceAsBytes(dnb));
        try side.upload(o_wq, std.mem.sliceAsBytes(wq));
        return .{
            .mod = m,
            .f_init = try m.function("fn_head_init"),
            .f_bound = try m.function("fn_head_bound"),
            .f_select = try m.function("fn_head_select"),
            .f_exact = try m.function("fn_head_exact"),
            .lin = .{ .w8 = buf.ptr, .bs = buf.ptr + frag.len, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad), .mx = true },
            .head = head,
            .buf = buf,
            .side = side,
            .dnb = side.ptr,
            .wq = side.ptr + o_wq,
            .a = side.ptr + o_a,
            .list = side.ptr + o_list,
            .tmax = side.ptr + o_tmax,
            .count = side.ptr + o_count,
        };
    }

    /// The f32 value at or above `x`.
    fn upF32(x: f64) f32 {
        const f: f32 = @floatCast(x);
        return if (@as(f64, f) >= x) f else std.math.nextAfter(f32, f, std.math.inf(f32));
    }

    pub fn deinit(h: *HeadScreen) void {
        h.side.free();
        h.buf.free();
        h.mod.unload();
    }

    /// The head's logits of `m` rows of `x` (`x_stride` elements apart) into `out` [m, V] bf16: exact at every id that
    /// could be a row's argmax, -inf elsewhere.
    pub fn run(h: *const HeadScreen, k8: *const fp8.Kernels, s: cuda.Stream, x: u64, x_stride: usize, out: u64, m: usize) !void {
        if (m < 1 or m > max_rows) return error.Invalid;
        const V: usize = h.head.n;
        const K: usize = h.head.k;
        {
            var a: cuda.Args = .{};
            a.add(h.tmax);
            a.add(h.count);
            a.add(@as(c_int, @intCast(m)));
            try cuda.launch.launch(h.f_init, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 }, .shared = 0 }, s, &a);
        }
        try fp8.matmul(k8, s, x, x_stride, h.lin, h.a, true, m);
        {
            var a: cuda.Args = .{};
            a.add(h.a);
            a.add(x);
            a.add(@as(c_int, @intCast(x_stride)));
            a.add(h.dnb);
            a.add(h.wq);
            a.add(h.tmax);
            for ([_]usize{ m, V, K }) |v| a.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(h.f_bound, .{ .grid = .{ .x = 192, .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = 0 }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(h.a);
            a.add(h.tmax);
            a.add(out);
            a.add(h.list);
            a.add(h.count);
            for ([_]usize{ m, V }) |v| a.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(h.f_select, .{ .grid = .{ .x = 192, .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = 0 }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@as(c_int, @intCast(x_stride)));
            a.add(h.head.weight);
            a.add(h.list);
            a.add(h.count);
            a.add(out);
            for ([_]usize{ V, K, m }) |v| a.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(h.f_exact, .{ .grid = .{ .x = 192, .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = 0 }, s, &a);
        }
    }
};
