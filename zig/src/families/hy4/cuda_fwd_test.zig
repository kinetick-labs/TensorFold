//! The Hy4 forward on a GPU against the CPU references at toy sizes (a few MiB): a sliding-window layer with the
//! dense MLP, an indexed layer (compressed indexer past its top-k) with the latent MoE, both whole and in chunks
//! down to single decode rows; the Engram residual; the vision tower on a 2-picture-row grid. Weights in f32, bf16
//! and f16 so every load path runs. On demand: HY4_FWD_FATBIN names forward.cu's fatbin (or cubin).

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("fwd.zig");
const gm = @import("gpu_model.zig");
const gv = @import("gpu_vision.zig");
const layer_ref = @import("layer_ref.zig");
const engram_ref = @import("engram_ref.zig");
const vision_ref = @import("vision_ref.zig");
const image_grid = @import("image_grid.zig");

const Gpu = struct {
    d: cuda.Driver,
    ctx: cuda.Context,
    k: fwd.Kernels,
    s: cuda.Stream,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    a: std.mem.Allocator,

    fn deinit(g: *Gpu) void {
        for (g.bufs.items) |*b| b.free();
        g.s.deinit();
        g.k.unload();
        g.ctx.deinit();
        g.d.close();
        g.bufs.deinit(g.a);
        g.a.destroy(g);
    }

    fn c(g: *Gpu) gm.Ctx {
        return .{ .d = &g.d, .k = &g.k, .dense = null, .s = g.s };
    }

    /// `xs` rounded to `dt` in place (so the reference sees the same values) and uploaded as `dt`.
    fn put(g: *Gpu, xs: []f32, dt: fwd.Dt) !fwd.W {
        const bytes: []const u8 = switch (dt) {
            .f32 => std.mem.sliceAsBytes(xs),
            .bf16 => blk: {
                const h = try g.a.alloc(u16, xs.len);
                for (xs, h) |*x, *o| {
                    const u: u32 = @bitCast(x.*);
                    const r: u32 = u +% 0x7fff +% ((u >> 16) & 1);
                    o.* = @intCast(r >> 16);
                    x.* = @bitCast(@as(u32, o.*) << 16);
                }
                break :blk std.mem.sliceAsBytes(h);
            },
            .f16 => blk: {
                const h = try g.a.alloc(f16, xs.len);
                for (xs, h) |*x, *o| {
                    o.* = @floatCast(x.*);
                    x.* = o.*;
                }
                break :blk std.mem.sliceAsBytes(h);
            },
        };
        const b = try cuda.DeviceBuffer.fromHost(&g.d, bytes);
        try g.bufs.append(g.a, b);
        return .{ .ptr = b.ptr, .dt = dt };
    }

    fn alloc(g: *Gpu, n: u64) !u64 {
        var b = try cuda.DeviceBuffer.alloc(&g.d, n);
        // on the test's stream: a legacy-stream memset does not order with a non-blocking stream and could land
        // after the uploads and kernels that follow (window 3: whole outputs or caches zeroed at random)
        try b.fill8(0, g.s.handle);
        try g.bufs.append(g.a, b);
        return b.ptr;
    }
};

/// On the heap: the stream, module and buffers keep `&g.d`, so the Gpu must not move after this.
fn open(a: std.mem.Allocator) !?*Gpu {
    const path = std.testing.environ.getPosix("HY4_FWD_FATBIN") orelse return null;
    const image = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    const g = try a.create(Gpu);
    g.* = .{ .d = try cuda.Driver.open(), .ctx = undefined, .k = undefined, .s = undefined, .a = a };
    g.ctx = try cuda.Context.init(&g.d, 0);
    g.k = try fwd.Kernels.load(&g.d, image);
    g.k.tc_min = 0; // the reference comparisons run the fp32 GEMM; the tensor-core one has its own test
    g.k.attn_tc_min = 0; // likewise the scalar attention (the tensor-core attention test compares the two)
    g.s = try cuda.Stream.init(&g.d, true);
    return g;
}

fn rand(a: std.mem.Allocator, r: std.Random, n: usize, scale: f32) ![]f32 {
    const xs = try a.alloc(f32, n);
    for (xs) |*x| x.* = (r.float(f32) * 2 - 1) * scale;
    return xs;
}

fn around(a: std.mem.Allocator, r: std.Random, n: usize, center: f32, spread: f32) ![]f32 {
    const xs = try a.alloc(f32, n);
    for (xs) |*x| x.* = center + (r.float(f32) * 2 - 1) * spread;
    return xs;
}

fn worst(got: []const f32, want: []const f32) f32 {
    var e: f32 = 0;
    for (got, want) |x, v| e = @max(e, @abs(x - v) / @max(1.0, @abs(v)));
    return e;
}

fn plain(w: fwd.W, k: u32, n: u32) gm.Lin {
    return .{ .plain = .{ .w = w, .k = k, .n = n } };
}

const T = 40;

/// 3D positions with a picture-like span (rows 10..17 share a time step and walk a 2 x 4 grid).
fn positions(a: std.mem.Allocator, t: usize) ![]i32 {
    const p = try a.alloc(i32, 3 * t);
    var time: i32 = 0;
    for (0..t) |i| {
        if (i >= 10 and i < 18) {
            const j: i32 = @intCast(i - 10);
            p[i] = 10;
            p[t + i] = 10 + @divTrunc(j, 4);
            p[2 * t + i] = 10 + @mod(j, 4);
            if (i == 17) time = 12;
            continue;
        }
        p[i] = time;
        p[t + i] = time;
        p[2 * t + i] = time;
        time += 1;
    }
    return p;
}

fn runLayer(g: *Gpu, L: *const gm.Layer, lim_in: gm.Limits, pos: []const i32, streams: []const f32, chunks: []const u32, dt: gm.CacheDt) ![]f32 {
    const a = g.a;
    var lim = lim_in;
    // the chunk the caches are cut for: the largest call, so a sliding ring smaller than T must wrap
    lim.chunk = 0;
    for (chunks) |t| lim.chunk = @max(lim.chunk, t);
    if (L.dims.index) |ix| lim.recall = ix.topk / ix.kpool;
    var w = try gm.Work.init(a, g.c(), lim);
    defer w.deinit();
    const cache_mem = try g.alloc(gm.cacheBytes(&L.dims, lim.ctx, lim.chunk, dt));
    const cache = gm.cacheAt(&L.dims, lim.ctx, lim.chunk, dt, cache_mem);
    // positions padded to the context, plane by plane
    const padded = try a.alloc(i32, 3 * lim.ctx);
    @memset(padded, 0);
    for (0..3) |pl| @memcpy(padded[pl * lim.ctx ..][0..T], pos[pl * T ..][0..T]);
    const pos_dev = try g.alloc(padded.len * 4);
    try g.c().upload(pos_dev, std.mem.sliceAsBytes(padded));
    const row = L.dims.hc * L.dims.hidden;
    const out = try a.alloc(f32, streams.len);
    var p0: u32 = 0;
    for (chunks) |t| {
        try g.c().upload(w.streams, std.mem.sliceAsBytes(streams[p0 * row ..][0 .. t * row]));
        try gm.layerOne(&w, L, cache, pos_dev, p0, t, w.emb);
        try g.c().download(std.mem.sliceAsBytes(out[p0 * row ..][0 .. t * row]), w.streams);
        p0 += t;
    }
    try g.s.synchronize();
    return out;
}

/// Two sequences side by side in each call (the second one shifted by `lag` rows): every row must match its own
/// sequence's single-sequence result.
fn runPair(g: *Gpu, L: *const gm.Layer, lim_in: gm.Limits, pos: []const i32, streams: []const f32, step: u32, lag: u32) ![]f32 {
    const a = g.a;
    var lim = lim_in;
    lim.chunk = 2 * step;
    lim.lanes = 2;
    if (L.dims.index) |ix| lim.recall = ix.topk / ix.kpool;
    var w = try gm.Work.init(a, g.c(), lim);
    defer w.deinit();
    var caches: [2]gm.Cache = undefined;
    for (&caches) |*cc| cc.* = gm.cacheAt(&L.dims, lim.ctx, lim.chunk, .f32, try g.alloc(gm.cacheBytes(&L.dims, lim.ctx, lim.chunk, .f32)));
    const padded = try a.alloc(i32, 3 * lim.ctx);
    @memset(padded, 0);
    for (0..3) |pl| @memcpy(padded[pl * lim.ctx ..][0..T], pos[pl * T ..][0..T]);
    const pos_dev = try g.alloc(padded.len * 4);
    try g.c().upload(pos_dev, std.mem.sliceAsBytes(padded));
    const row = L.dims.hc * L.dims.hidden;
    // out: sequence 0's rows then sequence 1's
    const out = try a.alloc(f32, 2 * streams.len);
    var at = [2]u32{ 0, 0 };
    var call: u32 = 0;
    while (at[0] < T or at[1] < T) : (call += 1) {
        var segs: [2]gm.Seg = undefined;
        var ns: usize = 0;
        var r: u32 = 0;
        for (0..2) |q| {
            if (q == 1 and call * step < lag) continue;
            if (at[q] >= T) continue;
            const rows = @min(step, T - at[q]);
            segs[ns] = .{ .cache = caches[q], .pos = pos_dev, .p0 = at[q], .row = r, .rows = rows };
            try g.c().upload(w.streams + 4 * @as(u64, r) * row, std.mem.sliceAsBytes(streams[at[q] * row ..][0 .. rows * row]));
            r += rows;
            ns += 1;
        }
        try gm.layer(&w, L, segs[0..ns], w.emb);
        for (segs[0..ns]) |sg| {
            const q: usize = if (sg.cache.view.lat == caches[0].view.lat) 0 else 1;
            try g.c().download(std.mem.sliceAsBytes(out[q * streams.len + sg.p0 * row ..][0 .. sg.rows * row]), w.streams + 4 * @as(u64, sg.row) * row);
            at[q] += sg.rows;
        }
    }
    return out;
}

test "Hy4 CUDA forward: a sliding-window layer with the dense MLP, and an indexed layer with the latent MoE, match layer_ref whole and in chunks (HY4_FWD_FATBIN)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = (try open(a)) orelse return error.SkipZigTest;
    defer g.deinit();
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();

    const H = 64;
    const n = 4;
    const heads = 4;
    const nope = 24;
    const rope = 8;
    const v = 16;
    const ql = 32;
    const kvl = 32;
    const inter = 48;
    const base: layer_ref.Dims = .{ .hidden = H, .hc = n, .heads = heads, .nope = nope, .rope = rope, .v = v, .q_lora = ql, .kv_lora = kvl, .inter = inter, .eps = 1e-5, .hc_eps = 1e-6, .hc_magnitude = 2.0, .theta = 10000.0, .sections = .{ 2, 1, 1 } };
    const qk = nope + rope;
    const pos = try positions(a, T);
    const streams = try rand(a, r, T * n * H, 1.5);

    for ([_]bool{ false, true }) |indexed| {
        var d = base;
        if (indexed) d.index = .{ .heads = 2, .dim = 128, .kpool = 4, .topk = 16 } else d.window = 8;
        // reference weights (rounded by put to their stored dtypes)
        const attn_fn = try rand(a, r, 2 * n * n * H, 0.05);
        const attn_base = try rand(a, r, 2 * n, 0.5);
        const ffn_fn = try rand(a, r, 2 * n * n * H, 0.05);
        const ffn_base = try rand(a, r, 2 * n, 0.5);
        var hc_scale = [2]f32{ 0.8, 1.2 };
        const in_norm = try around(a, r, H, 1, 0.2);
        const pa_norm = try around(a, r, H, 1, 0.2);
        const pm_norm = try around(a, r, H, 1, 0.2);
        const q_a = try rand(a, r, ql * H, 0.15);
        const q_a_norm = try around(a, r, ql, 1, 0.2);
        const q_b = try rand(a, r, heads * qk * ql, 0.2);
        const kv_a = try rand(a, r, (kvl + rope) * H, 0.15);
        const kv_a_norm = try around(a, r, kvl, 1, 0.2);
        const kv_b = try rand(a, r, heads * (nope + v) * kvl, 0.2);
        const gate = try rand(a, r, heads * v * H, 0.15);
        const o = try rand(a, r, H * heads * v, 0.15);
        const pp_norm = try around(a, r, H, 1, 0.2);
        const sinks = try rand(a, r, heads, 1.0);

        var L: gm.Layer = undefined;
        L.dims = d;
        L.engram = null;
        L.attn_hc = .{ .fn_ = (try g.put(attn_fn, .f32)).ptr, .base = (try g.put(attn_base, .f32)).ptr, .scale = (try g.put(&hc_scale, .f32)).ptr };
        L.ffn_hc = .{ .fn_ = (try g.put(ffn_fn, .f32)).ptr, .base = (try g.put(ffn_base, .f32)).ptr, .scale = L.attn_hc.scale };
        L.input_norm = try g.put(in_norm, .bf16);
        L.post_attention_norm = try g.put(pa_norm, .bf16);
        L.attn = .{
            .q_a = plain(try g.put(q_a, .bf16), H, ql),
            .q_a_norm = try g.put(q_a_norm, .bf16),
            .q_b = plain(try g.put(q_b, .f32), ql, heads * qk),
            .kv_a = plain(try g.put(kv_a, .bf16), H, kvl + rope),
            .kv_a_norm = try g.put(kv_a_norm, .bf16),
            .kv_b = try g.put(kv_b, .f16),
            .gate = plain(try g.put(gate, .f32), H, heads * v),
            .o = plain(try g.put(o, .f32), heads * v, H),
            .post_norm = try g.put(pp_norm, .bf16),
            .sinks = (try g.put(sinks, .f32)).ptr,
        };
        var w: layer_ref.Weights = .{ .attn_hc = .{ .fn_ = attn_fn, .base = attn_base, .scale = &hc_scale }, .ffn_hc = .{ .fn_ = ffn_fn, .base = ffn_base, .scale = &hc_scale }, .input_norm = in_norm, .post_attention_norm = pa_norm, .post_mlp_norm = pm_norm, .q_a = q_a, .q_a_norm = q_a_norm, .q_b = q_b, .kv_a = kv_a, .kv_a_norm = kv_a_norm, .kv_b = kv_b, .gate = gate, .o = o, .post_proj_norm = pp_norm, .sinks = sinks };
        var lim: gm.Limits = .{ .chunk = T, .ctx = 64, .hidden = H, .hc = n, .heads = heads, .qk = qk, .v = v, .kv_lora = kvl, .rope = rope, .q_lora = ql, .inter = inter };
        if (d.index) |ix| {
            const iw: layer_ref.IndexWeights = .{
                .query = try rand(a, r, ix.heads * ix.dim * ql, 0.2),
                .head_weight = try rand(a, r, ix.heads * H, 0.3),
                .token_key = try rand(a, r, ix.dim * H, 0.2),
                .token_key_norm = try around(a, r, ix.dim, 1, 0.2),
                .key_value = try rand(a, r, ix.dim * ix.dim, 0.1),
                .key_gate = try rand(a, r, ix.dim * ix.dim, 0.1),
                .key_norm = try around(a, r, ix.dim, 1, 0.2),
                .ape = try rand(a, r, ix.kpool * ix.dim, 0.5),
            };
            L.attn.index = .{
                .query = plain(try g.put(@constCast(iw.query), .f16), ql, ix.heads * ix.dim),
                .head_weight = plain(try g.put(@constCast(iw.head_weight), .f16), H, ix.heads),
                .token_key = plain(try g.put(@constCast(iw.token_key), .f16), H, ix.dim),
                .token_key_norm = try g.put(@constCast(iw.token_key_norm), .f16),
                .key_value = (try g.put(@constCast(iw.key_value), .f32)).ptr,
                .key_gate = (try g.put(@constCast(iw.key_gate), .f32)).ptr,
                .key_norm = (try g.put(@constCast(iw.key_norm), .f32)).ptr,
                .ape = (try g.put(@constCast(iw.ape), .f32)).ptr,
            };
            w.index = iw;
            lim.index_heads = ix.heads;
            lim.index_dim = ix.dim;
            lim.kpool = ix.kpool;
            // the latent MoE
            const E = 8;
            const top = 3;
            const lat = 24;
            const wd = 16;
            const m: layer_ref.Moe = .{
                .experts = E,
                .top = top,
                .latent = lat,
                .width = wd,
                .scale = 2.5,
                .clamp = 1.5,
                .router = try rand(a, r, E * H, 0.3),
                .bias = try rand(a, r, E, 0.05),
                .pre = try rand(a, r, lat * H, 0.2),
                .post = try rand(a, r, H * lat, 0.2),
                .gate_up = try rand(a, r, E * 2 * wd * lat, 0.4),
                .down = try rand(a, r, E * lat * wd, 0.3),
                .shared_gate = try rand(a, r, wd * H, 0.2),
                .shared_up = try rand(a, r, wd * H, 0.2),
                .shared_down = try rand(a, r, H * wd, 0.2),
                .norm = try around(a, r, H, 1, 0.2),
            };
            const experts = try a.alloc([3]gm.Lin, E);
            for (experts, 0..) |*ex, e| {
                const gu = @constCast(m.gate_up[e * 2 * wd * lat ..][0 .. 2 * wd * lat]);
                ex.* = .{ plain(try g.put(gu[0 .. wd * lat], .f32), lat, wd), plain(try g.put(gu[wd * lat ..], .f32), lat, wd), plain(try g.put(@constCast(m.down[e * lat * wd ..][0 .. lat * wd]), .f32), wd, lat) };
            }
            L.mlp = .{ .moe = .{
                .experts = E,
                .top = top,
                .latent = lat,
                .width = wd,
                .scale = m.scale,
                .clamp = m.clamp,
                .router = plain(try g.put(@constCast(m.router), .f16), H, E),
                .bias = (try g.put(@constCast(m.bias), .f32)).ptr,
                .pre = plain(try g.put(@constCast(m.pre), .bf16), H, lat),
                .post = plain(try g.put(@constCast(m.post), .bf16), lat, H),
                .expert = experts,
                .shared = .{ plain(try g.put(@constCast(m.shared_gate), .f32), H, wd), plain(try g.put(@constCast(m.shared_up), .f32), H, wd), plain(try g.put(@constCast(m.shared_down), .f32), wd, H) },
                .norm = try g.put(@constCast(m.norm), .bf16),
            } };
            w.moe = m;
            lim.experts = E;
            lim.top = top;
            lim.latent = lat;
            lim.width = wd;
        } else {
            const mg = try rand(a, r, inter * H, 0.2);
            const mu = try rand(a, r, inter * H, 0.2);
            const md = try rand(a, r, H * inter, 0.2);
            L.mlp = .{ .dense = .{ .gate = plain(try g.put(mg, .bf16), H, inter), .up = plain(try g.put(mu, .bf16), H, inter), .down = plain(try g.put(md, .bf16), inter, H), .norm = try g.put(pm_norm, .bf16) } };
            w.mlp_gate = mg;
            w.mlp_up = mu;
            w.mlp_down = md;
        }
        w.post_mlp_norm = pm_norm;

        const want = try a.dupe(f32, streams);
        try layer_ref.forward(a, &d, &w, pos, T, want);
        // whole, in chunks, down to decode rows; small chunks make the sliding ring (window + chunk) and the token-key
        // ring wrap several times over the 40 rows
        const plans = [_][]const u32{ &.{T}, &.{ 23, 17 }, &.{ 13, 24, 1, 1, 1 }, &.{ 5, 5, 5, 5, 5, 5, 5, 5 }, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 1 } };
        const name = if (indexed) "indexed + MoE" else "sliding + dense";
        for (plans) |plan| {
            const got = try runLayer(g, &L, lim, pos, streams, plan, .f32);
            const e = worst(got, want);
            std.debug.print("hy4 forward layer ({s}) chunks {any}: worst relative error {e}\n", .{ name, plan, e });
            if (std.c.getenv("HY4_TEST_SOFT") == null) try std.testing.expect(e <= 2e-3);
        }
        // narrower caches: bf16 latents and keys, e4m3 latents with a row scale (looser: the cache rounds). Window 3
        // (GB10): bf16 1.5e-2 / 2.0e-2, fp8 0.30 / 0.28 on these random toy layers, i.e. the 16x coarser mantissa
        // (bf16 x 16 ~ 0.24-0.32); the real model's fp8 smoke at 512k was 3/3
        for ([_]gm.CacheDt{ .bf16, .fp8 }) |dt| {
            const got = try runLayer(g, &L, lim, pos, streams, &.{ 13, 24, 1, 1, 1 }, dt);
            const e = worst(got, want);
            std.debug.print("hy4 forward layer ({s}) cache {s}: worst relative error {e}\n", .{ name, @tagName(dt), e });
            if (std.c.getenv("HY4_TEST_SOFT") == null) try std.testing.expect(e <= @as(f32, if (dt == .bf16) 3e-2 else 0.4));
        }
        // two sequences per call, the second starting later: each matches the reference alone
        for ([_]u32{ 0, 8 }) |lag| {
            const got = try runPair(g, &L, lim, pos, streams, 6, lag);
            const e = @max(worst(got[0..streams.len], want), worst(got[streams.len..], want));
            std.debug.print("hy4 forward layer ({s}) two sequences, lag {d}: worst relative error {e}\n", .{ name, lag, e });
            if (std.c.getenv("HY4_TEST_SOFT") == null) try std.testing.expect(e <= 2e-3);
        }
    }
}

test "Hy4 CUDA forward: the tensor-core GEMM (bf16 and f16 weights) matches the fp32 one on prompt-sized rows (HY4_FWD_FATBIN)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = (try open(a)) orelse return error.SkipZigTest;
    defer g.deinit();
    var rng = std.Random.DefaultPrng.init(13);
    const r = rng.random();
    // ragged sizes: rows, columns and k off the 64 / 32 tiles; a bias; accumulate on top of existing values
    const m = 203;
    const n = 157;
    const k = 1000;
    const ldx = k + 8;
    const ldy = n + 3;
    const x = try rand(a, r, m * ldx, 1.0);
    const xd = try g.alloc(x.len * 4);
    try g.c().upload(xd, std.mem.sliceAsBytes(x));
    for ([_]fwd.Dt{ .bf16, .f16 }) |dt| {
        const wv = try rand(a, r, n * k, 0.05);
        const w = try g.put(wv, dt);
        const bv = try rand(a, r, n, 0.5);
        const b = try g.put(bv, .f32);
        const y0 = try rand(a, r, m * ldy, 1.0);
        var outs: [2][]f32 = undefined;
        for (0..2) |pass| {
            const yd = try g.alloc(y0.len * 4);
            try g.c().upload(yd, std.mem.sliceAsBytes(y0));
            g.k.tc_min = if (pass == 0) 0 else 32;
            try g.k.gemm(g.s, xd, ldx, w, b, yd, ldy, m, n, k, true);
            outs[pass] = try a.alloc(f32, y0.len);
            try g.c().download(std.mem.sliceAsBytes(outs[pass]), yd);
        }
        g.k.tc_min = 0;
        var big: f32 = 0;
        for (outs[0]) |v| big = @max(big, @abs(v));
        var e: f32 = 0;
        var at: usize = 0;
        for (outs[0], outs[1], 0..) |want, got, ix| {
            if (@abs(got - want) / big > e) at = ix;
            e = @max(e, @abs(got - want) / big);
        }
        var bad: usize = 0;
        for (outs[0], outs[1]) |want, got| bad += @intFromBool(@abs(got - want) / big > 0.02);
        std.debug.print("hy4 tensor-core GEMM worst at row {d} col {d}: want {d} got {d} (y0 {d}); {d} of {d} values off\n", .{ at / ldy, at % ldy, outs[0][at], outs[1][at], y0[at], bad, y0.len });
        std.debug.print("hy4 tensor-core GEMM ({s} weights) {d}x{d}x{d}: worst error {e} of the largest |y|\n", .{ @tagName(dt), m, n, k, e });
        // x rounded to the weight type: bf16 keeps 8 bits, f16 11
        try std.testing.expect(e <= @as(f32, if (dt == .bf16) 1e-2 else 2e-3));
    }
}

test "Hy4 CUDA forward: the tensor-core attention matches the scalar one: sliding ring, indexed list, bf16 / fp8 caches (HY4_FWD_FATBIN)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = (try open(a)) orelse return error.SkipZigTest;
    defer g.deinit();
    var rng = std.Random.DefaultPrng.init(17);
    const r = rng.random();
    const kvl = 512;
    const rope = 64;
    const nope = 16;
    const qk = nope + rope;
    const Case = struct { name: []const u8, heads: u32, slots: u32, p0: u32, t: u32, window: u32, listed: bool, dt: fwd.CacheDt };
    const cases = [_]Case{
        .{ .name = "sliding ring bf16", .heads = 64, .slots = 96, .p0 = 120, .t = 37, .window = 40, .listed = false, .dt = .bf16 },
        .{ .name = "sliding ring fp8", .heads = 64, .slots = 96, .p0 = 120, .t = 37, .window = 40, .listed = false, .dt = .fp8 },
        .{ .name = "indexed list bf16", .heads = 32, .slots = 256, .p0 = 200, .t = 20, .window = 0, .listed = true, .dt = .bf16 },
        .{ .name = "indexed list f32", .heads = 32, .slots = 256, .p0 = 200, .t = 20, .window = 0, .listed = true, .dt = .f32 },
        .{ .name = "dense causal bf16", .heads = 32, .slots = 256, .p0 = 0, .t = 70, .window = 0, .listed = false, .dt = .bf16 },
    };
    for (cases) |cs| {
        // the cache: random latent + rotary rows stored through the cache kernel
        const src = try rand(a, r, @as(usize, cs.slots) * (kvl + rope), 1.0);
        const src_d = try g.alloc(src.len * 4);
        try g.c().upload(src_d, std.mem.sliceAsBytes(src));
        const lat = try g.alloc(@as(u64, cs.slots) * kvl * cs.dt.latBytes());
        const rot = try g.alloc(@as(u64, cs.slots) * rope * cs.dt.rotBytes());
        const scl = try g.alloc(@as(u64, cs.slots) * 4);
        const cv: fwd.Kernels.CacheView = .{ .lat = lat, .rot = rot, .scale = if (cs.dt == .fp8) scl else 0, .dt = cs.dt, .slots = cs.slots };
        try g.k.cacheStore(g.s, src_d, kvl + rope, kvl, rope, cv, 0, cs.slots);
        const ql = try rand(a, r, @as(usize, cs.t) * cs.heads * kvl, 0.08);
        const q = try rand(a, r, @as(usize, cs.t) * cs.heads * qk, 0.3);
        const sinks = try rand(a, r, cs.heads, 1.0);
        const ql_d = (try g.put(ql, .f32)).ptr;
        const q_d = (try g.put(q, .f32)).ptr;
        const sinks_d = (try g.put(sinks, .f32)).ptr;
        // indexed: up to 12 kept groups before the row's own, distinct, ascending
        const ld: u32 = 12;
        var keep: fwd.Kernels.Keep = .{};
        var nkmax: u64 = 0;
        if (cs.listed) {
            const list = try a.alloc(i32, cs.t * ld);
            const count = try a.alloc(i32, cs.t);
            for (0..cs.t) |row| {
                const own = (cs.p0 + row) / 4;
                var k: u32 = 0;
                var gg: u32 = 0;
                while (gg < own and k < ld) : (gg += 1) {
                    if (r.uintLessThan(u32, 4) != 0) continue;
                    list[row * ld + k] = @intCast(gg);
                    k += 1;
                }
                count[row] = @intCast(k);
            }
            const list_d = try g.alloc(list.len * 4);
            try g.c().upload(list_d, std.mem.sliceAsBytes(list));
            const count_d = try g.alloc(count.len * 4);
            try g.c().upload(count_d, std.mem.sliceAsBytes(count));
            keep = .{ .list = list_d, .ld = ld, .count = count_d, .kpool = 4 };
            nkmax = (ld + 1) * 4;
        } else nkmax = if (cs.window > 0) cs.window else cs.p0 + cs.t;
        const scale: f32 = 1.0 / @sqrt(@as(f32, qk));
        var outs: [2][]f32 = undefined;
        const piece: u32 = 16; // several pieces, the last one short
        const S = try g.alloc(piece * cs.heads * nkmax * 4);
        const P = try g.alloc(piece * cs.heads * nkmax * 2);
        for (0..2) |pass| {
            const ol = try g.alloc(@as(u64, cs.t) * cs.heads * kvl * 4);
            if (pass == 0) {
                try g.k.mlaAttn(g.s, ql_d, q_d, qk, nope, cv, sinks_d, ol, cs.t, cs.heads, kvl, rope, cs.p0, cs.window, keep, scale);
            } else {
                try g.k.mlaAttnTc(g.s, ql_d, q_d, qk, nope, cv, sinks_d, ol, cs.t, cs.heads, kvl, rope, cs.p0, cs.window, keep, scale, S, P, nkmax, piece);
            }
            outs[pass] = try a.alloc(f32, @as(usize, cs.t) * cs.heads * kvl);
            try g.c().download(std.mem.sliceAsBytes(outs[pass]), ol);
        }
        var big: f32 = 0;
        for (outs[0]) |v| big = @max(big, @abs(v));
        var e: f32 = 0;
        for (outs[0], outs[1]) |want, got| e = @max(e, @abs(got - want) / big);
        std.debug.print("hy4 tensor-core attention ({s}, {d} heads, {d} rows): worst error {e} of the largest |out| {e}\n", .{ cs.name, cs.heads, cs.t, e, big });
        try std.testing.expect(e <= 2e-2);
    }
}

test "Hy4 CUDA forward: the Engram residual matches engram_ref (HY4_FWD_FATBIN)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = (try open(a)) orelse return error.SkipZigTest;
    defer g.deinit();
    var rng = std.Random.DefaultPrng.init(12);
    const r = rng.random();
    const H = 64;
    const n = 4;
    const t = 9;
    const order = 3;
    const hpo = 2;
    const hd = 8;
    const heads = (order - 1) * hpo;
    const E = heads * hd;
    const vocab = 50;
    const map = try a.alloc(i32, vocab);
    for (map, 0..) |*m, i| m.* = @intCast(i % 37);
    const sizes = [_]u64{ 11, 13, 17, 19 };
    var rows_total: u64 = 0;
    for (sizes) |s| rows_total += s;
    const table = try rand(a, r, rows_total * hd, 1.0);
    const mult = [_]i64{ 7, 11, 13 };
    const key_w = try rand(a, r, n * H * E, 0.2);
    const key_b = try rand(a, r, n * H, 0.1);
    const value_w = try rand(a, r, H * E, 0.2);
    const value_b = try rand(a, r, H, 0.1);
    const e: engram_ref.Engram = .{
        .hidden = H,
        .hc = n,
        .order = order,
        .heads_per_order = hpo,
        .head_dim = hd,
        .pad = 3,
        .fp8 = false,
        .map = map,
        .mult = &mult,
        .sizes = &sizes,
        .table = table,
        .head_norms = try around(a, r, heads * hd, 1, 0.2),
        .key_w = key_w,
        .key_b = key_b,
        .norm_key = try around(a, r, n * H, 1, 0.2),
        .norm_query = try around(a, r, n * H, 1, 0.2),
        .value_w = value_w,
        .value_b = value_b,
        .pre_gamma = try around(a, r, H, 1, 0.2),
        .scales = try rand(a, r, n * H, 0.5),
    };
    const ids = [t]u32{ 5, 9, 22, 41, 3, 17, 17, 30, 8 };
    const time = [t]i32{ 0, 1, 2, 3, 0, 1, 2, 3, 4 };
    var mask = [t]bool{ true, true, true, false, true, true, true, true, true };
    const wins = try a.alloc(u32, t * order);
    engram_ref.windows(&ids, &time, order, e.pad, wins);
    // device copies first (bf16 weights round the host values the reference then uses)
    var ge: gm.Engram = .{
        .heads = heads,
        .head_dim = hd,
        .head_norms = try g.put(@constCast(e.head_norms), .bf16),
        .value = .{ .plain = .{ .w = try g.put(value_w, .bf16), .b = try g.put(value_b, .bf16), .k = E, .n = H } },
        .keys = undefined,
        .norm_key = try g.put(@constCast(e.norm_key), .bf16),
        .norm_query = try g.put(@constCast(e.norm_query), .bf16),
        .pre_gamma = try g.put(@constCast(e.pre_gamma), .bf16),
        .scales = try g.put(@constCast(e.scales), .bf16),
    };
    const keys = try a.alloc(gm.Lin, n);
    for (keys, 0..) |*kl, s| kl.* = .{ .plain = .{ .w = try g.put(key_w[s * H * E ..][0 .. H * E], .bf16), .b = try g.put(key_b[s * H ..][0..H], .bf16), .k = E, .n = H } };
    ge.keys = keys;
    const streams = try rand(a, r, t * n * H, 1.5);
    const res = try a.alloc(f32, streams.len);
    try engram_ref.residual(a, &e, wins, &mask, streams, res);
    const want = try a.dupe(f32, streams);
    for (want, res) |*x, y| x.* += y;
    // the rows the caller fetches: hashed on the host, table rows as they come
    const emb = try a.alloc(f32, t * E);
    const at = try a.alloc(usize, heads);
    for (0..t) |i| {
        engram_ref.rows(&e, wins[i * order ..][0..order], at);
        for (0..heads) |h| @memcpy(emb[i * E + h * hd ..][0..hd], table[at[h] * hd ..][0..hd]);
    }
    var mask_u8: [t]u8 = undefined;
    for (mask, &mask_u8) |m, *o| o.* = @intFromBool(m);
    const lim: gm.Limits = .{ .chunk = t, .ctx = 16, .hidden = H, .hc = n, .heads = 1, .qk = 1, .v = 1, .kv_lora = 1, .rope = 1, .q_lora = 1, .engram_width = E };
    var w = try gm.Work.init(a, g.c(), lim);
    defer w.deinit();
    try g.c().upload(w.streams, std.mem.sliceAsBytes(streams));
    try g.c().upload(w.emb, std.mem.sliceAsBytes(emb));
    try g.c().upload(w.mask, &mask_u8);
    try gm.engramStep(&w, &ge, w.emb, H, n, t);
    const got = try a.alloc(f32, streams.len);
    try g.c().download(std.mem.sliceAsBytes(got), w.streams);
    const err = worst(got, want);
    std.debug.print("hy4 forward Engram: worst relative error {e}\n", .{err});
    try std.testing.expect(err <= 1e-3);
    _ = &mask;
}

test "Hy4 CUDA forward: the vision tower matches vision_ref (HY4_FWD_FATBIN)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = (try open(a)) orelse return error.SkipZigTest;
    defer g.deinit();
    var rng = std.Random.DefaultPrng.init(13);
    const r = rng.random();
    const d: vision_ref.Dims = .{ .hidden = 32, .inter = 48, .heads = 4, .patch = 2, .out = 40, .edge = 6 };
    const h: u32 = 32;
    const pp: u32 = 3 * 2 * 2;
    const wide: u32 = h * 4;
    const grid: image_grid.Grid = .{ .h = 4, .w = 6 };
    const P = grid.patches();
    const blocks = try a.alloc(vision_ref.Block, 2);
    const gblocks = try a.alloc(gv.Block, 2);
    for (blocks, gblocks) |*b, *gb| {
        b.* = .{
            .norm1_w = try around(a, r, h, 1, 0.2),
            .norm1_b = try rand(a, r, h, 0.1),
            .q_w = try rand(a, r, h * h, 0.2),
            .q_b = try rand(a, r, h, 0.1),
            .k_w = try rand(a, r, h * h, 0.2),
            .k_b = try rand(a, r, h, 0.1),
            .v_w = try rand(a, r, h * h, 0.2),
            .v_b = try rand(a, r, h, 0.1),
            .o_w = try rand(a, r, h * h, 0.2),
            .o_b = try rand(a, r, h, 0.1),
            .norm2_w = try around(a, r, h, 1, 0.2),
            .norm2_b = try rand(a, r, h, 0.1),
            .fc1_w = try rand(a, r, d.inter * h, 0.2),
            .fc1_b = try rand(a, r, d.inter, 0.1),
            .fc2_w = try rand(a, r, h * d.inter, 0.2),
            .fc2_b = try rand(a, r, h, 0.1),
        };
        const P2 = struct {
            fn f(gg: *Gpu, w: []const f32, bb: []const f32, k: u32, n: u32) !gv.Proj {
                return .{ .w = try gg.put(@constCast(w), .bf16), .b = try gg.put(@constCast(bb), .bf16), .k = k, .n = n };
            }
        }.f;
        gb.* = .{
            .norm1_w = try g.put(@constCast(b.norm1_w), .bf16),
            .norm1_b = try g.put(@constCast(b.norm1_b), .bf16),
            .q = try P2(g, b.q_w, b.q_b, h, h),
            .k = try P2(g, b.k_w, b.k_b, h, h),
            .v = try P2(g, b.v_w, b.v_b, h, h),
            .o = try P2(g, b.o_w, b.o_b, h, h),
            .norm2_w = try g.put(@constCast(b.norm2_w), .bf16),
            .norm2_b = try g.put(@constCast(b.norm2_b), .bf16),
            .fc1 = try P2(g, b.fc1_w, b.fc1_b, h, d.inter),
            .fc2 = try P2(g, b.fc2_w, b.fc2_b, d.inter, h),
        };
    }
    const vw: vision_ref.Weights = .{
        .patch_w = try rand(a, r, h * pp, 0.3),
        .patch_b = try rand(a, r, h, 0.1),
        .positions = try rand(a, r, (d.edge * d.edge + 1) * h, 0.3),
        .blocks = blocks,
        .merge_norm = try around(a, r, h, 1, 0.2),
        .conv_w = try rand(a, r, wide * h * 4, 0.1),
        .conv_b = try rand(a, r, wide, 0.1),
        .out_w = try rand(a, r, d.out * wide, 0.1),
        .out_b = try rand(a, r, d.out, 0.1),
        .newline = try rand(a, r, d.out, 1.0),
    };
    const tower: gv.Tower = .{
        .hidden = h,
        .inter = @intCast(d.inter),
        .heads = @intCast(d.heads),
        .patch = 2,
        .out = @intCast(d.out),
        .edge = d.edge,
        .patch_proj = .{ .w = try g.put(@constCast(vw.patch_w), .bf16), .b = try g.put(@constCast(vw.patch_b), .bf16), .k = pp, .n = h },
        .positions = vw.positions[h..],
        .blocks = gblocks,
        .merge_norm = try g.put(@constCast(vw.merge_norm), .bf16),
        .conv = .{ .w = try g.put(@constCast(vw.conv_w), .bf16), .b = try g.put(@constCast(vw.conv_b), .bf16), .k = wide, .n = wide },
        .out_proj = .{ .w = try g.put(@constCast(vw.out_w), .bf16), .b = try g.put(@constCast(vw.out_b), .bf16), .k = wide, .n = @intCast(d.out) },
        .newline = try g.put(@constCast(vw.newline), .bf16),
    };
    const pixels = try rand(a, r, P * pp, 1.0);
    const want = try vision_ref.forward(a, &d, &vw, pixels, &.{grid});
    var w = try gv.Work.init(&g.d, &tower, 64);
    defer w.deinit();
    const tokens = grid.tokens(2);
    const out = try g.alloc(@as(u64, tokens) * d.out * 4);
    try gv.forward(a, g.c(), &tower, &w, pixels, grid, out);
    const got = try a.alloc(f32, want.len);
    try g.c().download(std.mem.sliceAsBytes(got), out);
    const err = worst(got, want);
    std.debug.print("hy4 forward vision: {d} tokens, worst relative error {e}\n", .{ tokens, err });
    try std.testing.expect(err <= 2e-3);
}
