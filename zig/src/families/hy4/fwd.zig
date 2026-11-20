//! The Hy4 forward kernels (zig/kernels/cuda/hy4/forward.cu and hc.cu, one image) behind typed calls: fp32 activations,
//! plain weights read as stored (f32, bf16, f16). The EXL3 projections go through cuda.exl3.Dense (see gpu_model.zig).

const std = @import("std");
const cuda = @import("cuda");

pub const Dt = enum(i32) { f32 = 0, bf16 = 1, f16 = 2 };

/// How a latent cache stores its rows (forward.cu CDT_*): f32; bf16; e4m3 latents with a per-row f32 scale (the
/// rotary halves bf16 in both narrow kinds).
pub const CacheDt = enum(i32) {
    f32 = 0,
    bf16 = 1,
    fp8 = 3,

    /// Bytes of one latent element.
    pub fn latBytes(c: CacheDt) u64 {
        return switch (c) {
            .f32 => 4,
            .bf16 => 2,
            .fp8 => 1,
        };
    }
    /// Bytes of one rotary (and group-key) element.
    pub fn rotBytes(c: CacheDt) u64 {
        return if (c == .f32) 4 else 2;
    }
    /// The element type the group keys are stored as (f32 or bf16).
    pub fn keyDt(c: CacheDt) CacheDt {
        return if (c == .f32) .f32 else .bf16;
    }
    pub fn parse(s: []const u8) ?CacheDt {
        if (std.mem.eql(u8, s, "f32")) return .f32;
        if (std.mem.eql(u8, s, "bf16")) return .bf16;
        if (std.mem.eql(u8, s, "fp8")) return .fp8;
        return null;
    }
};

/// A plain weight or vector on the device: its address and how its elements are stored.
pub const W = struct {
    ptr: u64,
    dt: Dt,

    pub const none: W = .{ .ptr = 0, .dt = .f32 };
};

const Fns = struct {
    gemm: cuda.Function,
    gemv: cuda.Function,
    rmsnorm: cuda.Function,
    layernorm: cuda.Function,
    swiglu: cuda.Function,
    pointwise: cuda.Function,
    copy2d: cuda.Function,
    embed: cuda.Function,
    expand: cuda.Function,
    hc_head: cuda.Function,
    rope: cuda.Function,
    absorb_q: cuda.Function,
    mla_attn: cuda.Function,
    uv_gate: cuda.Function,
    idx_pool: cuda.Function,
    idx_score: cuda.Function,
    idx_topk: cuda.Function,
    route: cuda.Function,
    gather: cuda.Function,
    scatter: cuda.Function,
    combine: cuda.Function,
    engram_gate: cuda.Function,
    vit_attn: cuda.Function,
    merge_gather: cuda.Function,
    newlines: cuda.Function,
    hc_pre: cuda.Function,
    hc_post: cuda.Function,
    cache_store: cuda.Function,
    store_rows: cuda.Function,
    to_half: cuda.Function,
    gemm_tc_bf16: cuda.Function,
    gemm_tc_f16: cuda.Function,
    mla_scores_tc: cuda.Function,
    mla_softmax: cuda.Function,
    mla_pv_tc: cuda.Function,
};

fn blocks(n: u64, per: u64) u32 {
    return @intCast((n + per - 1) / per);
}

pub const Kernels = struct {
    mod: cuda.Module,
    f: Fns,
    /// Plain bf16 / f16 projections of at least this many rows run on tensor cores (x rounded to the weight's type,
    /// fp32 sums); 0: never. HY4_TC_GEMM_ROWS, default 32.
    tc_min: u32 = 32,
    /// Attention calls of at least this many query rows take the tensor-core path (mlaAttnTc) when the caller has
    /// scratch for it; 0: never. HY4_ATTN_TC_ROWS, default 16 (prompt chunks; decode keeps the scalar kernel).
    attn_tc_min: u32 = 16,

    /// The image (a fatbin or a cubin) loaded on the current context.
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var k: Kernels = .{ .mod = try cuda.Module.load(d, image), .f = undefined };
        errdefer k.mod.unload();
        if (std.c.getenv("HY4_TC_GEMM_ROWS")) |v| k.tc_min = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 32;
        if (std.c.getenv("HY4_ATTN_TC_ROWS")) |v| k.attn_tc_min = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 16;
        inline for (comptime std.meta.fieldNames(Fns)) |name| @field(k.f, name) = try k.mod.function("hy4_" ++ name);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]u32, block: u32, shared: u32, args: anytype) !void {
        var a: cuda.launch.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block, .y = 1, .z = 1 }, .shared = shared }, s, &a);
    }

    /// y[m, n] (+)= x[m, k] . w[n, k]^T (+ b): rows `ldx` / `ldy` floats apart.
    pub fn gemm(k: *const Kernels, s: cuda.Stream, x: u64, ldx: u64, w: W, b: W, y: u64, ldy: u64, m: u32, n: u32, kk: u32, accumulate: bool) !void {
        const args = .{ x, @as(i64, @intCast(ldx)), w.ptr, @intFromEnum(w.dt), b.ptr, @intFromEnum(b.dt), y, @as(i64, @intCast(ldy)), @as(i32, @intCast(m)), @as(i32, @intCast(n)), @as(i32, @intCast(kk)), @as(i32, @intFromBool(accumulate)) };
        if (m <= 8) return go(k.f.gemv, s, .{ blocks(n, 8), 1, 1 }, 256, 0, args);
        if (k.tc_min > 0 and m >= k.tc_min and w.dt != .f32) {
            const f = if (w.dt == .bf16) k.f.gemm_tc_bf16 else k.f.gemm_tc_f16;
            return go(f, s, .{ blocks(n, 64), blocks(m, 64), 1 }, 128, 0, args);
        }
        return go(k.f.gemm, s, .{ blocks(n, 64), blocks(m, 64), 1 }, 256, 0, args);
    }

    /// One block a row: y = x * rsqrt(mean(x^2) + eps) * w[(row % wmod) * width + i] (w.ptr 0: unweighted).
    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, ldx: u64, w: W, wmod: u32, y: u64, ldy: u64, rows: u32, width: u32, eps: f32) !void {
        if (rows == 0) return;
        try go(k.f.rmsnorm, s, .{ rows, 1, 1 }, 256, 0, .{ x, @as(i64, @intCast(ldx)), w.ptr, @intFromEnum(w.dt), @as(i32, @intCast(wmod)), y, @as(i64, @intCast(ldy)), @as(i32, @intCast(width)), eps });
    }

    pub fn layernorm(k: *const Kernels, s: cuda.Stream, x: u64, w: W, b: W, y: u64, rows: u32, width: u32, eps: f32) !void {
        try go(k.f.layernorm, s, .{ rows, 1, 1 }, 256, 0, .{ x, w.ptr, b.ptr, @intFromEnum(w.dt), y, @as(i32, @intCast(width)), eps });
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, ld: u64, out: u64, rows: u32, width: u32, clamp: f32) !void {
        const n = @as(u64, rows) * width;
        if (n == 0) return;
        try go(k.f.swiglu, s, .{ blocks(n, 256), 1, 1 }, 256, 0, .{ g, u, @as(i64, @intCast(ld)), out, @as(i32, @intCast(rows)), @as(i32, @intCast(width)), clamp });
    }

    pub fn gelu(k: *const Kernels, s: cuda.Stream, x: u64, count: u64) !void {
        try go(k.f.pointwise, s, .{ blocks(count, 256), 1, 1 }, 256, 0, .{ x, @as(u64, 0), @as(i64, @intCast(count)), @as(i32, 1) });
    }

    /// x += y over `count` floats.
    pub fn add(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, count: u64) !void {
        try go(k.f.pointwise, s, .{ blocks(count, 256), 1, 1 }, 256, 0, .{ x, y, @as(i64, @intCast(count)), @as(i32, 0) });
    }

    pub fn copy2d(k: *const Kernels, s: cuda.Stream, src: u64, lds: u64, dst: u64, ldd: u64, rows: u32, width: u32) !void {
        const n = @as(u64, rows) * width;
        if (n == 0) return;
        try go(k.f.copy2d, s, .{ blocks(n, 256), 1, 1 }, 256, 0, .{ src, @as(i64, @intCast(lds)), dst, @as(i64, @intCast(ldd)), @as(i32, @intCast(rows)), @as(i32, @intCast(width)) });
    }

    pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, table: u64, x: u64, t: u32, hidden: u32) !void {
        try go(k.f.embed, s, .{ t, 1, 1 }, 256, 0, .{ ids, table, x, @as(i32, @intCast(hidden)) });
    }

    pub fn expand(k: *const Kernels, s: cuda.Stream, x: u64, streams: u64, t: u32, n: u32, hidden: u32) !void {
        try go(k.f.expand, s, .{ t, 1, 1 }, 256, 0, .{ x, streams, @as(i32, @intCast(n)), @as(i32, @intCast(hidden)) });
    }

    pub const Hc = struct { fn_: u64, base: u64, scale: u64 };

    pub fn hcPre(k: *const Kernels, s: cuda.Stream, streams: u64, h: Hc, mixed: u64, post: u64, t: u32, n: u32, hidden: u32, eps: f32, hc_eps: f32, magnitude: f32) !void {
        try go(k.f.hc_pre, s, .{ t, 1, 1 }, 256, 0, .{ streams, h.fn_, h.base, h.scale, mixed, post, @as(i32, @intCast(n)), @as(i32, @intCast(hidden)), eps, hc_eps, magnitude });
    }

    pub fn hcPost(k: *const Kernels, s: cuda.Stream, streams: u64, post: u64, y: u64, t: u32, n: u32, hidden: u32) !void {
        try go(k.f.hc_post, s, .{ t, 1, 1 }, 256, 0, .{ streams, post, y, @as(i32, @intCast(n)), @as(i32, @intCast(hidden)) });
    }

    pub fn hcHead(k: *const Kernels, s: cuda.Stream, streams: u64, h: Hc, out: u64, t: u32, n: u32, hidden: u32, eps: f32, hc_eps: f32) !void {
        try go(k.f.hc_head, s, .{ t, 1, 1 }, 256, 0, .{ streams, h.fn_, h.base, h.scale, out, @as(i32, @intCast(n)), @as(i32, @intCast(hidden)), eps, hc_eps });
    }

    pub const Rope = struct { pos: u64, total: u64, theta: f64, sections: [3]u32 };

    /// Rotates `rope` channels at `offset` of `heads` heads (`head_stride` apart) in `rows` rows `ld` apart; row r at
    /// position slot pos0 + r * step.
    pub fn rope(k: *const Kernels, s: cuda.Stream, x: u64, ld: u64, rows: u32, heads: u32, head_stride: u32, offset: u32, rope_dim: u32, r: Rope, pos0: u64, step: u32) !void {
        if (rows == 0) return;
        try go(k.f.rope, s, .{ rows, 1, 1 }, 128, 0, .{ x, @as(i64, @intCast(ld)), @as(i32, @intCast(heads)), @as(i32, @intCast(head_stride)), @as(i32, @intCast(offset)), @as(i32, @intCast(rope_dim)), r.pos, @as(i64, @intCast(r.total)), @as(i64, @intCast(pos0)), @as(i32, @intCast(step)), r.theta, @as(i32, @intCast(r.sections[1])), @as(i32, @intCast(r.sections[2])) });
    }

    pub fn absorbQ(k: *const Kernels, s: cuda.Stream, q: u64, kvb: W, ql: u64, t: u32, heads: u32, qk: u32, nope: u32, vdim: u32, kvl: u32) !void {
        try go(k.f.absorb_q, s, .{ t * heads, 1, 1 }, 256, nope * 4, .{ q, kvb.ptr, @intFromEnum(kvb.dt), ql, @as(i32, @intCast(heads)), @as(i32, @intCast(qk)), @as(i32, @intCast(nope)), @as(i32, @intCast(vdim)), @as(i32, @intCast(kvl)) });
    }

    /// The indexer's picks for hy4_mla_attn: per query row a list of kept groups (`ld` ints apart) and their count.
    pub const Keep = struct { list: u64 = 0, ld: u64 = 0, count: u64 = 0, kpool: u32 = 0 };

    /// A latent cache as the attention reads it: `dt` latents (fp8: times `scale[slot]`), slot = key % slots.
    pub const CacheView = struct { lat: u64, rot: u64, scale: u64 = 0, dt: CacheDt = .f32, slots: u64 };

    pub fn mlaAttn(k: *const Kernels, s: cuda.Stream, ql: u64, q: u64, qk: u32, nope: u32, cv: CacheView, sinks: u64, ol: u64, t: u32, heads: u32, kvl: u32, rope_dim: u32, qpos0: u64, window: u32, keep: Keep, scale: f32) !void {
        if (kvl + rope_dim > 640 or kvl > 1024) return error.Hy4Shape;
        if (t == 0) return;
        try go(k.f.mla_attn, s, .{ t, heads, 1 }, 256, 0, .{ ql, q, @as(i32, @intCast(qk)), @as(i32, @intCast(nope)), cv.lat, cv.rot, cv.scale, @intFromEnum(cv.dt), @as(i64, @intCast(cv.slots)), sinks, ol, @as(i32, @intCast(heads)), @as(i32, @intCast(kvl)), @as(i32, @intCast(rope_dim)), @as(i64, @intCast(qpos0)), @as(i32, @intCast(window)), keep.list, @as(i64, @intCast(keep.ld)), keep.count, @as(i32, @intCast(if (keep.kpool == 0) 1 else keep.kpool)), scale });
    }

    /// Scratch of mlaAttnTc: scores (f32) and weights (bf16) for `rows` query rows of `heads` heads and `keys` keys.
    pub fn attnTcBytes(rows: u64, heads: u64, keys: u64) u64 {
        return rows * heads * keys * 6;
    }

    /// mlaAttn on tensor cores for `t` query rows: scores, softmax with the sink, weights times latents, in pieces of
    /// at most `piece` rows. `keys` (>= every row's key count) is the row stride of the scratch `S` (f32) / `P`
    /// (bf16), each `piece * heads * keys` long. Q and K are rounded to bf16 (the caches are bf16 or fp8 by default).
    pub fn mlaAttnTc(k: *const Kernels, s: cuda.Stream, ql: u64, q: u64, qk: u32, nope: u32, cv: CacheView, sinks: u64, ol: u64, t: u32, heads: u32, kvl: u32, rope_dim: u32, qpos0: u64, window: u32, keep: Keep, scale: f32, S: u64, P: u64, keys: u64, piece: u32) !void {
        if (t == 0) return;
        if (kvl % 64 != 0 or (kvl + rope_dim) % 32 != 0 or piece == 0) return error.Hy4Shape;
        const kp: i32 = @intCast(if (keep.kpool == 0) 1 else keep.kpool);
        var r: u32 = 0;
        while (r < t) : (r += piece) {
            const n = @min(piece, t - r);
            const ru: u64 = r;
            const list = if (keep.list != 0) keep.list + 4 * ru * keep.ld else 0;
            const count = if (keep.count != 0) keep.count + 4 * ru else 0;
            const p0: i64 = @intCast(qpos0 + ru);
            const hz = blocks(heads, 64);
            try go(k.f.mla_scores_tc, s, .{ n, blocks(keys, 64), hz }, 128, 0, .{ ql + 4 * ru * heads * kvl, q + 4 * ru * heads * qk, @as(i32, @intCast(qk)), @as(i32, @intCast(nope)), cv.lat, cv.rot, cv.scale, @intFromEnum(cv.dt), @as(i64, @intCast(cv.slots)), S, @as(i64, @intCast(keys)), @as(i32, @intCast(heads)), @as(i32, @intCast(kvl)), @as(i32, @intCast(rope_dim)), p0, @as(i32, @intCast(window)), list, @as(i64, @intCast(keep.ld)), count, kp, scale });
            try go(k.f.mla_softmax, s, .{ n, heads, 1 }, 256, 0, .{ S, P, @as(i64, @intCast(keys)), sinks, @as(i32, @intCast(heads)), p0, @as(i32, @intCast(window)), count, @as(i32, @intFromBool(list != 0)), kp });
            try go(k.f.mla_pv_tc, s, .{ n, blocks(kvl, 64), hz }, 128, 0, .{ P, @as(i64, @intCast(keys)), cv.lat, cv.scale, @intFromEnum(cv.dt), @as(i64, @intCast(cv.slots)), ol + 4 * ru * heads * kvl, @as(i32, @intCast(heads)), @as(i32, @intCast(kvl)), p0, @as(i32, @intCast(window)), list, @as(i64, @intCast(keep.ld)), count, kp });
        }
    }

    /// Rows [p0, p0 + rows) of `src` (latent then rotary key, `lds` floats apart) into the cache's slots.
    pub fn cacheStore(k: *const Kernels, s: cuda.Stream, src: u64, lds: u64, kvl: u32, rope_dim: u32, cv: CacheView, p0: u64, rows: u32) !void {
        if (rows == 0) return;
        if (cv.dt == .fp8 and cv.scale == 0) return error.Hy4Shape;
        try go(k.f.cache_store, s, .{ rows, 1, 1 }, 256, 0, .{ src, @as(i64, @intCast(lds)), @as(i32, @intCast(kvl)), @as(i32, @intCast(rope_dim)), cv.lat, cv.rot, cv.scale, @intFromEnum(cv.dt), @as(i64, @intCast(cv.slots)), @as(i64, @intCast(p0)) });
    }

    /// dst rows [row0, row0 + rows) = src rows as `dt` (f32 or bf16).
    pub fn storeRows(k: *const Kernels, s: cuda.Stream, src: u64, lds: u64, dst: u64, dt: CacheDt, width: u32, row0: u64, rows: u32) !void {
        if (rows == 0) return;
        if (dt == .fp8) return error.Hy4Shape;
        try go(k.f.store_rows, s, .{ rows, 1, 1 }, 128, 0, .{ src, @as(i64, @intCast(lds)), dst, @intFromEnum(dt), @as(i32, @intCast(width)), @as(i64, @intCast(row0)) });
    }

    pub fn toHalf(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, count: u64) !void {
        if (count == 0) return;
        try go(k.f.to_half, s, .{ blocks(count, 256), 1, 1 }, 256, 0, .{ x, y, @as(i64, @intCast(count)) });
    }

    pub fn uvGate(k: *const Kernels, s: cuda.Stream, ol: u64, kvb: W, gate: u64, out: u64, t: u32, heads: u32, nope: u32, vdim: u32, kvl: u32) !void {
        try go(k.f.uv_gate, s, .{ blocks(vdim, 8), heads, blocks(t, 8) }, 256, 0, .{ ol, kvb.ptr, @intFromEnum(kvb.dt), gate, out, @as(i32, @intCast(t)), @as(i32, @intCast(heads)), @as(i32, @intCast(nope)), @as(i32, @intCast(vdim)), @as(i32, @intCast(kvl)) });
    }

    /// Groups [g0, g0 + groups) pooled from the token-key ring `tok` (`tslots` rows) into rows [0, groups) of `out` (f32).
    pub fn idxPool(k: *const Kernels, s: cuda.Stream, tok: u64, tslots: u64, kv: u64, kg: u64, ape: u64, knorm: u64, out: u64, g0: u64, groups: u32, dim: u32, kpool: u32, eps: f32) !void {
        if (groups == 0) return;
        if (dim != 128 or kpool > 8) return error.Hy4Shape;
        try go(k.f.idx_pool, s, .{ groups, 1, 1 }, 128, 0, .{ tok, @as(i64, @intCast(tslots)), kv, kg, ape, knorm, out, @as(i64, @intCast(g0)), @as(i32, @intCast(dim)), @as(i32, @intCast(kpool)), eps });
    }

    pub fn idxScore(k: *const Kernels, s: cuda.Stream, iq: u64, hw: u64, gk: u64, gdt: CacheDt, logits: u64, ld: u64, t: u32, heads: u32, dim: u32, groups: u32, qpos0: u64, kpool: u32, hscale: f32) !void {
        if (gdt == .fp8) return error.Hy4Shape;
        try go(k.f.idx_score, s, .{ blocks(groups, 8), t, 1 }, 256, 0, .{ iq, hw, gk, @intFromEnum(gdt), logits, @as(i64, @intCast(ld)), @as(i32, @intCast(heads)), @as(i32, @intCast(dim)), @as(i32, @intCast(groups)), @as(i64, @intCast(qpos0)), @as(i32, @intCast(kpool)), hscale });
    }

    /// The keep map, and with `list` non-zero the kept groups ascending (`lld` ints a row) and their count per row.
    pub fn idxTopk(k: *const Kernels, s: cuda.Stream, logits: u64, ld: u64, keep: u64, kld: u64, t: u32, groups: u32, qpos0: u64, kpool: u32, recall: u32, list: u64, lld: u64, count: u64) !void {
        try go(k.f.idx_topk, s, .{ t, 1, 1 }, 256, 0, .{ logits, @as(i64, @intCast(ld)), keep, @as(i64, @intCast(kld)), @as(i32, @intCast(groups)), @as(i64, @intCast(qpos0)), @as(i32, @intCast(kpool)), @as(i32, @intCast(recall)), list, @as(i64, @intCast(lld)), count });
    }

    pub fn route(k: *const Kernels, s: cuda.Stream, logits: u64, bias: u64, t: u32, experts: u32, top: u32, scale: f32, ids: u64, wts: u64) !void {
        if (experts > 1024 or top > 256) return error.Hy4Shape;
        try go(k.f.route, s, .{ t, 1, 1 }, 256, 0, .{ logits, bias, @as(i32, @intCast(experts)), @as(i32, @intCast(top)), scale, ids, wts });
    }

    pub fn gather(k: *const Kernels, s: cuda.Stream, src: u64, idx: u64, dst: u64, rows: u32, width: u32) !void {
        if (rows == 0) return;
        try go(k.f.gather, s, .{ rows, 1, 1 }, 256, 0, .{ src, idx, dst, @as(i32, @intCast(width)) });
    }

    pub fn scatter(k: *const Kernels, s: cuda.Stream, src: u64, idx: u64, dst: u64, rows: u32, width: u32) !void {
        if (rows == 0) return;
        try go(k.f.scatter, s, .{ rows, 1, 1 }, 256, 0, .{ src, idx, dst, @as(i32, @intCast(width)) });
    }

    pub fn combine(k: *const Kernels, s: cuda.Stream, slots: u64, wts: u64, acc: u64, t: u32, top: u32, width: u32) !void {
        try go(k.f.combine, s, .{ t, 1, 1 }, 256, 0, .{ slots, wts, acc, @as(i32, @intCast(top)), @as(i32, @intCast(width)) });
    }

    pub fn engramGate(k: *const Kernels, s: cuda.Stream, streams: u64, nk: u64, main: u64, nq_w: W, scales: W, mask: u64, t: u32, n: u32, hidden: u32, eps: f32) !void {
        try go(k.f.engram_gate, s, .{ t, n, 1 }, 256, 0, .{ streams, nk, main, nq_w.ptr, scales.ptr, @intFromEnum(nq_w.dt), mask, @as(i32, @intCast(n)), @as(i32, @intCast(hidden)), eps });
    }

    pub fn vitAttn(k: *const Kernels, s: cuda.Stream, q: u64, kk: u64, v: u64, out: u64, ld: u64, len: u32, heads: u32, hd: u32, scale: f32) !void {
        if (hd > 256) return error.Hy4Shape;
        try go(k.f.vit_attn, s, .{ len, heads, 1 }, 128, 0, .{ q, kk, v, out, @as(i64, @intCast(ld)), @as(i32, @intCast(len)), @as(i32, @intCast(hd)), scale });
    }

    pub fn mergeGather(k: *const Kernels, s: cuda.Stream, x: u64, cell: u64, gh2: u32, gw: u32, gw2: u32, hidden: u32) !void {
        try go(k.f.merge_gather, s, .{ gh2 * gw2, 1, 1 }, 256, 0, .{ x, cell, @as(i32, @intCast(gw)), @as(i32, @intCast(gw2)), @as(i32, @intCast(hidden)) });
    }

    pub fn newlines(k: *const Kernels, s: cuda.Stream, cells: u64, newline: W, out: u64, gh2: u32, gw2: u32, width: u32) !void {
        try go(k.f.newlines, s, .{ gh2 * (gw2 + 1), 1, 1 }, 256, 0, .{ cells, newline.ptr, @intFromEnum(newline.dt), out, @as(i32, @intCast(gw2)), @as(i32, @intCast(width)) });
    }
};
