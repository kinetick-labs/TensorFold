//! The Hy4-flash decoder on CUDA: fp32 activations through the forward kernels (fwd.zig), EXL3 projections through the
//! existing trellis kernels (cuda.exl3.Dense), plain weights read as stored. One request at a time: an absorbed MLA
//! latent cache per layer (normed 512-wide latent plus the rotated 64-wide key, f32), the compressed indexer's token
//! and group keys, the latent MoE grouped on the host per chunk (route on the device, the top-k ids downloaded,
//! each used expert run over its rows, a deterministic combine), Engram rows fetched by the caller. Layer math
//! follows layer_ref.zig / engram_ref.zig, which the GPU tests compare against.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const fwd = @import("fwd.zig");
const layer_ref = @import("layer_ref.zig");

pub const W = fwd.W;
pub const Dims = layer_ref.Dims;
pub const f32_eps: f32 = std.math.floatEps(f32);

/// A projection `y[rows, n] = x[rows, k] . W^T (+ b)`.
pub const Lin = union(enum) {
    plain: Plain,
    exl3: cuda.exl3.Weight,

    pub const Plain = struct { w: W, b: W = .none, k: u32, n: u32 };

    pub fn n(l: Lin) u32 {
        return switch (l) {
            .plain => |p| p.n,
            .exl3 => |e| e.n,
        };
    }
    pub fn k(l: Lin) u32 {
        return switch (l) {
            .plain => |p| p.k,
            .exl3 => |e| e.k,
        };
    }
};

pub const Ctx = struct {
    d: *const cuda.Driver,
    k: *const fwd.Kernels,
    dense: ?*cuda.exl3.Dense,
    s: cuda.Stream,

    pub fn lin(c: Ctx, l: Lin, x: u64, ldx: u64, rows: u32, y: u64, ldy: u64) !void {
        if (rows == 0) return;
        switch (l) {
            .plain => |p| try c.k.gemm(c.s, x, ldx, p.w, p.b, y, ldy, rows, p.n, p.k, false),
            .exl3 => |w| {
                if (ldx != w.k or ldy != w.n) return error.Hy4Stride;
                const de = c.dense orelse return error.Hy4NoExl3;
                try de.decode(c.s, w, x, .f32, rows, y, .f32);
            },
        }
    }

    pub fn upload(c: Ctx, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try c.d.check(c.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, c.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn download(c: Ctx, out: []u8, src: u64) !void {
        if (out.len == 0) return;
        try c.d.check(c.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, c.s.handle), "cuMemcpyDtoHAsync");
        try c.s.synchronize();
    }
};

pub const Index = struct { query: Lin, head_weight: Lin, token_key: Lin, token_key_norm: W, key_value: u64, key_gate: u64, key_norm: u64, ape: u64 };

pub const Attn = struct {
    q_a: Lin,
    q_a_norm: W,
    q_b: Lin,
    kv_a: Lin,
    kv_a_norm: W,
    kv_b: W, // [heads * (nope + v), kv_lora]
    gate: Lin,
    o: Lin,
    post_norm: W,
    sinks: u64, // f32 [heads]
    index: ?Index = null,
};

pub const Moe = struct {
    experts: u32,
    top: u32,
    latent: u32,
    width: u32,
    scale: f32,
    clamp: f32,
    router: Lin,
    bias: u64, // f32 [experts]
    pre: Lin, // hidden -> latent
    post: Lin, // latent -> hidden
    expert: []const [3]Lin, // gate, up (latent -> width), down (width -> latent)
    shared: [3]Lin,
    norm: W,
};

pub const Dense = struct { gate: Lin, up: Lin, down: Lin, norm: W };

pub const Engram = struct {
    heads: u32, // hash heads
    head_dim: u32,
    head_norms: W, // [heads, head_dim] contiguous
    value: Lin, // width -> hidden, biased
    keys: []const Lin, // per stream, width -> hidden, biased
    norm_key: W, // [hc, hidden] contiguous
    norm_query: W, // [hc, hidden], same dtype as scales
    pre_gamma: W,
    scales: W, // [hc, hidden]

    pub fn width(e: *const Engram) u32 {
        return e.heads * e.head_dim;
    }
};

pub const Layer = struct {
    dims: Dims,
    attn_hc: fwd.Kernels.Hc,
    ffn_hc: fwd.Kernels.Hc,
    input_norm: W,
    post_attention_norm: W,
    attn: Attn,
    mlp: union(enum) { dense: Dense, moe: Moe },
    engram: ?Engram = null,
};

/// Device sizes the work buffers are cut for.
pub const Limits = struct {
    chunk: u32, // rows a forward call takes at most
    ctx: u32, // cache slots
    hidden: u32,
    hc: u32,
    heads: u32,
    qk: u32,
    v: u32,
    kv_lora: u32,
    rope: u32,
    q_lora: u32,
    inter: u32 = 0, // dense MLP width
    experts: u32 = 0,
    top: u32 = 0,
    latent: u32 = 0,
    width: u32 = 0, // expert and shared width
    index_heads: u32 = 0,
    index_dim: u32 = 0,
    kpool: u32 = 4,
    engram_width: u32 = 0,
    vocab: u32 = 0,
};

pub const Cache = struct { lat: u64, rot: u64, tok: u64 = 0, gk: u64 = 0 };

/// Bump allocation inside one device buffer, 256-byte aligned.
const Bump = struct {
    at: u64 = 0,
    fn take(b: *Bump, bytes: u64) u64 {
        const o = b.at;
        b.at = std.mem.alignForward(u64, b.at + @max(bytes, 1), 256);
        return o;
    }
};

pub const Work = struct {
    c: Ctx,
    lim: Limits,
    buf: cuda.DeviceBuffer,
    host_ids: []i32,
    host_idx: []i32,
    a: Allocator,
    // [chunk, ...] f32 unless noted
    streams: u64,
    mixed: u64,
    x: u64,
    y: u64,
    post: u64,
    qa: u64,
    qres: u64,
    q: u64,
    kva: u64,
    gate: u64,
    ql: u64,
    ol: u64,
    att: u64,
    tk: u64,
    iq: u64,
    hw: u64,
    logits: u64, // [chunk, ctx / kpool]
    keep: u64, // u8 [chunk, ctx / kpool]
    rlog: u64,
    ids: u64, // i32 [chunk, top]
    wts: u64,
    lat: u64,
    acc: u64,
    slots: u64, // [chunk * top, latent]
    xg: u64,
    g: u64,
    u: u64,
    h: u64,
    eo: u64,
    idx: u64, // i32 [2, chunk * top]
    sum: u64,
    emb: u64, // [chunk, engram_width]
    main: u64,
    keys: u64, // [chunk, hc, hidden]
    mask: u64, // u8 [chunk]
    pos: u64, // i32 [3, ctx]
    tokens: u64, // i32 [chunk]
    out: u64, // [chunk, hidden]
    head: u64, // [vocab]

    pub fn init(a: Allocator, c: Ctx, lim: Limits) !Work {
        const C: u64 = lim.chunk;
        const f = 4;
        const groups: u64 = lim.ctx / @max(lim.kpool, 1);
        var b: Bump = .{};
        var w: Work = undefined;
        w.c = c;
        w.lim = lim;
        w.a = a;
        w.streams = b.take(C * lim.hc * lim.hidden * f);
        w.mixed = b.take(C * lim.hidden * f);
        w.x = b.take(C * lim.hidden * f);
        w.y = b.take(C * lim.hidden * f);
        w.post = b.take(C * lim.hc * f);
        w.qa = b.take(C * lim.q_lora * f);
        w.qres = b.take(C * lim.q_lora * f);
        w.q = b.take(C * lim.heads * lim.qk * f);
        w.kva = b.take(C * (lim.kv_lora + lim.rope) * f);
        w.gate = b.take(C * lim.heads * lim.v * f);
        w.ql = b.take(C * lim.heads * lim.kv_lora * f);
        w.ol = b.take(C * lim.heads * lim.kv_lora * f);
        w.att = b.take(C * lim.heads * lim.v * f);
        w.tk = b.take(C * lim.index_dim * f);
        w.iq = b.take(C * lim.index_heads * lim.index_dim * f);
        w.hw = b.take(C * lim.index_heads * f);
        w.logits = b.take(if (lim.index_heads > 0) C * groups * f else 0);
        w.keep = b.take(if (lim.index_heads > 0) C * groups else 0);
        w.rlog = b.take(C * lim.experts * f);
        w.ids = b.take(C * lim.top * 4);
        w.wts = b.take(C * lim.top * f);
        w.lat = b.take(C * lim.latent * f);
        w.acc = b.take(C * lim.latent * f);
        w.slots = b.take(C * lim.top * lim.latent * f);
        const wide = @max(@max(lim.width, lim.inter), lim.latent);
        w.xg = b.take(C * lim.latent * f);
        w.g = b.take(C * wide * f);
        w.u = b.take(C * wide * f);
        w.h = b.take(C * wide * f);
        w.eo = b.take(C * @max(lim.latent, lim.hidden) * f);
        w.idx = b.take(2 * C * lim.top * 4);
        w.sum = b.take(C * lim.hidden * f);
        w.emb = b.take(C * lim.engram_width * f);
        w.main = b.take(C * lim.hidden * f);
        w.keys = b.take(C * lim.hc * lim.hidden * f);
        w.mask = b.take(C);
        w.pos = b.take(3 * @as(u64, lim.ctx) * 4);
        w.tokens = b.take(C * 4);
        w.out = b.take(C * lim.hidden * f);
        w.head = b.take(@as(u64, lim.vocab) * f);
        w.buf = try cuda.DeviceBuffer.alloc(c.d, b.at);
        errdefer w.buf.free();
        const base = w.buf.ptr;
        const info = @typeInfo(Work).@"struct";
        inline for (info.field_names, info.field_types) |name, ty| {
            if (ty == u64) @field(w, name) += base;
        }
        w.host_ids = try a.alloc(i32, C * @max(lim.top, 1));
        errdefer a.free(w.host_ids);
        w.host_idx = try a.alloc(i32, 2 * C * @max(lim.top, 1));
        return w;
    }

    pub fn deinit(w: *Work) void {
        w.buf.free();
        w.a.free(w.host_ids);
        w.a.free(w.host_idx);
    }

    pub fn bytes(w: *const Work) u64 {
        return w.buf.len;
    }
};

/// One layer's cache slots: the latent and the rotated key per slot, and an indexed layer's token and group keys.
pub fn cacheBytes(d: *const Dims, ctx: u32) u64 {
    var n: u64 = @as(u64, ctx) * (d.kv_lora + d.rope) * 4;
    if (d.index) |ix| n += @as(u64, ctx) * ix.dim * 4 + @as(u64, ctx / ix.kpool) * ix.dim * 4 + 512;
    return n + 1024;
}

pub fn cacheAt(d: *const Dims, ctx: u32, base: u64) Cache {
    var b: Bump = .{};
    var c: Cache = .{ .lat = base + b.take(@as(u64, ctx) * d.kv_lora * 4), .rot = 0 };
    c.rot = base + b.take(@as(u64, ctx) * d.rope * 4);
    if (d.index) |ix| {
        c.tok = base + b.take(@as(u64, ctx) * ix.dim * 4);
        c.gk = base + b.take(@as(u64, ctx / ix.kpool) * ix.dim * 4);
    }
    return c;
}

fn f4(n: u64) u64 {
    return n * 4;
}

/// The Engram residual added to `w.streams` (engram_ref.residual): `w.emb` holds the fetched rows, `w.mask` which
/// tokens own an n-gram.
pub fn engramStep(w: *Work, e: *const Engram, H: u32, n: u32, t: u32) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const E = e.width();
    try k.rmsnorm(s, w.emb, e.head_dim, e.head_norms, e.heads, w.emb, e.head_dim, t * e.heads, e.head_dim, f32_eps);
    try c.lin(e.value, w.emb, E, t, w.main, H);
    try k.rmsnorm(s, w.main, H, e.pre_gamma, 1, w.main, H, t, H, f32_eps);
    for (e.keys, 0..) |kl, si| switch (kl) {
        .plain => |p| try k.gemm(s, w.emb, E, p.w, p.b, w.keys + f4(si * H), n * H, t, H, E, false),
        .exl3 => return error.Hy4Shape,
    };
    try k.rmsnorm(s, w.keys, H, e.norm_key, n, w.keys, H, t * n, H, f32_eps);
    try k.engramGate(s, w.streams, w.keys, w.main, e.norm_query, e.scales, w.mask, t, n, H, f32_eps);
}

/// Layer `L` over `t` rows of `w.streams` at cache slots [p0, p0 + t); `w.pos` holds every slot's 3D position. With
/// an Engram layer, `w.emb` (the fetched rows, `[t, heads * head_dim]`) and `w.mask` must already be filled.
pub fn layer(w: *Work, L: *const Layer, cache: Cache, p0: u32, t: u32) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const d = &L.dims;
    const n = d.hc;
    const H = d.hidden;
    const heads = d.heads;
    const qk = d.nope + d.rope;
    const kvl = d.kv_lora;
    if (t > w.lim.chunk or p0 + t > w.lim.ctx) return error.Hy4Shape;
    const rope: fwd.Kernels.Rope = .{ .pos = w.pos, .total = w.lim.ctx, .theta = d.theta, .sections = d.sections };

    if (L.engram) |*e| try engramStep(w, e, H, n, t);

    // attention
    const A = &L.attn;
    try k.hcPre(s, w.streams, L.attn_hc, w.mixed, w.post, t, n, H, d.eps, d.hc_eps, d.hc_magnitude);
    try k.rmsnorm(s, w.mixed, H, L.input_norm, 1, w.x, H, t, H, d.eps);
    try c.lin(A.q_a, w.x, H, t, w.qa, d.q_lora);
    try k.rmsnorm(s, w.qa, d.q_lora, A.q_a_norm, 1, w.qres, d.q_lora, t, d.q_lora, d.eps);
    try c.lin(A.q_b, w.qres, d.q_lora, t, w.q, heads * qk);
    try k.rope(s, w.q, heads * qk, t, heads, qk, d.nope, d.rope, rope, p0, 1);
    try c.lin(A.kv_a, w.x, H, t, w.kva, kvl + d.rope);
    try k.rmsnorm(s, w.kva, kvl + d.rope, A.kv_a_norm, 1, cache.lat + f4(@as(u64, p0) * kvl), kvl, t, kvl, d.eps);
    const rot = cache.rot + f4(@as(u64, p0) * d.rope);
    try k.copy2d(s, w.kva + f4(kvl), kvl + d.rope, rot, d.rope, t, d.rope);
    try k.rope(s, rot, d.rope, t, 1, d.rope, 0, d.rope, rope, p0, 1);
    try c.lin(A.gate, w.x, H, t, w.gate, heads * d.v);

    var keep: fwd.Kernels.Keep = .{};
    if (d.index) |ix| {
        const I = A.index orelse return error.Hy4Shape;
        const dim = ix.dim;
        const tok = cache.tok + f4(@as(u64, p0) * dim);
        try c.lin(I.token_key, w.x, H, t, w.tk, dim);
        try k.rmsnorm(s, w.tk, dim, I.token_key_norm, 1, tok, dim, t, dim, d.eps);
        const g_old = p0 / ix.kpool;
        const g_new = (p0 + t) / ix.kpool;
        try k.idxPool(s, cache.tok, I.key_value, I.key_gate, I.ape, I.key_norm, cache.gk, g_old, g_new - g_old, dim, ix.kpool, d.eps);
        try k.rope(s, cache.gk + f4(@as(u64, g_old) * dim), dim, g_new - g_old, 1, dim, dim - d.rope, d.rope, rope, @as(u64, g_old) * ix.kpool, ix.kpool);
        if (p0 + t > ix.topk) {
            const ld: u64 = w.lim.ctx / ix.kpool;
            try c.lin(I.query, w.qres, d.q_lora, t, w.iq, ix.heads * dim);
            try k.rope(s, w.iq, ix.heads * dim, t, ix.heads, dim, dim - d.rope, d.rope, rope, p0, 1);
            try c.lin(I.head_weight, w.x, H, t, w.hw, ix.heads);
            const hscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ix.heads * dim)));
            try k.idxScore(s, w.iq, w.hw, cache.gk, w.logits, ld, t, ix.heads, dim, g_new, p0, ix.kpool, hscale);
            try k.idxTopk(s, w.logits, ld, w.keep, ld, t, g_new, p0, ix.kpool, ix.topk / ix.kpool - 1);
            keep = .{ .map = w.keep, .ld = ld, .kpool = ix.kpool };
        }
    }
    try k.absorbQ(s, w.q, A.kv_b, w.ql, t, heads, qk, d.nope, d.v, kvl);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    try k.mlaAttn(s, w.ql, w.q, qk, d.nope, cache.lat, cache.rot, A.sinks, w.ol, t, heads, kvl, d.rope, p0, d.window orelse 0, keep, scale);
    try k.uvGate(s, w.ol, A.kv_b, w.gate, w.att, t, heads, d.nope, d.v, kvl);
    try c.lin(A.o, w.att, heads * d.v, t, w.mixed, H);
    try k.rmsnorm(s, w.mixed, H, A.post_norm, 1, w.y, H, t, H, d.eps);
    try k.hcPost(s, w.streams, w.post, w.y, t, n, H);

    // MLP
    try k.hcPre(s, w.streams, L.ffn_hc, w.mixed, w.post, t, n, H, d.eps, d.hc_eps, d.hc_magnitude);
    try k.rmsnorm(s, w.mixed, H, L.post_attention_norm, 1, w.x, H, t, H, d.eps);
    switch (L.mlp) {
        .dense => |*m| {
            const I = m.gate.n();
            try c.lin(m.gate, w.x, H, t, w.g, I);
            try c.lin(m.up, w.x, H, t, w.u, I);
            try k.swiglu(s, w.g, w.u, I, w.h, t, I, 0);
            try c.lin(m.down, w.h, I, t, w.mixed, H);
            try k.rmsnorm(s, w.mixed, H, m.norm, 1, w.y, H, t, H, d.eps);
        },
        .moe => |*m| try moe(w, m, H, t, d.eps),
    }
    try k.hcPost(s, w.streams, w.post, w.y, t, n, H);
}

fn moe(w: *Work, m: *const Moe, H: u32, t: u32, eps: f32) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const top = m.top;
    const L = m.latent;
    const WD = m.width;
    try c.lin(m.router, w.x, H, t, w.rlog, m.experts);
    try k.route(s, w.rlog, m.bias, t, m.experts, top, m.scale, w.ids, w.wts);
    try c.lin(m.pre, w.x, H, t, w.lat, L);
    const ids = w.host_ids[0 .. t * top];
    try c.download(std.mem.sliceAsBytes(ids), w.ids);
    // rows grouped by expert, ascending; each row's slot is token * top + k
    var count: [1024]u32 = @splat(0);
    if (m.experts > count.len) return error.Hy4Shape;
    for (ids) |e| {
        if (e < 0 or e >= m.experts) return error.Hy4Route;
        count[@intCast(e)] += 1;
    }
    var start: [1025]u32 = undefined;
    start[0] = 0;
    for (0..m.experts) |e| start[e + 1] = start[e] + count[e];
    var fill = start;
    const nrow = t * top;
    const gidx = w.host_idx[0..nrow];
    const sidx = w.host_idx[nrow .. 2 * nrow];
    for (ids, 0..) |e, slot| {
        const at = fill[@intCast(e)];
        fill[@intCast(e)] += 1;
        gidx[at] = @intCast(slot / top);
        sidx[at] = @intCast(slot);
    }
    try c.upload(w.idx, std.mem.sliceAsBytes(w.host_idx[0 .. 2 * nrow]));
    const gptr = w.idx;
    const sptr = w.idx + f4(nrow);
    for (0..m.experts) |e| {
        const rows = count[e];
        if (rows == 0) continue;
        const o = start[e];
        const ex = m.expert[e];
        try k.gather(s, w.lat, gptr + f4(o), w.xg, rows, L);
        try c.lin(ex[0], w.xg, L, rows, w.g, WD);
        try c.lin(ex[1], w.xg, L, rows, w.u, WD);
        try k.swiglu(s, w.g, w.u, WD, w.h, rows, WD, m.clamp);
        try c.lin(ex[2], w.h, WD, rows, w.eo, L);
        try k.scatter(s, w.eo, sptr + f4(o), w.slots, rows, L);
    }
    try k.combine(s, w.slots, w.wts, w.acc, t, top, L);
    try c.lin(m.post, w.acc, L, t, w.sum, H);
    const SW = m.shared[0].n();
    try c.lin(m.shared[0], w.x, H, t, w.g, SW);
    try c.lin(m.shared[1], w.x, H, t, w.u, SW);
    try k.swiglu(s, w.g, w.u, SW, w.h, t, SW, 0);
    try c.lin(m.shared[2], w.h, SW, t, w.mixed, H);
    try k.add(s, w.mixed, w.sum, @as(u64, t) * H);
    try k.rmsnorm(s, w.mixed, H, m.norm, 1, w.y, H, t, H, eps);
    // the host-side lists are reused by the next layer: wait until the uploads were read
    try c.s.synchronize();
}

/// The model's ends: embedding, the final hyper-head, norm and LM head.
pub const Head = struct {
    embed: u64, // bf16 [vocab, hidden]
    hc: fwd.Kernels.Hc,
    norm: W,
    lm_head: Lin,
};

/// `w.streams` from `t` token ids (uploaded by the caller into `w.tokens`).
pub fn embed(w: *Work, h: *const Head, t: u32, n: u32, hidden: u32) !void {
    try w.c.k.embed(w.c.s, w.tokens, h.embed, w.out, t, hidden);
    try w.c.k.expand(w.c.s, w.out, w.streams, t, n, hidden);
}

/// Streams row `row` -> logits in `w.head`.
pub fn logits(w: *Work, h: *const Head, row: u32, n: u32, hidden: u32, eps: f32, hc_eps: f32) !void {
    const st = w.streams + f4(@as(u64, row) * n * hidden);
    try w.c.k.hcHead(w.c.s, st, h.hc, w.out, 1, n, hidden, eps, hc_eps);
    try w.c.k.rmsnorm(w.c.s, w.out, hidden, h.norm, 1, w.mixed, hidden, 1, hidden, eps);
    try w.c.lin(h.lm_head, w.mixed, hidden, 1, w.head, h.lm_head.n());
}

/// Image features (merged rows, `[tokens, hidden]`, f32 on the device) written over the embedding rows of the
/// picture's placeholder tokens: rows [at, at + count) of `w.out` before `expand`.
pub fn embedWithImage(w: *Work, h: *const Head, t: u32, n: u32, hidden: u32, feats: []const ImageSpan) !void {
    try w.c.k.embed(w.c.s, w.tokens, h.embed, w.out, t, hidden);
    for (feats) |f| try w.c.k.copy2d(w.c.s, f.src, hidden, w.out + f4(@as(u64, f.row) * hidden), hidden, f.rows, hidden);
    try w.c.k.expand(w.c.s, w.out, w.streams, t, n, hidden);
}

pub const ImageSpan = struct { src: u64, row: u32, rows: u32 };
