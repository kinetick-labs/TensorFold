//! The Hy4-flash decoder on CUDA: fp32 activations through the forward kernels (fwd.zig), EXL3 projections through the
//! existing trellis kernels (cuda.exl3.Dense), plain weights read as stored. A forward call takes rows of one or more
//! sequences side by side (Seg): projections, MoE and norms run over every row at once, the cache work and attention
//! per sequence. Each sequence has per layer an absorbed MLA latent cache (normed 512-wide latent plus the rotated
//! 64-wide key, f32, bf16 or e4m3 with a per-row scale): a ring of window + chunk slots in a sliding layer, the whole
//! context in an indexed one, whose compressed indexer keeps a small ring of token keys and every group key. The
//! indexer hands attention a compact list of kept groups. The latent MoE runs grouped on the device (exl3_experts)
//! up to `Limits.grouped_rows` rows, else grouped on the host per chunk (each used expert over its rows). Engram rows
//! are fetched by the caller. Layer math follows layer_ref.zig / engram_ref.zig, which the GPU tests compare against.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const fwd = @import("fwd.zig");
const layer_ref = @import("layer_ref.zig");

pub const W = fwd.W;
pub const CacheDt = fwd.CacheDt;
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
    experts: ?*const cuda.exl3_experts.Experts = null,

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
    grouped: ?cuda.exl3_experts.Layer = null, // the same experts as one grouped layer (moe_grouped.zig)
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
    engram_layers: u32 = 1, // Engram layers whose rows the caller fetches up front
    vocab: u32 = 0,
    recall: u32 = 0, // the indexer's kept groups a row (topk / kpool)
    lanes: u32 = 1, // sequences a call may carry (one logits row each)
    grouped_rows: u32 = 0, // rows the device-grouped MoE takes at most (0: never)
    grouped_scratch: u64 = 0, // its scratch bytes for that many rows
};

/// One sequence's rows in a forward call: rows [row, row + rows) of the work buffers are its positions
/// [p0, p0 + rows); `pos` its device `[3, ctx]` 3D positions; `cache` its cache of the layer being run.
pub const Seg = struct { cache: Cache, pos: u64, p0: u32, row: u32, rows: u32 };

/// One layer's cache of one sequence: the latent slots (CacheView), and an indexed layer's token-key ring (f32,
/// `tslots` rows) and group keys (`gdt`, ctx / kpool rows).
pub const Cache = struct {
    view: fwd.Kernels.CacheView,
    tok: u64 = 0,
    tslots: u64 = 0,
    gk: u64 = 0,
    gdt: CacheDt = .f32,
};

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
    check_left: u32 = 0, // MoE calls still compared grouped against per-expert (HY4_MOE_CHECK)
    check_worst: f32 = 0,
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
    kvn: u64, // [chunk, kv_lora + rope]: normed latent and rotated key before the cache store
    gate: u64,
    ql: u64,
    ol: u64,
    att: u64,
    tk: u64,
    gks: u64, // [chunk / kpool + 2, index_dim]: new group keys before their store
    iq: u64,
    hw: u64,
    logits: u64, // [chunk, ctx / kpool]
    keep: u64, // u8 [chunk, ctx / kpool]
    list: u64, // i32 [chunk, recall]: the kept groups
    count: u64, // i32 [chunk]
    rlog: u64,
    ids: u64, // i32 [chunk, top]
    wts: u64,
    lat: u64,
    half: u64, // f16 [chunk, latent]: the grouped experts' input
    acc: u64,
    slots: u64, // [chunk * top, latent]
    xg: u64,
    g: u64,
    u: u64,
    h: u64,
    eo: u64,
    idx: u64, // i32 [2, chunk * top]
    sum: u64,
    emb: u64, // [engram_layers, chunk, engram_width]
    main: u64,
    keys: u64, // [chunk, hc, hidden]
    mask: u64, // u8 [chunk]
    tokens: u64, // i32 [chunk]
    out: u64, // [chunk, hidden]
    last: u64, // [lanes, hc, hidden]: the rows whose logits are wanted
    head: u64, // [lanes, vocab]
    gscr: u64, // the grouped experts' scratch

    pub fn init(a: Allocator, c: Ctx, lim: Limits) !Work {
        const C: u64 = lim.chunk;
        const f = 4;
        const groups: u64 = lim.ctx / @max(lim.kpool, 1);
        const lanes: u64 = @max(lim.lanes, 1);
        var b: Bump = .{};
        var w: Work = undefined;
        w.c = c;
        w.lim = lim;
        w.a = a;
        w.check_left = 0;
        w.check_worst = 0;
        w.streams = b.take(C * lim.hc * lim.hidden * f);
        w.mixed = b.take(C * lim.hidden * f);
        w.x = b.take(C * lim.hidden * f);
        w.y = b.take(C * lim.hidden * f);
        w.post = b.take(C * lim.hc * f);
        w.qa = b.take(C * lim.q_lora * f);
        w.qres = b.take(C * lim.q_lora * f);
        w.q = b.take(C * lim.heads * lim.qk * f);
        w.kva = b.take(C * (lim.kv_lora + lim.rope) * f);
        w.kvn = b.take(C * (lim.kv_lora + lim.rope) * f);
        w.gate = b.take(C * lim.heads * lim.v * f);
        w.ql = b.take(C * lim.heads * lim.kv_lora * f);
        w.ol = b.take(C * lim.heads * lim.kv_lora * f);
        w.att = b.take(C * lim.heads * lim.v * f);
        w.tk = b.take(C * lim.index_dim * f);
        w.gks = b.take((C / @max(lim.kpool, 1) + 2) * lim.index_dim * f);
        w.iq = b.take(C * lim.index_heads * lim.index_dim * f);
        w.hw = b.take(C * lim.index_heads * f);
        w.logits = b.take(if (lim.index_heads > 0) C * groups * f else 0);
        w.keep = b.take(if (lim.index_heads > 0) C * groups else 0);
        w.list = b.take(C * @max(lim.recall, 1) * 4);
        w.count = b.take(C * 4);
        w.rlog = b.take(C * lim.experts * f);
        w.ids = b.take(C * lim.top * 4);
        w.wts = b.take(C * lim.top * f);
        w.lat = b.take(C * lim.latent * f);
        w.half = b.take(C * lim.latent * 2);
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
        w.emb = b.take(@as(u64, @max(lim.engram_layers, 1)) * C * lim.engram_width * f);
        w.main = b.take(C * lim.hidden * f);
        w.keys = b.take(C * lim.hc * lim.hidden * f);
        w.mask = b.take(C);
        w.tokens = b.take(C * 4);
        w.out = b.take(C * lim.hidden * f);
        w.last = b.take(lanes * lim.hc * lim.hidden * f);
        w.head = b.take(lanes * @as(u64, lim.vocab) * f);
        w.gscr = b.take(if (lim.grouped_rows > 0) lim.grouped_scratch else 0);
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

    /// Engram layer `slot`'s fetched rows.
    pub fn embAt(w: *const Work, slot: u32) u64 {
        return w.emb + @as(u64, slot) * w.lim.chunk * w.lim.engram_width * 4;
    }
};

/// How many slots one layer's cache holds for a context of `ctx` and forward calls of up to `chunk` rows.
pub const Shape = struct {
    slots: u64, // latent slots: window + chunk (a ring) in a sliding layer, the context in an indexed one
    tslots: u64 = 0, // token-key ring rows
    groups: u64 = 0, // group-key rows

    pub fn of(d: *const Dims, ctx: u32, chunk: u32) Shape {
        var sh: Shape = .{ .slots = ctx };
        if (d.window) |win| sh.slots = @min(@as(u64, ctx), std.mem.alignForward(u64, @as(u64, win) + chunk, 16));
        if (d.index) |ix| {
            // the token keys a call's new groups read: up to kpool - 1 rows before its first, its own rows
            sh.tslots = @min(@as(u64, ctx), std.mem.alignForward(u64, @as(u64, chunk) + ix.kpool, ix.kpool));
            sh.groups = ctx / ix.kpool;
        }
        return sh;
    }
};

/// One layer's cache bytes for one sequence.
pub fn cacheBytes(d: *const Dims, ctx: u32, chunk: u32, dt: CacheDt) u64 {
    const sh = Shape.of(d, ctx, chunk);
    var n: u64 = align256(sh.slots * d.kv_lora * dt.latBytes()) + align256(sh.slots * d.rope * dt.rotBytes());
    if (dt == .fp8) n += align256(sh.slots * 4);
    if (d.index) |ix| n += align256(sh.tslots * ix.dim * 4) + align256(sh.groups * ix.dim * dt.keyDt().rotBytes());
    return n;
}

/// The cache of cacheBytes() at `base`.
pub fn cacheAt(d: *const Dims, ctx: u32, chunk: u32, dt: CacheDt, base: u64) Cache {
    const sh = Shape.of(d, ctx, chunk);
    var b: Bump = .{};
    var c: Cache = .{ .view = .{ .lat = base + b.take(sh.slots * d.kv_lora * dt.latBytes()), .rot = 0, .dt = dt, .slots = sh.slots } };
    c.view.rot = base + b.take(sh.slots * d.rope * dt.rotBytes());
    if (dt == .fp8) c.view.scale = base + b.take(sh.slots * 4);
    if (d.index) |ix| {
        c.tok = base + b.take(sh.tslots * ix.dim * 4);
        c.tslots = sh.tslots;
        c.gdt = dt.keyDt();
        c.gk = base + b.take(sh.groups * ix.dim * c.gdt.rotBytes());
    }
    return c;
}

fn align256(n: u64) u64 {
    return std.mem.alignForward(u64, @max(n, 1), 256);
}

fn f4(n: u64) u64 {
    return n * 4;
}

/// The Engram residual added to `w.streams` (engram_ref.residual): `emb` holds the fetched rows (`[t, width]`, normed
/// in place), `w.mask` which tokens own an n-gram.
pub fn engramStep(w: *Work, e: *const Engram, emb: u64, H: u32, n: u32, t: u32) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const E = e.width();
    try k.rmsnorm(s, emb, e.head_dim, e.head_norms, e.heads, emb, e.head_dim, t * e.heads, e.head_dim, f32_eps);
    try c.lin(e.value, emb, E, t, w.main, H);
    try k.rmsnorm(s, w.main, H, e.pre_gamma, 1, w.main, H, t, H, f32_eps);
    for (e.keys, 0..) |kl, si| switch (kl) {
        .plain => |p| try k.gemm(s, emb, E, p.w, p.b, w.keys + f4(si * H), n * H, t, H, E, false),
        .exl3 => return error.Hy4Shape,
    };
    try k.rmsnorm(s, w.keys, H, e.norm_key, n, w.keys, H, t * n, H, f32_eps);
    try k.engramGate(s, w.streams, w.keys, w.main, e.norm_query, e.scales, w.mask, t, n, H, f32_eps);
}

/// Layer `L` over the rows of `segs` (side by side in `w.streams`, each its sequence's positions [p0, p0 + rows) with
/// that sequence's cache of this layer). With an Engram layer, `emb` (the fetched rows, `[t, heads * head_dim]`) and
/// `w.mask` must already be filled.
pub fn layer(w: *Work, L: *const Layer, segs: []const Seg, emb: u64) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const d = &L.dims;
    const n = d.hc;
    const H = d.hidden;
    const heads = d.heads;
    const qk = d.nope + d.rope;
    const kvl = d.kv_lora;
    const kw = kvl + d.rope;
    var t: u32 = 0;
    for (segs) |g| {
        if (g.row != t or g.p0 + g.rows > w.lim.ctx) return error.Hy4Shape;
        t += g.rows;
    }
    if (t == 0) return;
    if (t > w.lim.chunk) return error.Hy4Shape;

    if (L.engram) |*e| try engramStep(w, e, emb, H, n, t);

    // attention: projections over every row
    const A = &L.attn;
    try k.hcPre(s, w.streams, L.attn_hc, w.mixed, w.post, t, n, H, d.eps, d.hc_eps, d.hc_magnitude);
    try k.rmsnorm(s, w.mixed, H, L.input_norm, 1, w.x, H, t, H, d.eps);
    try c.lin(A.q_a, w.x, H, t, w.qa, d.q_lora);
    try k.rmsnorm(s, w.qa, d.q_lora, A.q_a_norm, 1, w.qres, d.q_lora, t, d.q_lora, d.eps);
    try c.lin(A.q_b, w.qres, d.q_lora, t, w.q, heads * qk);
    try c.lin(A.kv_a, w.x, H, t, w.kva, kw);
    try k.rmsnorm(s, w.kva, kw, A.kv_a_norm, 1, w.kvn, kw, t, kvl, d.eps);
    try k.copy2d(s, w.kva + f4(kvl), kw, w.kvn + f4(kvl), kw, t, d.rope);
    try c.lin(A.gate, w.x, H, t, w.gate, heads * d.v);
    var ix_query = false;
    if (d.index) |ix| {
        const I = A.index orelse return error.Hy4Shape;
        try c.lin(I.token_key, w.x, H, t, w.tk, ix.dim);
        for (segs) |g| ix_query = ix_query or g.p0 + g.rows > ix.topk;
        if (ix_query) {
            try c.lin(I.query, w.qres, d.q_lora, t, w.iq, ix.heads * ix.dim);
            try c.lin(I.head_weight, w.x, H, t, w.hw, ix.heads);
        }
    }
    try k.absorbQ(s, w.q, A.kv_b, w.ql, t, heads, qk, d.nope, d.v, kvl);

    // per sequence: positions, the cache, the indexer, attention
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    for (segs) |g| {
        const r0: u64 = g.row;
        const rows = g.rows;
        const p0 = g.p0;
        const cache = g.cache;
        const rope: fwd.Kernels.Rope = .{ .pos = g.pos, .total = w.lim.ctx, .theta = d.theta, .sections = d.sections };
        try k.rope(s, w.q + f4(r0 * heads * qk), heads * qk, rows, heads, qk, d.nope, d.rope, rope, p0, 1);
        try k.rope(s, w.kvn + f4(r0 * kw), kw, rows, 1, d.rope, kvl, d.rope, rope, p0, 1);
        if (d.window) |win| if (cache.view.slots < @min(@as(u64, w.lim.ctx), @as(u64, win) + rows - 1)) return error.Hy4Shape;
        try k.cacheStore(s, w.kvn + f4(r0 * kw), kw, kvl, d.rope, cache.view, p0, rows);
        var keep: fwd.Kernels.Keep = .{};
        if (d.index) |ix| {
            const I = A.index.?;
            const dim = ix.dim;
            // the rows' token keys (normed) into the ring, split where it wraps
            if (cache.tslots < rows + ix.kpool - 1) return error.Hy4Shape;
            var done: u32 = 0;
            while (done < rows) {
                const at = (@as(u64, p0) + done) % cache.tslots;
                const run: u32 = @intCast(@min(@as(u64, rows - done), cache.tslots - at));
                try k.rmsnorm(s, w.tk + f4((r0 + done) * dim), dim, I.token_key_norm, 1, cache.tok + f4(at * dim), dim, run, dim, d.eps);
                done += run;
            }
            const g_old = p0 / ix.kpool;
            const g_new = (p0 + rows) / ix.kpool;
            const fresh = g_new - g_old;
            if (fresh > w.lim.chunk / @max(w.lim.kpool, 1) + 2) return error.Hy4Shape;
            try k.idxPool(s, cache.tok, cache.tslots, I.key_value, I.key_gate, I.ape, I.key_norm, w.gks, g_old, fresh, dim, ix.kpool, d.eps);
            try k.rope(s, w.gks, dim, fresh, 1, dim, dim - d.rope, d.rope, rope, @as(u64, g_old) * ix.kpool, ix.kpool);
            try k.storeRows(s, w.gks, dim, cache.gk, cache.gdt, dim, g_old, fresh);
            if (p0 + rows > ix.topk) {
                const ld: u64 = w.lim.ctx / ix.kpool;
                const recall = ix.topk / ix.kpool - 1;
                if (recall > w.lim.recall) return error.Hy4Shape;
                const iq = w.iq + f4(r0 * ix.heads * dim);
                try k.rope(s, iq, ix.heads * dim, rows, ix.heads, dim, dim - d.rope, d.rope, rope, p0, 1);
                const hscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ix.heads * dim)));
                const lg = w.logits + f4(r0 * ld);
                try k.idxScore(s, iq, w.hw + f4(r0 * ix.heads), cache.gk, cache.gdt, lg, ld, rows, ix.heads, dim, g_new, p0, ix.kpool, hscale);
                const list = w.list + 4 * r0 * w.lim.recall;
                const count = w.count + 4 * r0;
                try k.idxTopk(s, lg, ld, w.keep + r0 * ld, ld, rows, g_new, p0, ix.kpool, recall, list, w.lim.recall, count);
                keep = .{ .list = list, .ld = w.lim.recall, .count = count, .kpool = ix.kpool };
            }
        }
        try k.mlaAttn(s, w.ql + f4(r0 * heads * kvl), w.q + f4(r0 * heads * qk), qk, d.nope, cache.view, A.sinks, w.ol + f4(r0 * heads * kvl), rows, heads, kvl, d.rope, p0, d.window orelse 0, keep, scale);
    }
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

/// One sequence at positions [p0, p0 + t) in rows [0, t): layer() with a single Seg.
pub fn layerOne(w: *Work, L: *const Layer, cache: Cache, pos: u64, p0: u32, t: u32, emb: u64) !void {
    const one = [1]Seg{.{ .cache = cache, .pos = pos, .p0 = p0, .row = 0, .rows = t }};
    return layer(w, L, &one, emb);
}

fn moe(w: *Work, m: *const Moe, H: u32, t: u32, eps: f32) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const L = m.latent;
    try c.lin(m.router, w.x, H, t, w.rlog, m.experts);
    try k.route(s, w.rlog, m.bias, t, m.experts, m.top, m.scale, w.ids, w.wts);
    try c.lin(m.pre, w.x, H, t, w.lat, L);
    const grouped = if (m.grouped) |gl| (if (c.experts != null and t <= w.lim.grouped_rows) gl else null) else null;
    if (grouped) |gl| {
        try k.toHalf(s, w.lat, w.half, @as(u64, t) * L);
        const sc = try cuda.exl3_experts.scratch(w.gscr, gl, w.lim.grouped_rows, m.top);
        try c.experts.?.run(s, gl, w.half, L, .f16, w.ids, w.wts, t, w.acc, sc);
        if (w.check_left > 0) {
            // HY4_MOE_CHECK: the per-expert path beside it, the worst relative difference logged
            w.check_left -= 1;
            try perExpert(w, m, t, w.eo);
            const got = try w.a.alloc(f32, @as(usize, t) * L);
            defer w.a.free(got);
            const want = try w.a.alloc(f32, @as(usize, t) * L);
            defer w.a.free(want);
            try c.download(std.mem.sliceAsBytes(got), w.acc);
            try c.download(std.mem.sliceAsBytes(want), w.eo);
            var e: f32 = 0;
            var big: f32 = 0;
            for (want) |v| big = @max(big, @abs(v));
            for (got, want) |x, v| e = @max(e, @abs(x - v) / @max(big, 1e-6));
            w.check_worst = @max(w.check_worst, e);
            std.log.info("hy4 moe check: grouped vs per-expert over {d} rows, worst {e} of the largest |value| {e}", .{ t, e, big });
        }
    } else try perExpert(w, m, t, w.acc);
    try c.lin(m.post, w.acc, L, t, w.sum, H);
    const SW = m.shared[0].n();
    try c.lin(m.shared[0], w.x, H, t, w.g, SW);
    try c.lin(m.shared[1], w.x, H, t, w.u, SW);
    try k.swiglu(s, w.g, w.u, SW, w.h, t, SW, 0);
    try c.lin(m.shared[2], w.h, SW, t, w.mixed, H);
    try k.add(s, w.mixed, w.sum, @as(u64, t) * H);
    try k.rmsnorm(s, w.mixed, H, m.norm, 1, w.y, H, t, H, eps);
}

/// The routed experts grouped on the host: `w.ids`, `w.wts` and `w.lat` from moe(), the weighted sum into `acc`.
fn perExpert(w: *Work, m: *const Moe, t: u32, acc: u64) !void {
    const c = w.c;
    const k = c.k;
    const s = c.s;
    const top = m.top;
    const L = m.latent;
    const WD = m.width;
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
    try k.combine(s, w.slots, w.wts, acc, t, top, L);
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
    const one = [1]u32{row};
    return logitsOf(w, h, &one, n, hidden, eps, hc_eps);
}

/// The streams rows `rows` -> logits rows [0, rows.len) of `w.head` (`vocab` floats apart), in one LM-head pass.
pub fn logitsOf(w: *Work, h: *const Head, rows: []const u32, n: u32, hidden: u32, eps: f32, hc_eps: f32) !void {
    const m: u32 = @intCast(rows.len);
    if (m == 0) return;
    if (m > @max(w.lim.lanes, 1)) return error.Hy4Shape;
    const row_w: u64 = @as(u64, n) * hidden;
    for (rows, 0..) |r, i| try w.c.k.copy2d(w.c.s, w.streams + f4(@as(u64, r) * row_w), row_w, w.last + f4(i * row_w), row_w, 1, @intCast(row_w));
    try w.c.k.hcHead(w.c.s, w.last, h.hc, w.out, m, n, hidden, eps, hc_eps);
    try w.c.k.rmsnorm(w.c.s, w.out, hidden, h.norm, 1, w.mixed, hidden, m, hidden, eps);
    try w.c.lin(h.lm_head, w.mixed, hidden, m, w.head, h.lm_head.n());
}

/// Image features (merged rows, `[tokens, hidden]`, f32 on the device) written over the embedding rows of the
/// picture's placeholder tokens: rows [at, at + count) of `w.out` before `expand`.
pub fn embedWithImage(w: *Work, h: *const Head, t: u32, n: u32, hidden: u32, feats: []const ImageSpan) !void {
    try w.c.k.embed(w.c.s, w.tokens, h.embed, w.out, t, hidden);
    for (feats) |f| try w.c.k.copy2d(w.c.s, f.src, hidden, w.out + f4(@as(u64, f.row) * hidden), hidden, f.rows, hidden);
    try w.c.k.expand(w.c.s, w.out, w.streams, t, n, hidden);
}

pub const ImageSpan = struct { src: u64, row: u32, rows: u32 };
