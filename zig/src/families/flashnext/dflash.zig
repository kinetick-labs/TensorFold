//! DFlash2 drafts for Flash Next (qwen4_exp), in place of the MTP head's: incoai/Qwen3.8-Flash-Next-DFlash2 (read
//! unmodified) after z-lab (MIT). qwen3_5's CUDA drafter (cuda_draft*.zig, qwen35_draft.cu) over the kernels and
//! target weights it borrows here: the target's taps are its layers 5/15/25/35/45 completed, the four residual streams
//! averaged (the SGLang capture), written by the batched pass (cuda_batch.zig's tapAfter); the head is the target's
//! lm_head packed affine-4 at load. A hold drafts a block's 7 rows as one chain the window verifies like the head's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const weights = @import("dflash_weights.zig");
pub const tree = @import("dflash_tree.zig");
pub const frame = @import("dflash_frame.zig");
pub const config = @import("dflash_config.zig");
pub const Config = config.Config;

const DeviceBuffer = cuda.DeviceBuffer;
const Affine4 = cuda.qlinear.Affine4;
const fp8 = @import("cuda_fp8.zig");
pub const Lin = weights.Lin;

/// Rows an absorb runs at once (a prompt chunk's taps in pieces).
const max_rows = 128;
/// The logprob row's words (sample.cu's LP_WORDS).
const lp_words = 3 + 2 * 20;
/// A window's rows at most (the held drafts of a grown tree).
pub const most_window = 64;

/// One stream's request for drafts: its pending token at `position` (its context's end + 1) and how many.
pub const Hold = struct {
    pending: u32,
    position: u64,
    depth: u32,
    lanes: ?*const lanes.shape.Shape = null,
    tokens: []const u32 = &.{},
    sampling: ?lanes.Sampling = null,
};

/// The embedding table the target reads: affine-4 groups of 32, or bf16 rows.
pub const Table = struct { w: u64, s: u64, b: u64, dense: bool };

/// What qwen3_5's drafter reads off its engine: a stream, the affine-4 tiles, top-k, three row kernels, the
/// target's embedding (through our glue) and its head (packed here).
pub const Engine = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    mods: [5]cuda.Module,
    affine: cuda.qlinear.Affine4Kernels,
    draws: cuda.sample.Draws,
    fns: struct { sums: cuda.Function, norm: cuda.Function, mlp_act: cuda.Function },
    embedder: Embedder,
    hidden: usize,
    /// the target engine's MXFP8 lane matmul (TF_FLASHNEXT_DFLASH_MX: the drafter's projections in MXFP8)
    k8: ?*const fp8.Kernels = null,
    mx: bool = false,
    head: Affine4 = undefined,
    head_arena: ?DeviceBuffer = null,
    q35: cuda.Module = undefined,

    /// The target's token embedding: `rows` u32 ids into out [rows, hidden] bf16 (one copy, not the streams).
    pub const Embedder = struct {
        ctx: *anyopaque,
        f: *const fn (ctx: *anyopaque, ids: u64, out: u64, rows: usize) anyerror!void,
    };

    pub fn init(d: *const cuda.Driver, stream: cuda.Stream, embedder: Embedder, hidden: usize) !Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.sample };
        var e: Engine = .{ .d = d, .stream = stream, .mods = undefined, .affine = undefined, .draws = undefined, .fns = undefined, .embedder = embedder, .hidden = hidden };
        var loaded: usize = 0;
        errdefer for (e.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            e.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        var q35 = try cuda.Module.load(d, kk.qwen35);
        errdefer q35.unload();
        var device: c_int = 0;
        var sms: c_int = 0;
        var major: c_int = 0;
        var minor: c_int = 0;
        try d.check(d.api.cuDeviceGet(&device, 0), "cuDeviceGet");
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "cuDeviceGetAttribute");
        try d.check(d.api.cuDeviceGetAttribute(&major, .compute_capability_major, device), "cuDeviceGetAttribute");
        try d.check(d.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, device), "cuDeviceGetAttribute");
        const gb10 = major == 12 and minor == 1;
        e.affine = try cuda.qlinear.Affine4Kernels.resolve(d, e.mods[0], e.mods[1], e.mods[2], e.mods[3], @intCast(sms), gb10);
        e.affine.half_prompt = !gb10;
        e.draws = try cuda.sample.Draws.resolve(e.mods[4]);
        e.fns = .{ .sums = try q35.function("tf_q35_sums"), .norm = try q35.function("tf_q35_norm"), .mlp_act = try q35.function("tf_q35_mlp_act") };
        e.q35 = q35;
        return e;
    }

    pub fn deinit(e: *Engine) void {
        if (e.head_arena) |*a| a.free();
        e.affine.deinit();
        e.q35.unload();
        for (&e.mods) |*m| m.unload();
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// The target's embedding of `rows` ids into out [rows, hidden] bf16 (one copy, not the four streams).
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, _: u64) !void {
        return e.embedder.f(e.embedder.ctx, ids, out, rows);
    }

    /// The target's lm_head (bf16 in the checkpoint) packed affine-4, group 64, as the drafter's own projections.
    pub fn loadHead(e: *Engine, gpa: std.mem.Allocator, io: std.Io, target_dir: []const u8, vocab: usize) !void {
        var ck = try @import("core").Checkpoint.openModel(gpa, io, target_dir);
        defer ck.close();
        const l = Affine4.layout(vocab, e.hidden);
        var w: weights.Weights = .{ .fc = undefined, .hidden_norm = 0, .final_norm = 0, .selector = undefined, .predecessor = &.{}, .successor = &.{}, .layers = &.{}, .arena = try DeviceBuffer.alloc(e.d, std.mem.alignForward(usize, l.words, 256) + 2 * std.mem.alignForward(usize, l.scales, 256) + 1024) };
        errdefer w.arena.free();
        var stage = try DeviceBuffer.alloc(e.d, weights.rawBytes(vocab, e.hidden));
        defer stage.free();
        e.head = try weights.linear(gpa, e, &stage, &w, &ck, "lm_head.weight", vocab, e.hidden);
        e.head_arena = w.arena;
    }
};

/// A stream's context: its layers' key and value rings over [begin, end), and the drafts it holds (a chain or tree).
pub const Ctx = struct {
    ring: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    held: [most_window]u32 = undefined,
    held_n: usize = 0,
    probs: [most_window]f64 = undefined,
    parents: [most_window]i32 = undefined,
    own: [most_window]f64 = undefined,
    grown: usize = 0,
    scanned: usize = 0,
    wide: bool = false,
    /// the sequence's reset count this context follows (the engine's)
    gen: u64 = 0,

    /// A fresh sequence: nothing absorbed, nothing held (the ring's rows are overwritten as it fills).
    pub fn reset(ctx: *Ctx) void {
        ctx.* = .{ .ring = ctx.ring };
    }
};

const Fns = struct { gather_taps: cuda.Function, gather_rows: cuda.Function, dyn_conv: cuda.Function, norm_rope: cuda.Function, ring_put: cuda.Function, attn: cuda.Function, dyn_conv_batch: cuda.Function, norm_rope_batch: cuda.Function, mask_rows: cuda.Function, attn_partial: cuda.Function, attn_merge: cuda.Function };

pub const Role = enum { prompt, live, proposal };

pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *Engine,
    projection_role: Role = .proposal,
    c: Config,
    w: weights.Weights,
    mod: cuda.Module,
    fns: Fns,
    scratch: DeviceBuffer,
    selection_base: usize = 0,
    drawn: struct { sampling: ?lanes.Sampling = null, position: u64 = 0 } = .{},
    segments: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    po: u64 = 0,
    taps_in: u64 = 0,
    th: u64 = 0,
    thn: u64 = 0,
    kabs: u64 = 0,
    kn: u64 = 0,
    vabs: u64 = 0,
    xs: u64 = 0,
    h: u64 = 0,
    h2: u64 = 0,
    x: u64 = 0,
    xin: u64 = 0,
    dyn: u64 = 0,
    qraw: u64 = 0,
    q: u64 = 0,
    kraw: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    att: u64 = 0,
    o: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    act: u64 = 0,
    logits: u64 = 0,
    words: u64 = 0,
    parts: u64 = 0,
    hp: u64 = 0,
    ids: u64 = 0,
    staged: cuda.HostBuffer,
    pool: std.ArrayList(*Ctx) = .empty,
    landing: tree.Landing = .{},
    /// One level of a hold (a hold drafts every level at once; timed at load).
    step_ms: f64 = 1.0,
    frame_streams: usize = 1,
    // TF_DFLASH_LOG=n: the first n holds' drafts logged; and counters
    log: u64 = 0,
    holds: u64 = 0,

    const words_at = frame.words_at;
    const hp_at = frame.hp_at;

    /// The drafter from `dir` (config.json, model.safetensors) over `e`, its head packed from `target_dir`.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, target_dir: []const u8, target: config.Target, streams: usize) !*Draft {
        const c = try Config.read(gpa, io, dir, target);
        try e.loadHead(gpa, io, target_dir, c.vocab);
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .c = c, .w = undefined, .mod = undefined, .fns = undefined, .scratch = undefined, .staged = undefined, .frame_streams = @max(1, @min(streams, frame.max_streams)) };
        if (std.c.getenv("TF_DFLASH_LOG")) |v| d.log = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0;
        d.mod = try cuda.Module.load(e.d, cuda.kernels.qwen35_draft);
        errdefer d.mod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(d.fns, name) = try d.mod.function("tf_q35d_" ++ name);
        d.w = try weights.load(gpa, io, e, dir, .{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .vocab = c.vocab, .q_width = c.qWidth(), .kv_width = c.kvWidth(), .head_dim = c.head_dim, .dynamic = c.dynamic(), .rank = c.rank, .taps = c.n_taps }, e.mx);
        errdefer d.w.deinit(gpa);
        d.scratch = try DeviceBuffer.alloc(e.d, try d.layout(false));
        errdefer d.scratch.free();
        try d.scratch.fill8(0, null);
        _ = try d.layout(true);
        d.staged = try cuda.HostBuffer.alloc(e.d, try frame.stagedBytes(c.rank));
        errdefer d.staged.free();
        // the first context, and a hold timed on it (the quickest of several: the first ones pay the driver's stalls)
        const ctx = try d.create();
        errdefer d.drop(ctx);
        try e.stream.synchronize();
        try d.absorb(ctx, d.taps_in, 0, 1, true);
        var quickest: f64 = std.math.inf(f64);
        for (0..8) |_| {
            ctx.end = 1;
            const t0 = std.Io.Clock.awake.now(io);
            try d.hold(&.{.{ .pending = 0, .position = 2, .depth = @intCast(c.block - 1) }}, &.{ctx});
            quickest = @min(quickest, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6);
        }
        d.step_ms = quickest / @as(f64, @floatFromInt(c.block - 1));
        ctx.reset();
        try d.pool.append(gpa, ctx);
        std.log.info("dflash2: {s}: {d} layers, taps {any}, block {d}, a hold {d:.2} ms", .{ dir, c.layers, c.taps[0..c.n_taps], c.block, quickest });
        return d;
    }

    pub fn taps(d: *const Draft) []const u32 {
        return d.c.taps[0..d.c.n_taps];
    }

    /// One row's taps in bytes (bf16).
    pub fn tapRowBytes(d: *const Draft) usize {
        return d.c.n_taps * d.c.hidden * 2;
    }

    /// Device bytes a stream's context takes.
    pub fn streamBytes(d: *const Draft) usize {
        return d.c.ringBytes();
    }

    pub fn deinit(d: *Draft) void {
        for (d.pool.items) |ctx| d.drop(ctx);
        d.pool.deinit(d.gpa);
        d.staged.free();
        d.scratch.free();
        d.w.deinit(d.gpa);
        d.mod.unload();
        d.gpa.destroy(d);
    }

    fn bytes(factors: anytype) !usize {
        var value: usize = 1;
        inline for (factors) |factor| value = try std.math.mul(usize, value, factor);
        return value;
    }

    /// The scratch rows, measured (nothing assigned) or assigned from the scratch buffer.
    pub fn layout(d: *Draft, assign: bool) !usize {
        const c = d.c;
        const B = c.block * d.frame_streams;
        const R = @max(max_rows, B);
        const sizes = .{
            .{ "taps_in", try bytes(.{ R, c.n_taps, c.hidden, 2 }) },                 .{ "th", try bytes(.{ R, c.hidden, 2 }) },              .{ "thn", try bytes(.{ R, c.hidden, 2 }) },                         .{ "kabs", try bytes(.{ R, c.kvWidth(), 2 }) },
            .{ "kn", try bytes(.{ R, c.kvWidth(), 2 }) },                             .{ "vabs", try bytes(.{ R, c.kvWidth(), 2 }) },         .{ "xs", try bytes(.{ R, c.n_taps, (c.hidden / 64), 4 }) },         .{ "h", try bytes(.{ B, c.hidden, 2 }) },
            .{ "h2", try bytes(.{ B, c.hidden, 2 }) },                                .{ "x", try bytes(.{ B, c.hidden, 2 }) },               .{ "xin", try bytes(.{ B, c.hidden, 2 }) },                         .{ "dyn", try bytes(.{ B, c.dynamic(), 2 }) },
            .{ "qraw", try bytes(.{ B, c.qWidth(), 2 }) },                            .{ "q", try bytes(.{ B, c.qWidth(), 2 }) },             .{ "kraw", try bytes(.{ B, c.kvWidth(), 2 }) },                     .{ "kb", try bytes(.{ B, c.kvWidth(), 2 }) },
            .{ "vb", try bytes(.{ B, c.kvWidth(), 2 }) },                             .{ "att", try bytes(.{ B, c.qWidth(), 2 }) },           .{ "o", try bytes(.{ B, c.hidden, 2 }) },                           .{ "gate", try bytes(.{ B, c.intermediate, 2 }) },
            .{ "up", try bytes(.{ B, c.intermediate, 2 }) },                          .{ "act", try bytes(.{ B, c.intermediate, 2 }) },       .{ "logits", try bytes(.{ B, c.vocab, 2 }) },                       .{ "words", try bytes(.{ B, lp_words, 4 }) },
            .{ "segments", try bytes(.{ d.frame_streams, @sizeOf(frame.Segment) }) }, .{ "pm", try bytes(.{ B, c.heads, frame.chunks, 4 }) }, .{ "pl", try bytes(.{ B, c.heads, frame.chunks, 4 }) },             .{ "po", try bytes(.{ B, c.heads, frame.chunks, c.head_dim, 4 }) },
            .{ "hp", try bytes(.{ B, c.rank, 2 }) },                                  .{ "ids", try bytes(.{ B, 4 }) },                       .{ "parts", cuda.sample.Draws.topkPartsBytes(c.vocab, c.topk, B) },
        };
        var total: usize = 0;
        inline for (sizes) |s| {
            const at = (try std.math.add(usize, total, 255)) & ~@as(usize, 255);
            total = try std.math.add(usize, at, s[1]);
            if (assign) @field(d, s[0]) = d.scratch.ptr + at;
        }
        return total;
    }

    fn drop(d: *Draft, ctx: *Ctx) void {
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    /// A new context, its ring touched once (an absorb, then a hold), so no request pays the driver's stall.
    fn create(d: *Draft) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try DeviceBuffer.alloc(d.e.d, d.c.ringBytes()) };
        errdefer ctx.ring.free();
        try ctx.ring.fill8(0, null);
        try d.absorb(ctx, d.taps_in, 0, 1, true); // any tap rows do: the drafter's own zeroed staging
        try d.hold(&.{.{ .pending = 0, .position = 2, .depth = @intCast(d.c.block - 1) }}, &.{ctx});
        ctx.reset();
        return ctx;
    }

    /// A stream's context, from the pool or new; `release` returns it.
    pub fn open(d: *Draft) !*Ctx {
        const ctx = d.pool.pop() orelse try d.create();
        ctx.reset();
        return ctx;
    }

    pub fn release(d: *Draft, ctx: *Ctx) void {
        d.pool.append(d.gpa, ctx) catch d.drop(ctx);
    }

    fn ringK(d: *Draft, ctx: *Ctx, layer: usize) u64 {
        return ctx.ring.ptr + layer * 2 * d.c.ringRows() * d.c.kvWidth() * 2;
    }

    fn ringV(d: *Draft, ctx: *Ctx, layer: usize) u64 {
        return d.ringK(ctx, layer) + d.c.ringRows() * d.c.kvWidth() * 2;
    }

    pub fn proj(d: *Draft, l: anytype, x: u64, out: u64, rows: usize) !void {
        return d.projSums(l, x, out, rows, true);
    }

    pub fn projAgain(d: *Draft, l: anytype, x: u64, out: u64, rows: usize) !void {
        return d.projSums(l, x, out, rows, false);
    }

    pub fn projMany(d: *Draft, ls: []const Lin, x: u64, outs: []const u64, rows: usize) !void {
        const e = d.e;
        var any_mx = false;
        for (ls) |l| any_mx = any_mx or l.mx != null;
        if (any_mx or rows > 16 or ls.len > cuda.qlinear.Affine4Kernels.segs_max) {
            for (ls, outs, 0..) |l, out, i| try d.projSums(l, x, out, rows, i == 0);
            return;
        }
        try e.run(e.fns.sums, .{ rows, (ls[0].a.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(ls[0].a.k)), @as(u64, 0) });
        var many: [cuda.qlinear.Affine4Kernels.segs_max]cuda.qlinear.Affine4Kernels.Out = undefined;
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a, .out = out };
        try e.affine.decodeMany(e.stream, x, d.xs, many[0..ls.len], rows);
    }

    /// A projection (a Lin or a bare Affine4): MXFP8 on the engine's lane matmul (the 64-group sums still taken when
    /// asked, for affine-4 projections after it on the same rows), else qwen3_5's affine path.
    pub fn projSums(d: *Draft, l: anytype, x: u64, out: u64, rows: usize, sums: bool) !void {
        if (@TypeOf(l) == Lin) {
            if (l.mx) |m| {
                const e = d.e;
                if (sums) try e.run(e.fns.sums, .{ rows, (l.a.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.a.k)), @as(u64, 0) });
                return fp8.matmul(e.k8 orelse return error.NoFp8Kernels, e.stream, x, l.a.k, m, out, false, rows);
            }
            return d.projAffine(l.a, x, out, rows, sums);
        }
        return d.projAffine(l, x, out, rows, sums);
    }

    /// qwen3_5's cuda_draft_precision.affine: a prompt's rows past 16 on the prompt GEMM, else the verify tiles.
    fn projAffine(d: *Draft, l: Affine4, x: u64, out: u64, rows: usize, sums: bool) !void {
        const e = d.e;
        if (d.projection_role == .prompt and rows > 16) return e.affine.prompt(e.stream, x, l, out, rows);
        if (sums) try e.run(e.fns.sums, .{ rows, (l.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
        const tile = e.affine.verifyRows();
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            try e.affine.verify(e.stream, x + at * l.k * 2, d.xs + at * (l.k / 64) * 4, l, out + at * l.n * 2, @min(tile, rows - at));
        }
    }

    pub fn norm(d: *Draft, h: u64, w: u64, out: u64, rows: usize) !void {
        const e = d.e;
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, @as(u64, 0), w, out, d.c.eps, @as(c_int, @intCast(d.c.hidden)), @as(u64, 0), d.xs });
    }

    fn normRope(d: *Draft, x: u64, w: u64, out: u64, rows: usize, heads: usize, pos0: u64) !void {
        try d.e.run(d.fns.norm_rope, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * d.c.head_dim)), @as(c_longlong, @intCast(pos0)), d.c.eps, d.c.theta });
    }

    /// Target rows [start, start + n), their taps back to back at `taps` (bf16 rows), into the context: a start
    /// behind its end rewinds it (drafts the window refused), one past it leaves a gap (a restored prompt).
    pub fn absorb(d: *Draft, ctx: *Ctx, taps_: u64, start_: u64, n_: usize, prompt: bool) !void {
        const previous = d.projection_role;
        defer d.projection_role = previous;
        d.projection_role = if (prompt) .prompt else .live;
        const c = d.c;
        const row = d.tapRowBytes();
        var start = start_;
        var n = n_;
        var src = taps_;
        if (start < ctx.end) ctx.end = start;
        if (start > ctx.end) ctx.begin = start;
        if (n > c.window) {
            const skip = n - c.window;
            start += skip;
            src += skip * row;
            n = c.window;
            ctx.begin = start;
        }
        ctx.end = start;
        const e = d.e;
        while (n > 0) {
            const rows = @min(n, max_rows);
            try d.proj(d.w.fc, src, d.th, rows);
            try d.norm(d.th, d.w.hidden_norm, d.thn, rows);
            for (d.w.layers, 0..) |l, i| {
                try d.projAgain(l.k, d.thn, d.kabs, rows);
                try d.normRope(d.kabs, l.k_norm, d.kn, rows, c.kv_heads, start);
                try d.projAgain(l.v, d.thn, d.vabs, rows);
                try e.run(d.fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn, d.vabs, d.ringK(ctx, i), d.ringV(ctx, i), @as(c_int, @intCast(c.kvWidth())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.ringRows())) });
            }
            start += rows;
            src += rows * row;
            n -= rows;
            ctx.end = start;
        }
        if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
    }

    /// Target rows [start, start + n) from a sequence's tap ring [ring_rows, taps, hidden] bf16 (row p at
    /// p % ring_rows) into the context, the window's last rows only.
    pub fn absorbRing(d: *Draft, ctx: *Ctx, ring: u64, ring_rows: usize, start_: u64, n_: usize) !void {
        var start = start_;
        var n = n_;
        if (n > d.c.window) {
            start += n - d.c.window;
            n = d.c.window;
        }
        const prompt = n > 16;
        const width = d.c.n_taps * d.c.hidden;
        while (n > 0) {
            const rows = @min(n, max_rows);
            try d.e.run(d.fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ring, @as(c_int, @intCast(ring_rows)), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(width)), d.taps_in });
            try d.absorb(ctx, d.taps_in, start, rows, prompt);
            start += rows;
            n -= rows;
        }
    }

    /// One stream's target rows to absorb from its tap ring (absorbMany).
    pub const Absorb = struct { ctx: *Ctx, ring: u64, start: u64, n: usize };

    /// Several streams' tap rows in shared projections: each stream's rows gathered back to back, then the fc
    /// projection, the norm and every layer's key and value projections once over all of them, each stream's
    /// rotary norm and ring store on its own rows (a prompt's long runs, and what does not fit, go alone).
    pub fn absorbMany(d: *Draft, items: []const Absorb, ring_rows: usize) !void {
        const c = d.c;
        const row = d.tapRowBytes();
        const width = c.n_taps * c.hidden;
        const kvw = c.kvWidth();
        var batch: [64]Absorb = undefined;
        var nb: usize = 0;
        var total: usize = 0;
        for (items) |it0| {
            var it = it0;
            if (it.n > c.window) {
                it.start += it.n - c.window;
                it.n = c.window;
            }
            if (it.n == 0) continue;
            if (it.n > 16 or total + it.n > max_rows or nb == batch.len) {
                try d.absorbRing(it.ctx, it.ring, ring_rows, it.start, it.n);
                continue;
            }
            batch[nb] = it;
            nb += 1;
            total += it.n;
        }
        if (nb == 0) return;
        if (nb == 1) return d.absorbRing(batch[0].ctx, batch[0].ring, ring_rows, batch[0].start, batch[0].n);
        const previous = d.projection_role;
        defer d.projection_role = previous;
        d.projection_role = .live;
        const e = d.e;
        var off: usize = 0;
        for (batch[0..nb]) |it| {
            try e.run(d.fns.gather_taps, .{ it.n, 1, 1 }, .{ 256, 1, 1 }, .{ it.ring, @as(c_int, @intCast(ring_rows)), @as(c_longlong, @intCast(it.start)), @as(c_int, @intCast(width)), d.taps_in + off * row });
            off += it.n;
            const ctx = it.ctx;
            if (it.start < ctx.end) ctx.end = it.start;
            if (it.start > ctx.end) ctx.begin = it.start;
        }
        try d.proj(d.w.fc, d.taps_in, d.th, total);
        try d.norm(d.th, d.w.hidden_norm, d.thn, total);
        for (d.w.layers, 0..) |l, i| {
            try d.projAgain(l.k, d.thn, d.kabs, total);
            try d.projAgain(l.v, d.thn, d.vabs, total);
            off = 0;
            for (batch[0..nb]) |it| {
                const at = off * kvw * 2;
                try d.normRope(d.kabs + at, l.k_norm, d.kn + at, it.n, c.kv_heads, it.start);
                try e.run(d.fns.ring_put, .{ it.n, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn + at, d.vabs + at, d.ringK(it.ctx, i), d.ringV(it.ctx, i), @as(c_int, @intCast(kvw)), @as(c_longlong, @intCast(it.start)), @as(c_int, @intCast(c.ringRows())) });
                off += it.n;
            }
        }
        for (batch[0..nb]) |it| {
            const ctx = it.ctx;
            ctx.end = it.start + it.n;
            if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
        }
    }

    /// One block pass for every item (its context's end must be its position - 1): each context then holds its drafts.
    pub fn hold(d: *Draft, items: []const Hold, contexts: []const *Ctx) !void {
        if (items.len == 0) return;
        if (items.len > d.frame_streams) return error.BadDraftFrame;
        for (items, contexts) |it, ctx| frame.scan(ctx, it.tokens);
        try frame.forward(d, items, contexts);
        if (d.log > d.holds) for (items, contexts) |it, ctx| std.log.info("dflash2 hold {d}: pos {d} pending {d} drafts {any}", .{ d.holds, it.position, it.pending, ctx.held[0..ctx.held_n] });
        d.holds += 1;
    }

    fn candidates(d: *const Draft, rows: usize) tree.Candidates {
        const base = d.selection_base;
        return .{ .words = d.staged.bytes[words_at + base * lp_words * 4 ..][0 .. rows * lp_words * 4], .projected = d.staged.bytes[hp_at + base * d.c.rank * 2 ..][0 .. rows * d.c.rank * 2], .topk = d.c.topk, .rank = d.c.rank, .predecessor = d.w.predecessor, .successor = d.w.successor, .sampling = d.drawn.sampling, .position = d.drawn.position };
    }

    /// A hold's drafts: the chain through its rows' candidates (and each one's chance under the one before).
    pub fn selectHold(d: *Draft, ctx: *Ctx, it: Hold) !void {
        const rows = d.c.block - 1;
        ctx.grown = 0;
        // the target's keyed draws are this engine's, not the selector's noise: candidates rank by their scores
        d.drawn = .{};
        const n = @min(it.depth, rows);
        const cand = d.candidates(rows);
        tree.chain(cand, it.pending, ctx.held[0..rows]);
        tree.chainChances(cand, it.pending, ctx.held[0..n], ctx.probs[0..n]);
        ctx.held_n = n;
    }
};

test {
    _ = tree;
    _ = config;
}
