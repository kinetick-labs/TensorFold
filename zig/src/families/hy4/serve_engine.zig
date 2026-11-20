//! Hy4-flash's engine for the native server (the `native_engines` of tensorfold-hy4): the pack loaded into one
//! device arena, the forward of gpu_model.zig / gpu_vision.zig on an engine thread. Up to `lanes` requests at once,
//! each with its own caches: every step runs the decoding lanes' next rows and a piece of the prompts still filling
//! side by side in one forward call (HY4_CHUNK rows at most), then samples each finished row on the host from the LM
//! head's logits. Pictures go through the tower before their prompt and replace their placeholder rows (one lane at
//! a time holds the picture buffer); Engram rows are read from the sidecar for every Engram layer at once, by a pool
//! of threads. Caches are bf16 by default (HY4_CACHE_DT=f32|bf16|fp8); lanes are what the memory left after the
//! weights holds, at most --parallel.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("cuda.zig");
const config = hy4.config;
const load_plan = hy4.load_plan;
const device_layout = hy4.device_layout;
const pack_read = hy4.pack_read;
const engram_hash = @import("engram_hash.zig");
const image_grid = @import("image_grid.zig");
const preprocess = @import("preprocess.zig");
const bind = @import("bind.zig");
const gm = @import("gpu_model.zig");
const gv = @import("gpu_vision.zig");
const fwd = @import("fwd.zig");
const sampling = @import("lanes").sampling;
const moe_grouped = @import("moe_grouped.zig");
const engram_batch = @import("engram_batch.zig");
const capacity = @import("capacity.zig");

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = &.{.{ .model_type = "hy_v4_flash", .formats = &.{"exl3"} }};

const forward_image: []const u8 = @embedFile("hy4_forward_fatbin");

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "hy_v4_flash")) {
        problem.* = try std.fmt.allocPrint(a, "this build serves hy_v4_flash checkpoints only, not {s}", .{o.model_type});
        return null;
    }
    const r = Runner.open(gpa, io, o) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the Hy4-flash CUDA engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = .{ .ctx = r, .vtable = &vtable }, .close = Runner.closeCtx, .ctx = r, .halt = Runner.haltCtx };
}

const vtable: api.Engine.VTable = .{ .info = Runner.infoCtx, .submit = Runner.submitCtx, .cancel = Runner.cancelCtx, .status = Runner.statusCtx, .memory = Runner.memoryCtx };

const Job = struct { id: api.Id, req: *const api.Request, sink: api.Sink, cancelled: bool = false };

fn envU32(name: [*:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

/// One request's state: its caches, positions and tokens.
const Lane = struct {
    job: ?*Job = null,
    caches: []gm.Cache,
    pos_dev: u64, // i32 [3, ctx]
    pos: [3][]i32,
    tokens: std.ArrayList(u32) = .empty,
    n_prompt: u32 = 0,
    filled: u32 = 0, // rows in the cache
    next_pos: i32 = 0,
    made: u32 = 0,
    spans: std.ArrayList(gm.ImageSpan) = .empty, // the prompt's picture rows in the image buffer
    holds_images: bool = false,
    prefilled_sent: bool = false,

    fn pending(l: *const Lane) u32 {
        return @intCast(l.tokens.items.len - l.filled);
    }
};

/// One lane's rows in the step being built.
const Part = struct { lane: u32, p0: u32, rows: u32, row: u32 };

pub const Runner = struct {
    gpa: Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    d: cuda.Driver,
    ctx: cuda.Context,
    k: fwd.Kernels,
    dense: cuda.exl3.Dense,
    experts: ?cuda.exl3_experts.Experts = null,
    grouped_buf: ?cuda.DeviceBuffer = null,
    s: cuda.Stream,
    c: config.Config,
    dev: hy4.Device,
    m: bind.Model,
    work: gm.Work,
    lanes: []Lane,
    cache_buf: cuda.DeviceBuffer,
    cache_dt: gm.CacheDt,
    chunk: u32,
    vwork: ?gv.Work = null,
    img_buf: ?cuda.DeviceBuffer = null,
    img_tokens: u32 = 0,
    img_owner: ?u32 = null,
    engram: []?bind.EngramHost,
    engram_slot: []u32, // layer -> its slot in the work's Engram rows
    engram_layers: []u32, // slot -> layer
    engram_threads: u32,
    reqs: std.ArrayList(engram_batch.Req) = .empty,
    side_files: std.StringHashMapUnmanaged(std.Io.File) = .empty,
    e4m3: [256]f32,
    ctx_len: u32,
    info_: api.Info,
    // a step's host buffers
    emb_host: []f32,
    mask_host: []u8,
    ids_host: []i32,
    logits: []f32, // [lanes, vocab]
    values: []f64,
    cand: []u64,
    parts: []Part,
    segs: []gm.Seg,
    spans_step: std.ArrayList(gm.ImageSpan) = .empty,
    rr: u32 = 0,
    // the queue
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    active: u32 = 0,
    stopping: bool = false,
    thread: ?std.Thread = null,
    healthy: bool = true,

    fn open(gpa: Allocator, io: std.Io, o: api.Open) !*Runner {
        const r = try gpa.create(Runner);
        errdefer gpa.destroy(r);
        r.* = undefined;
        r.gpa = gpa;
        r.io = io;
        r.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer r.arena.deinit();
        const a = r.arena.allocator();
        r.mutex = .init;
        r.cond = .init;
        r.queue = .empty;
        r.active = 0;
        r.stopping = false;
        r.thread = null;
        r.healthy = true;
        r.side_files = .empty;
        r.vwork = null;
        r.img_buf = null;
        r.img_tokens = 0;
        r.img_owner = null;
        r.experts = null;
        r.grouped_buf = null;
        r.reqs = .empty;
        r.spans_step = .empty;
        r.rr = 0;
        r.e4m3 = bind.e4m3Table();

        r.c = try config.Config.read(a, io, o.dir);
        const want_ctx: u32 = @intCast(@min(@max(o.context orelse 32768, 1024), 1 << 20));
        r.ctx_len = std.mem.alignForward(u32, want_ctx, 64);
        r.chunk = @max(16, @min(envU32("HY4_CHUNK", 256), 1024));
        r.cache_dt = if (std.c.getenv("HY4_CACHE_DT")) |v| (fwd.CacheDt.parse(std.mem.span(v)) orelse return error.Hy4CacheDt) else .bf16;
        var main = try pack_read.Pack.open(a, io, o.dir, "model.safetensors.index.json");
        var side = try pack_read.Pack.open(a, io, o.dir, "engram.safetensors.index.json");
        var problem: load_plan.Problem = .{};
        const plan = load_plan.build(a, &r.c, load_plan.Source.ofPack(&main), load_plan.Source.ofPack(&side), .{ .vision = o.vision != null }, &problem) catch |e| {
            std.log.err("hy4 plan refused at {s}: {s}", .{ problem.name, problem.what });
            return e;
        };
        const layout = try device_layout.build(a, &r.c, &plan);

        r.d = try cuda.Driver.open();
        errdefer r.d.close();
        r.ctx = try cuda.Context.init(&r.d, 0);
        errdefer r.ctx.deinit();
        const t = &r.c.text;
        // one sequence's caches; the weights load only when at least one fits beside the reserve
        const per_lane = capacity.seqBytes(t, r.ctx_len, r.chunk, r.cache_dt);
        const reserve_mib: u64 = envU32("HY4_RESERVE_MIB", 4096);
        const reserve: u64 = reserve_mib << 20;
        {
            var ck = try core.Checkpoint.openModel(gpa, io, o.dir);
            defer ck.close();
            const t0 = std.Io.Clock.awake.now(io);
            r.dev = try hy4.load(&r.d, &ck, &plan, layout, per_lane + reserve, 2 << 30, progress);
            const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
            std.log.info("hy4: {d} MiB of weights loaded in {d:.1} s", .{ hy4.uploaded(&layout) >> 20, secs });
        }
        errdefer r.dev.deinit();
        r.k = try fwd.Kernels.load(&r.d, forward_image);
        r.dense = try cuda.exl3.Dense.load(&r.d);
        r.s = try cuda.Stream.init(&r.d, true);
        r.m = try bind.model(a, &r.d, &r.c, &r.dev, &plan, &main, o.vision != null);
        // the routed experts grouped on the device unless HY4_MOE_GROUPED=0
        if (envU32("HY4_MOE_GROUPED", 1) != 0) {
            if (cuda.exl3_experts.Experts.load(&r.d)) |e| {
                r.experts = e;
                r.grouped_buf = try moe_grouped.bindAll(gpa, &r.d, r.s, r.m.layers);
            } else |e| std.log.warn("hy4: the grouped expert kernels are unavailable ({s}); experts run one by one", .{@errorName(e)});
        }

        const want_lanes: u32 = @max(1, @min(o.lanes, 64));
        var lim: gm.Limits = .{ .chunk = r.chunk, .ctx = r.ctx_len, .hidden = t.hidden, .hc = t.hc_mult, .heads = @max(t.indexed.heads, t.sliding.heads), .qk = @max(t.indexed.qkDim(), t.sliding.qkDim()), .v = @max(t.indexed.v, t.sliding.v), .kv_lora = t.kv_lora, .rope = @max(t.indexed.qk_rope, t.sliding.qk_rope), .q_lora = t.q_lora, .inter = t.dense_width, .experts = t.experts, .top = t.topk, .latent = t.latent, .width = t.expert_width, .index_heads = t.index_heads, .index_dim = t.index_dim, .kpool = t.index_pool, .engram_width = t.hashHeads() * t.engramHeadDim(), .vocab = t.vocab, .recall = t.index_topk / @max(t.index_pool, 1), .lanes = want_lanes };
        if (lim.experts == 0) lim.top = 0;
        // the grouped experts' scratch, for calls of up to HY4_MOE_GROUPED_ROWS rows (default: the chunk)
        for (r.m.layers) |L| if (L.mlp == .moe) if (L.mlp.moe.grouped) |gl| {
            lim.grouped_rows = @min(envU32("HY4_MOE_GROUPED_ROWS", r.chunk), r.chunk);
            if (lim.grouped_rows > 0) lim.grouped_scratch = try cuda.exl3_experts.scratchBytes(gl, lim.grouped_rows, t.topk);
            break;
        };
        // Engram: hashes over the checkpoint's normalized vocabulary, rows from the sidecar
        r.engram = try a.alloc(?bind.EngramHost, t.layers);
        @memset(r.engram, null);
        r.engram_slot = try a.alloc(u32, t.layers);
        @memset(r.engram_slot, 0);
        const tables = try load_plan.engramTables(a, &r.c, &side);
        r.engram_layers = try a.alloc(u32, tables.len);
        for (tables, 0..) |tb, slot| {
            const e = t.engramAt(tb.layer) orelse return error.Hy4Engram;
            const map_bytes = try main.raw(a, try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.lookup_table", .{tb.layer}));
            const map = try a.alloc(i32, map_bytes.len / 4);
            for (map, 0..) |*v, i| v.* = std.mem.readInt(i32, map_bytes[4 * i ..][0..4], .little);
            const mult = try engram_hash.releaseMultipliers(e.id, t.engram_seed, map);
            const hasher = try engram_hash.Hasher.of(t, e, map, mult);
            const offsets = try a.alloc(u64, hasher.sizes.len);
            engram_hash.headOffsets(hasher.sizes, offsets);
            const gop = try r.side_files.getOrPut(a, tb.shard);
            if (!gop.found_existing) gop.value_ptr.* = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.dir, tb.shard }), .{});
            if (tb.dim != t.engramHeadDim() or tb.dim > engram_batch.max_dim) return error.Hy4Engram;
            r.engram[tb.layer] = .{ .layer = tb.layer, .hasher = hasher, .offsets = offsets, .file = gop.value_ptr.*, .values_at = tb.values_at, .scales_at = tb.scales_at, .dim = tb.dim, .rows = tb.rows };
            r.engram_slot[tb.layer] = @intCast(slot);
            r.engram_layers[slot] = tb.layer;
        }
        r.engram_threads = engram_batch.threadsFromEnv();
        lim.engram_layers = @intCast(@max(tables.len, 1));
        r.work = try gm.Work.init(a, r.ctx_(), lim);
        r.work.check_left = envU32("HY4_MOE_CHECK", 0);
        // vision
        if (r.m.tower) |*tw| if (o.vision) |vo| {
            const max_side = r.c.vision.?.max_image / tw.patch;
            r.vwork = try gv.Work.init(&r.d, tw, max_side * max_side);
            r.img_tokens = vo.image_tokens;
            r.img_buf = try cuda.DeviceBuffer.alloc(&r.d, @as(u64, vo.image_tokens) * t.hidden * 4);
        };
        // lanes: what the device holds now beside the reserve
        var free: usize = 0;
        var total: usize = 0;
        try r.d.check(r.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        const fits = capacity.fit(free, reserve, per_lane, want_lanes);
        for ([_]u32{ 1 << 18, 1 << 19, 1 << 20 }) |cx| {
            std.log.info("hy4 capacity: {d}k tokens a lane: f32 {d}, bf16 {d}, fp8 {d} lanes in {d} MiB free beside {d} MiB kept", .{ cx >> 10, capacity.fit(free, reserve, capacity.seqBytes(t, cx, r.chunk, .f32), 64), capacity.fit(free, reserve, capacity.seqBytes(t, cx, r.chunk, .bf16), 64), capacity.fit(free, reserve, capacity.seqBytes(t, cx, r.chunk, .fp8), 64), free >> 20, reserve_mib });
        }
        if (fits == 0) {
            std.log.err("hy4: budget fits no stream: one {d}-token lane needs {d} MiB of {s} caches, {d} MiB free beside {d} MiB kept", .{ r.ctx_len, per_lane >> 20, @tagName(r.cache_dt), free >> 20, reserve_mib });
            return error.Hy4BudgetFitsNoStream;
        }
        if (o.lanes_fixed and fits < o.lanes) {
            std.log.err("hy4: --parallel {d} asked, the memory holds {d} lanes of {d} tokens ({s} caches)", .{ o.lanes, fits, r.ctx_len, @tagName(r.cache_dt) });
            return error.Hy4BudgetFitsFewerStreams;
        }
        r.cache_buf = try cuda.DeviceBuffer.alloc(&r.d, per_lane * fits);
        try r.cache_buf.fill8(0, null);
        // the legacy-stream memset does not order with the engine's non-blocking stream: finish it now
        try r.d.check(r.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        r.lanes = try a.alloc(Lane, fits);
        var at: u64 = r.cache_buf.ptr;
        for (r.lanes) |*ln| {
            ln.* = .{ .caches = try a.alloc(gm.Cache, t.layers), .pos_dev = at, .pos = undefined };
            at += std.mem.alignForward(u64, 3 * @as(u64, r.ctx_len) * 4, 256);
            for (ln.caches, 0..) |*cc, i| {
                const dm = bind.dims(t, @intCast(i));
                cc.* = gm.cacheAt(&dm, r.ctx_len, r.chunk, r.cache_dt, at);
                at += gm.cacheBytes(&dm, r.ctx_len, r.chunk, r.cache_dt);
            }
            for (&ln.pos) |*p| p.* = try a.alloc(i32, r.ctx_len);
        }
        if (at - r.cache_buf.ptr != per_lane * fits) return error.Hy4Shape;
        std.log.info("hy4: {d} lanes of {d} tokens, {s} caches, {d} MiB each; chunk {d}", .{ fits, r.ctx_len, @tagName(r.cache_dt), per_lane >> 20, r.chunk });
        // host buffers
        r.emb_host = try a.alloc(f32, @as(usize, lim.engram_layers) * r.chunk * lim.engram_width);
        r.mask_host = try a.alloc(u8, r.chunk);
        r.ids_host = try a.alloc(i32, r.chunk);
        r.logits = try a.alloc(f32, @as(usize, fits) * t.vocab);
        r.values = try a.alloc(f64, t.vocab);
        r.cand = try a.alloc(u64, t.vocab);
        for (r.cand, 0..) |*x, i| x.* = i;
        r.parts = try a.alloc(Part, fits);
        r.segs = try a.alloc(gm.Seg, fits);
        r.info_ = .{ .name = "hy4-cuda", .lanes = fits, .context_window = r.ctx_len, .thinking_budget = false, .reasoning = true };
        if (o.vision) |vo| if (r.m.tower != null) {
            r.info_.vision = .{ .ctx = r, .prepare = prepareCtx, .image_token = 120120, .markers = &.{ 120118, 120120, 120119 }, .image_tokens = vo.image_tokens, .max_images = vo.max_images };
        };
        try r.d.check(r.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.log.info("hy4: ready, work {d} MiB, {d} MiB of the device free", .{ r.work.bytes() >> 20, free >> 20 });
        r.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{r});
        return r;
    }

    fn progress(done: u64, total: u64) void {
        std.log.info("hy4 load: {d} / {d} MiB", .{ done >> 20, total >> 20 });
    }

    fn ctx_(r: *Runner) gm.Ctx {
        return .{ .d = &r.d, .k = &r.k, .dense = &r.dense, .s = r.s, .experts = if (r.experts) |*e| e else null };
    }

    fn of(ctx: *anyopaque) *Runner {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoCtx(ctx: *anyopaque) api.Info {
        return of(ctx).info_;
    }

    fn submitCtx(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r = of(ctx);
        if (req.promptOnly()) return error.InvalidRequest;
        const job = r.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .req = req, .sink = sink };
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.stopping) {
            r.gpa.destroy(job);
            return error.Closed;
        }
        r.queue.append(r.gpa, job) catch {
            r.gpa.destroy(job);
            return error.Busy;
        };
        r.cond.signal(r.io);
    }

    fn cancelCtx(ctx: *anyopaque, id: api.Id) void {
        const r = of(ctx);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        for (r.lanes) |*ln| if (ln.job) |j| if (j.id == id) @atomicStore(bool, &j.cancelled, true, .release);
        for (r.queue.items) |j| if (j.id == id) @atomicStore(bool, &j.cancelled, true, .release);
    }

    fn statusCtx(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const r = of(ctx);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        out.* = .{ .healthy = r.healthy, .running = r.active, .waiting = @intCast(r.queue.items.len) };
    }

    fn memoryCtx(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.memory.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak, .pinned = u.host };
    }

    fn haltCtx(ctx: *anyopaque, _: []const u8) void {
        const r = of(ctx);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        r.stopping = true;
        for (r.lanes) |*ln| if (ln.job) |j| @atomicStore(bool, &j.cancelled, true, .release);
        for (r.queue.items) |j| @atomicStore(bool, &j.cancelled, true, .release);
        r.cond.broadcast(r.io);
    }

    fn closeCtx(ctx: *anyopaque) void {
        const r = of(ctx);
        haltCtx(ctx, "");
        if (r.thread) |t| t.join();
        r.ctx.makeCurrent() catch {};
        r.work.deinit();
        r.cache_buf.free();
        if (r.vwork) |*v| v.deinit();
        if (r.img_buf) |*b| b.free();
        if (r.grouped_buf) |*b| b.free();
        if (r.experts) |*e| e.unload();
        r.m.extra.free();
        r.dense.unload();
        r.k.unload();
        r.s.deinit();
        r.dev.deinit();
        var it = r.side_files.valueIterator();
        while (it.next()) |f| f.close(r.io);
        r.ctx.deinit();
        r.d.close();
        r.queue.deinit(r.gpa);
        for (r.lanes) |*ln| {
            ln.tokens.deinit(r.gpa);
            ln.spans.deinit(r.gpa);
        }
        r.reqs.deinit(r.gpa);
        r.spans_step.deinit(r.gpa);
        r.arena.deinit();
        r.gpa.destroy(r);
    }

    fn run(r: *Runner) void {
        r.ctx.makeCurrent() catch |e| {
            std.log.err("hy4: the engine thread cannot bind the CUDA context ({s})", .{@errorName(e)});
            r.healthy = false;
            return;
        };
        while (true) {
            // admit queued requests into free lanes (in order; a picture waits for the picture buffer)
            r.mutex.lockUncancelable(r.io);
            while (r.active == 0 and r.queue.items.len == 0 and !r.stopping) r.cond.waitUncancelable(r.io, &r.mutex);
            if (r.stopping and r.active == 0 and r.queue.items.len == 0) {
                r.mutex.unlock(r.io);
                return;
            }
            var admitted: [64]struct { job: *Job, lane: u32 } = undefined;
            var na: usize = 0;
            while (r.queue.items.len > 0) {
                const job = r.queue.items[0];
                const free_lane = for (r.lanes, 0..) |*ln, i| (if (ln.job == null) break @as(u32, @intCast(i))) else null;
                const lane = free_lane orelse break;
                if (job.req.images.len > 0 and r.img_owner != null) break;
                _ = r.queue.orderedRemove(0);
                r.lanes[lane].job = job;
                if (job.req.images.len > 0) r.img_owner = lane;
                r.active += 1;
                admitted[na] = .{ .job = job, .lane = lane };
                na += 1;
            }
            r.mutex.unlock(r.io);
            for (admitted[0..na]) |ad| r.begin(ad.lane) catch |e| {
                var buf: [256]u8 = undefined;
                r.finish(ad.lane, .failed, std.fmt.bufPrint(&buf, "hy4 forward failed: {s}", .{@errorName(e)}) catch "hy4 forward failed");
            };
            if (r.active == 0) continue;
            r.step() catch |e| {
                // a failed step leaves every lane's cache in doubt: each running request ends
                var buf: [256]u8 = undefined;
                const msg = std.fmt.bufPrint(&buf, "hy4 forward failed: {s}", .{@errorName(e)}) catch "hy4 forward failed";
                std.log.err("hy4: step: {s}", .{msg});
                for (r.lanes, 0..) |ln, i| if (ln.job != null) r.finish(@intCast(i), .failed, msg);
            };
        }
    }

    fn cancelled(j: *Job) bool {
        return @atomicLoad(bool, &j.cancelled, .acquire);
    }

    /// The request's lane ends: its finished event, the lane free again.
    fn finish(r: *Runner, lane: u32, reason: api.Reason, msg: []const u8) void {
        const ln = &r.lanes[lane];
        const job = ln.job orelse return;
        if (reason == .failed) std.log.err("hy4: request {d}: {s}", .{ job.id, msg });
        job.sink.event(job.sink.ctx, job.id, &.{ .finished = .{ .reason = reason, .message = msg } });
        r.mutex.lockUncancelable(r.io);
        ln.job = null;
        if (r.img_owner == lane) r.img_owner = null;
        r.active -= 1;
        r.mutex.unlock(r.io);
        ln.holds_images = false;
        ln.spans.clearRetainingCapacity();
        r.gpa.destroy(job);
    }

    /// A newly admitted request: its tokens, positions and pictures; refusals end it at once.
    fn begin(r: *Runner, lane: u32) !void {
        const ln = &r.lanes[lane];
        const job = ln.job.?;
        const req = job.req;
        const n: u32 = @intCast(req.prompt.len);
        if (n == 0) return r.finish(lane, .failed, "empty prompt");
        if (n + 1 >= r.ctx_len) return r.finish(lane, .length, "the prompt fills the context");
        ln.tokens.clearRetainingCapacity();
        try ln.tokens.appendSlice(r.gpa, req.prompt);
        ln.n_prompt = n;
        ln.filled = 0;
        ln.made = 0;
        ln.prefilled_sent = false;
        ln.next_pos = try r.positions(ln, req);
        ln.spans.clearRetainingCapacity();
        ln.holds_images = false;
        if (req.images.len > 0) {
            const tw = &(r.m.tower orelse return r.finish(lane, .failed, "this server was started without --vision"));
            var used: u32 = 0;
            var scratch = std.heap.ArenaAllocator.init(r.gpa);
            defer scratch.deinit();
            for (req.images) |im| {
                if (used + im.tokens > r.img_tokens) return r.finish(lane, .failed, "the request's pictures exceed --vision-image-tokens");
                const dst = r.img_buf.?.ptr + @as(u64, used) * r.c.text.hidden * 4;
                try gv.forward(scratch.allocator(), r.ctx_(), tw, &r.vwork.?, im.pixels, .{ .h = im.gh, .w = im.gw }, dst);
                try ln.spans.append(r.gpa, .{ .src = dst, .row = im.at, .rows = im.tokens });
                used += im.tokens;
            }
            ln.holds_images = true;
        }
    }

    /// One forward call: every decoding lane's next row, then prompt pieces (round robin) up to the chunk; each lane
    /// whose rows reached its last token samples the next one.
    fn step(r: *Runner) !void {
        const tx = &r.c.text;
        const c = r.ctx_();
        const w = &r.work;
        // cancelled requests end before the step
        for (r.lanes, 0..) |ln, i| if (ln.job) |j| if (cancelled(j)) r.finish(@intCast(i), .cancelled, "");
        var np: usize = 0;
        var used: u32 = 0;
        for (r.lanes, 0..) |*ln, i| if (ln.job != null and ln.pending() == 1 and used < r.chunk) {
            r.parts[np] = .{ .lane = @intCast(i), .p0 = ln.filled, .rows = 1, .row = used };
            np += 1;
            used += 1;
        };
        const L: u32 = @intCast(r.lanes.len);
        for (0..L) |k| {
            const i: u32 = @intCast((r.rr + k) % L);
            const ln = &r.lanes[i];
            if (ln.job == null or ln.pending() <= 1 or used >= r.chunk) continue;
            const rows = @min(ln.pending(), r.chunk - used);
            r.parts[np] = .{ .lane = i, .p0 = ln.filled, .rows = rows, .row = used };
            np += 1;
            used += rows;
        }
        r.rr = (r.rr + 1) % L;
        if (np == 0) return;
        const parts = r.parts[0..np];
        const t = used;
        // token ids, positions and picture rows
        r.spans_step.clearRetainingCapacity();
        for (parts) |p| {
            const ln = &r.lanes[p.lane];
            for (0..p.rows) |q| r.ids_host[p.row + q] = @intCast(ln.tokens.items[p.p0 + q]);
            for (0..3) |pl| try c.upload(ln.pos_dev + 4 * (@as(u64, pl) * r.ctx_len + p.p0), std.mem.sliceAsBytes(ln.pos[pl][p.p0..][0..p.rows]));
            for (ln.spans.items) |sp| {
                const lo = @max(sp.row, p.p0);
                const hi = @min(sp.row + sp.rows, p.p0 + p.rows);
                if (lo >= hi) continue;
                try r.spans_step.append(r.gpa, .{ .src = sp.src + @as(u64, lo - sp.row) * tx.hidden * 4, .row = p.row + (lo - p.p0), .rows = hi - lo });
            }
        }
        try c.upload(w.tokens, std.mem.sliceAsBytes(r.ids_host[0..t]));
        try r.engramRows(parts);
        try gm.embedWithImage(w, &r.m.head, t, tx.hc_mult, tx.hidden, r.spans_step.items);
        const segs = r.segs[0..np];
        for (r.m.layers, 0..) |*Ly, li| {
            for (parts, segs) |p, *sg| sg.* = .{ .cache = r.lanes[p.lane].caches[li], .pos = r.lanes[p.lane].pos_dev, .p0 = p.p0, .row = p.row, .rows = p.rows };
            try gm.layer(w, Ly, segs, w.embAt(r.engram_slot[li]));
        }
        // the rows that end a lane's known tokens: their logits in one pass
        var last: [64]u32 = undefined;
        var who: [64]u32 = undefined;
        var nl: usize = 0;
        for (parts) |p| {
            const ln = &r.lanes[p.lane];
            ln.filled += p.rows;
            if (ln.filled == ln.tokens.items.len) {
                last[nl] = p.row + p.rows - 1;
                who[nl] = p.lane;
                nl += 1;
            }
            if (ln.holds_images and ln.filled >= ln.n_prompt) {
                ln.holds_images = false;
                r.mutex.lockUncancelable(r.io);
                if (r.img_owner == p.lane) r.img_owner = null;
                r.mutex.unlock(r.io);
            }
        }
        if (nl == 0) return;
        try gm.logitsOf(w, &r.m.head, last[0..nl], tx.hc_mult, tx.hidden, tx.eps, tx.hc_eps);
        try c.download(std.mem.sliceAsBytes(r.logits[0 .. nl * tx.vocab]), w.head);
        for (who[0..nl], 0..) |lane, q| try r.advance(lane, r.logits[q * tx.vocab ..][0..tx.vocab]);
    }

    /// A lane's next token from its logits row: delivered, then the stop checks.
    fn advance(r: *Runner, lane: u32, row: []const f32) !void {
        const ln = &r.lanes[lane];
        const job = ln.job.?;
        const req = job.req;
        if (!ln.prefilled_sent) {
            job.sink.event(job.sink.ctx, job.id, &.{ .prefilled = 0 });
            ln.prefilled_sent = true;
        }
        const tok = try r.sample(req, row, ln.tokens.items.len);
        try ln.tokens.append(r.gpa, tok);
        ln.made += 1;
        const one = [1]u32{tok};
        job.sink.event(job.sink.ctx, job.id, &.{ .tokens = &one });
        if (std.mem.indexOfScalar(u32, req.eos, tok) != null) return r.finish(lane, .stop, "");
        if (req.stop) |st| if (st.check(st.ctx, ln.tokens.items[ln.n_prompt..])) return r.finish(lane, .stop, "");
        if (ln.made >= req.max_tokens) return r.finish(lane, .length, "");
        if (cancelled(job)) return r.finish(lane, .cancelled, "");
        const at: u32 = @intCast(ln.tokens.items.len - 1);
        if (at + 1 >= r.ctx_len) return r.finish(lane, .length, "");
        for (0..3) |p| ln.pos[p][at] = ln.next_pos;
        ln.next_pos += 1;
    }

    /// The prompt's 3D positions (pictures as their grids) into the lane, the next text position returned.
    fn positions(r: *Runner, ln: *Lane, req: *const api.Request) !i32 {
        const n = req.prompt.len;
        var cur: i32 = 0;
        var i: usize = 0;
        var img: usize = 0;
        const merge: u32 = if (r.c.vision) |v| v.merge else 2;
        while (i < n) {
            if (img < req.images.len and req.images[img].at == i) {
                const im = req.images[img];
                const g: image_grid.Grid = .{ .h = im.gh, .w = im.gw };
                if (g.tokens(merge) != im.tokens or i + im.tokens > n) return error.Hy4ImageSpan;
                const rows = g.h / merge;
                const cols = g.w / merge + 1;
                var k: usize = 0;
                for (0..rows) |y| for (0..cols) |x| {
                    ln.pos[0][i + k] = cur;
                    ln.pos[1][i + k] = cur + @as(i32, @intCast(y));
                    ln.pos[2][i + k] = cur + @as(i32, @intCast(x));
                    k += 1;
                };
                cur += @intCast(@max(rows, cols));
                i += im.tokens;
                img += 1;
                continue;
            }
            for (0..3) |p| ln.pos[p][i] = cur;
            cur += 1;
            i += 1;
        }
        return cur;
    }

    /// Every Engram layer's rows for the step's tokens (hashed here, read by the thread pool), the owner mask, one
    /// upload each.
    fn engramRows(r: *Runner, parts: []const Part) !void {
        const tx = &r.c.text;
        if (r.engram_layers.len == 0) return;
        const E: usize = tx.hashHeads() * tx.engramHeadDim();
        var ids: [config.max_hash_heads]u32 = undefined;
        r.reqs.clearRetainingCapacity();
        var t: u32 = 0;
        for (parts) |p| t = @max(t, p.row + p.rows);
        for (parts) |p| {
            const ln = &r.lanes[p.lane];
            const toks = ln.tokens.items;
            for (0..p.rows) |q| {
                const g = p.p0 + q;
                const row = p.row + q;
                const owns = engram_hash.owns(toks[g], 120120, 120124, tx.engram_pad);
                r.mask_host[row] = @intFromBool(owns);
                for (r.engram_layers, 0..) |layer, slot| {
                    const out = r.emb_host[(slot * r.chunk + row) * E ..][0..E];
                    if (!owns) {
                        @memset(out, 0);
                        continue;
                    }
                    const e = &(r.engram[layer] orelse return error.Hy4Engram);
                    const heads = e.hasher.sizes.len;
                    const time: usize = @intCast(@max(ln.pos[0][g], 0));
                    const start = g - @min(time, g);
                    e.hasher.row(toks[0 .. g + 1], start, g, ids[0..heads]);
                    for (0..heads) |h| {
                        const rw = e.offsets[h] + ids[h];
                        if (rw >= e.rows) return error.Hy4EngramRow;
                        try r.reqs.append(r.gpa, .{ .fd = e.file.handle, .values_at = e.values_at, .scales_at = e.scales_at, .row = rw, .dim = @intCast(e.dim), .out = out[h * e.dim ..].ptr });
                    }
                }
            }
        }
        try engram_batch.readAll(r.reqs.items, &r.e4m3, r.engram_threads);
        const c = r.ctx_();
        for (0..r.engram_layers.len) |slot| try c.upload(r.work.embAt(@intCast(slot)), std.mem.sliceAsBytes(r.emb_host[slot * r.chunk * E ..][0 .. t * E]));
        try c.upload(r.work.mask, r.mask_host[0..t]);
        // the host buffers are refilled by the next step: wait until the uploads were read
        try r.s.synchronize();
    }

    fn sample(r: *Runner, req: *const api.Request, row: []const f32, position: u64) !u32 {
        if (req.sampling) |s| if (s.temperature > 0) {
            for (r.values, row) |*v, l| v.* = l;
            return @intCast(try sampling.choose(r.gpa, r.values, r.cand, position, s));
        };
        var best: usize = 0;
        for (row, 0..) |l, i| if (l > row[best]) {
            best = i;
        };
        return @intCast(best);
    }

    fn prepareCtx(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const r = of(ctx);
        var rgb = try core.picture.decode(r.io, a, bytes);
        defer rgb.deinit(a);
        var lim: image_grid.Limits = .{};
        // at most `max_tokens` rows: (h / 32) * (w / 32 + 1) <= max_tokens, so h * w / 1024 below it is enough
        const cap: u64 = @as(u64, max_tokens) * 1024 * 9 / 10;
        if (cap < lim.max_pixels) lim.max_pixels = @max(cap, lim.min_pixels);
        var p = try preprocess.picture(a, rgb.pixels, @intCast(rgb.width), @intCast(rgb.height), lim);
        const tokens = p.grid.tokens(lim.merge);
        if (tokens > max_tokens) {
            p.deinit(a);
            return error.Hy4ImageTooLarge;
        }
        const id = core.picture.identity.withGrid(core.picture.identity.ofBytes(0x4859345649, bytes), .{ p.grid.h, p.grid.w });
        return .{ .pixels = p.patches, .gh = p.grid.h, .gw = p.grid.w, .tokens = tokens, .hash = id, .frame = .{ .width = p.size.width, .height = p.size.height } };
    }
};
