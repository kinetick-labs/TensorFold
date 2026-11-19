//! Hy4-flash's engine for the native server (the `native_engines` of tensorfold-hy4): the pack loaded into one
//! device arena, the forward of gpu_model.zig / gpu_vision.zig, one request at a time on an engine thread (prompt in
//! 256-row chunks, then a token a step), greedy or seeded sampling on the host from the LM head's logits. Pictures
//! go through the tower before the prompt and replace their placeholder rows; Engram rows are read from the sidecar.

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

const chunk_rows = 256;

const Job = struct { id: api.Id, req: *const api.Request, sink: api.Sink, cancelled: bool = false };

pub const Runner = struct {
    gpa: Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    d: cuda.Driver,
    ctx: cuda.Context,
    k: fwd.Kernels,
    dense: cuda.exl3.Dense,
    s: cuda.Stream,
    c: config.Config,
    dev: hy4.Device,
    m: bind.Model,
    work: gm.Work,
    caches: []gm.Cache,
    cache_buf: cuda.DeviceBuffer,
    vwork: ?gv.Work = null,
    img_buf: ?cuda.DeviceBuffer = null,
    img_tokens: u32 = 0,
    engram: []?bind.EngramHost,
    side_files: std.StringHashMapUnmanaged(std.Io.File) = .empty,
    e4m3: [256]f32,
    ctx_len: u32,
    info_: api.Info,
    // the request being served
    tokens: std.ArrayList(u32) = .empty,
    pos: [3][]i32,
    row_buf: []u8,
    emb_host: []f32,
    mask_host: []u8,
    ids_host: []i32,
    logits: []f32,
    values: []f64,
    cand: []u64,
    // the queue
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    current: ?*Job = null,
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
        r.current = null;
        r.stopping = false;
        r.thread = null;
        r.healthy = true;
        r.tokens = .empty;
        r.side_files = .empty;
        r.vwork = null;
        r.img_buf = null;
        r.img_tokens = 0;
        r.e4m3 = bind.e4m3Table();

        r.c = try config.Config.read(a, io, o.dir);
        const want_ctx: u32 = @intCast(@min(@max(o.context orelse 32768, 1024), 1 << 20));
        r.ctx_len = std.mem.alignForward(u32, want_ctx, 64);
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
        // the caches and work buffers this context will need, kept free beside the arena
        var cache_bytes: u64 = 0;
        for (0..r.c.text.layers) |i| {
            const dm = bind.dims(&r.c.text, @intCast(i));
            cache_bytes += gm.cacheBytes(&dm, r.ctx_len);
        }
        const reserve_mib: u64 = if (std.c.getenv("HY4_RESERVE_MIB")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 4096 else 4096;
        {
            var ck = try core.Checkpoint.openModel(gpa, io, o.dir);
            defer ck.close();
            const t0 = std.Io.Clock.awake.now(io);
            r.dev = try hy4.load(&r.d, &ck, &plan, layout, cache_bytes + (reserve_mib << 20), 2 << 30, progress);
            const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
            std.log.info("hy4: {d} MiB of weights loaded in {d:.1} s; caches {d} MiB for {d} tokens", .{ hy4.uploaded(&layout) >> 20, secs, cache_bytes >> 20, r.ctx_len });
        }
        errdefer r.dev.deinit();
        r.k = try fwd.Kernels.load(&r.d, forward_image);
        r.dense = try cuda.exl3.Dense.load(&r.d);
        r.s = try cuda.Stream.init(&r.d, true);
        r.m = try bind.model(a, &r.d, &r.c, &r.dev, &plan, &main, o.vision != null);

        const t = &r.c.text;
        var lim: gm.Limits = .{ .chunk = chunk_rows, .ctx = r.ctx_len, .hidden = t.hidden, .hc = t.hc_mult, .heads = @max(t.indexed.heads, t.sliding.heads), .qk = @max(t.indexed.qkDim(), t.sliding.qkDim()), .v = @max(t.indexed.v, t.sliding.v), .kv_lora = t.kv_lora, .rope = @max(t.indexed.qk_rope, t.sliding.qk_rope), .q_lora = t.q_lora, .inter = t.dense_width, .experts = t.experts, .top = t.topk, .latent = t.latent, .width = t.expert_width, .index_heads = t.index_heads, .index_dim = t.index_dim, .kpool = t.index_pool, .engram_width = t.hashHeads() * t.engramHeadDim(), .vocab = t.vocab };
        if (lim.experts == 0) lim.top = 0;
        r.work = try gm.Work.init(a, r.ctx_(), lim);
        r.cache_buf = try cuda.DeviceBuffer.alloc(&r.d, cache_bytes);
        try r.cache_buf.fill8(0, null);
        r.caches = try a.alloc(gm.Cache, t.layers);
        var at: u64 = r.cache_buf.ptr;
        for (r.caches, 0..) |*cc, i| {
            const dm = bind.dims(t, @intCast(i));
            cc.* = gm.cacheAt(&dm, r.ctx_len, at);
            at += gm.cacheBytes(&dm, r.ctx_len);
        }
        // Engram: hashes over the checkpoint's normalized vocabulary, rows from the sidecar
        r.engram = try a.alloc(?bind.EngramHost, t.layers);
        @memset(r.engram, null);
        const tables = try load_plan.engramTables(a, &r.c, &side);
        for (tables) |tb| {
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
            if (tb.dim != t.engramHeadDim()) return error.Hy4Engram;
            r.engram[tb.layer] = .{ .layer = tb.layer, .hasher = hasher, .offsets = offsets, .file = gop.value_ptr.*, .values_at = tb.values_at, .scales_at = tb.scales_at, .dim = tb.dim, .rows = tb.rows };
        }
        // host buffers
        for (&r.pos) |*p| p.* = try a.alloc(i32, r.ctx_len);
        r.row_buf = try a.alloc(u8, t.engramHeadDim() + 2);
        r.emb_host = try a.alloc(f32, chunk_rows * lim.engram_width);
        r.mask_host = try a.alloc(u8, chunk_rows);
        r.ids_host = try a.alloc(i32, chunk_rows);
        r.logits = try a.alloc(f32, t.vocab);
        r.values = try a.alloc(f64, t.vocab);
        r.cand = try a.alloc(u64, t.vocab);
        for (r.cand, 0..) |*x, i| x.* = i;
        // vision
        if (r.m.tower) |*tw| if (o.vision) |vo| {
            const max_side = r.c.vision.?.max_image / tw.patch;
            r.vwork = try gv.Work.init(&r.d, tw, max_side * max_side);
            r.img_tokens = vo.image_tokens;
            r.img_buf = try cuda.DeviceBuffer.alloc(&r.d, @as(u64, vo.image_tokens) * t.hidden * 4);
        };
        r.info_ = .{ .name = "hy4-cuda", .lanes = 1, .context_window = r.ctx_len, .thinking_budget = false, .reasoning = true };
        if (o.vision) |vo| if (r.m.tower != null) {
            r.info_.vision = .{ .ctx = r, .prepare = prepareCtx, .image_token = 120120, .markers = &.{ 120118, 120120, 120119 }, .image_tokens = vo.image_tokens, .max_images = vo.max_images };
        };
        var free: usize = 0;
        var total: usize = 0;
        try r.d.check(r.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.log.info("hy4: ready, work {d} MiB, {d} MiB of the device free", .{ r.work.bytes() >> 20, free >> 20 });
        r.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{r});
        return r;
    }

    fn progress(done: u64, total: u64) void {
        std.log.info("hy4 load: {d} / {d} MiB", .{ done >> 20, total >> 20 });
    }

    fn ctx_(r: *Runner) gm.Ctx {
        return .{ .d = &r.d, .k = &r.k, .dense = &r.dense, .s = r.s };
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
        if (r.current) |j| if (j.id == id) {
            @atomicStore(bool, &j.cancelled, true, .release);
        };
        for (r.queue.items) |j| if (j.id == id) @atomicStore(bool, &j.cancelled, true, .release);
    }

    fn statusCtx(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const r = of(ctx);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        out.* = .{ .healthy = r.healthy, .running = @intFromBool(r.current != null), .waiting = @intCast(r.queue.items.len) };
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
        if (r.current) |j| @atomicStore(bool, &j.cancelled, true, .release);
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
        r.tokens.deinit(r.gpa);
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
            r.mutex.lockUncancelable(r.io);
            while (r.queue.items.len == 0 and !r.stopping) r.cond.waitUncancelable(r.io, &r.mutex);
            if (r.queue.items.len == 0) {
                r.mutex.unlock(r.io);
                return;
            }
            const job = r.queue.orderedRemove(0);
            r.current = job;
            r.mutex.unlock(r.io);
            var msg_buf: [256]u8 = undefined;
            const reason: api.Reason, const msg: []const u8 = r.serve(job) catch |e| .{ .failed, std.fmt.bufPrint(&msg_buf, "hy4 forward failed: {s}", .{@errorName(e)}) catch "hy4 forward failed" };
            if (reason == .failed) std.log.err("hy4: request {d}: {s}", .{ job.id, msg });
            r.mutex.lockUncancelable(r.io);
            r.current = null;
            r.mutex.unlock(r.io);
            job.sink.event(job.sink.ctx, job.id, &.{ .finished = .{ .reason = reason, .message = msg } });
            r.gpa.destroy(job);
        }
    }

    fn cancelled(j: *Job) bool {
        return @atomicLoad(bool, &j.cancelled, .acquire);
    }

    /// The prompt's 3D positions (pictures as their grids), the next text position returned.
    fn positions(r: *Runner, req: *const api.Request) !i32 {
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
                    r.pos[0][i + k] = cur;
                    r.pos[1][i + k] = cur + @as(i32, @intCast(y));
                    r.pos[2][i + k] = cur + @as(i32, @intCast(x));
                    k += 1;
                };
                cur += @intCast(@max(rows, cols));
                i += im.tokens;
                img += 1;
                continue;
            }
            for (0..3) |p| r.pos[p][i] = cur;
            cur += 1;
            i += 1;
        }
        return cur;
    }

    /// The Engram rows of tokens [p0, p0 + t) for layer `e` into the work buffer, and the owner mask.
    fn engramRows(r: *Runner, e: *const bind.EngramHost, p0: u32, t: u32) !void {
        const tx = &r.c.text;
        const heads = e.hasher.sizes.len;
        const hd = e.dim;
        const E = heads * hd;
        var ids: [config.max_hash_heads]u32 = undefined;
        const toks = r.tokens.items;
        for (0..t) |i| {
            const g = p0 + i;
            const tok = toks[g];
            const owns = engram_hash.owns(tok, 120120, 120124, tx.engram_pad);
            r.mask_host[i] = @intFromBool(owns);
            const out = r.emb_host[i * E ..][0..E];
            if (!owns) {
                @memset(out, 0);
                continue;
            }
            const time: usize = @intCast(@max(r.pos[0][g], 0));
            const start = g - @min(time, g);
            e.hasher.row(toks[0 .. g + 1], start, g, ids[0..heads]);
            for (0..heads) |h| {
                const row = e.offsets[h] + ids[h];
                if (row >= e.rows) return error.Hy4EngramRow;
                const vb = r.row_buf[0..hd];
                if (try e.file.readPositionalAll(r.io, vb, e.values_at + row * hd) != hd) return error.Hy4EngramRead;
                const sb = r.row_buf[hd..][0..2];
                if (try e.file.readPositionalAll(r.io, sb, e.scales_at + row * 2) != 2) return error.Hy4EngramRead;
                const scale: f32 = @bitCast(@as(u32, std.mem.readInt(u16, sb, .little)) << 16);
                for (out[h * hd ..][0..hd], vb) |*o, b| o.* = r.e4m3[b] * scale;
            }
        }
        const c = r.ctx_();
        try c.upload(r.work.emb, std.mem.sliceAsBytes(r.emb_host[0 .. t * E]));
        try c.upload(r.work.mask, r.mask_host[0..t]);
        try r.s.synchronize();
    }

    /// Rows [p0, p0 + t) of the request through every layer; the last row's logits into `r.logits`.
    fn forward(r: *Runner, p0: u32, t: u32, spans: []const gm.ImageSpan) !void {
        const tx = &r.c.text;
        const c = r.ctx_();
        const w = &r.work;
        for (0..t) |i| r.ids_host[i] = @intCast(r.tokens.items[p0 + i]);
        try c.upload(w.tokens, std.mem.sliceAsBytes(r.ids_host[0..t]));
        for (0..3) |p| try c.upload(w.pos + 4 * (@as(u64, p) * r.ctx_len + p0), std.mem.sliceAsBytes(r.pos[p][p0..][0..t]));
        try gm.embedWithImage(w, &r.m.head, t, tx.hc_mult, tx.hidden, spans);
        for (r.m.layers, 0..) |*L, li| {
            if (L.engram != null) {
                const e = &(r.engram[li] orelse return error.Hy4Engram);
                try r.engramRows(e, p0, t);
            }
            try gm.layer(w, L, r.caches[li], p0, t);
        }
        try gm.logits(w, &r.m.head, t - 1, tx.hc_mult, tx.hidden, tx.eps, tx.hc_eps);
        try c.download(std.mem.sliceAsBytes(r.logits), w.head);
    }

    fn sample(r: *Runner, req: *const api.Request, position: u64) !u32 {
        if (req.sampling) |s| if (s.temperature > 0) {
            for (r.values, r.logits) |*v, l| v.* = l;
            return @intCast(try sampling.choose(r.gpa, r.values, r.cand, position, s));
        };
        var best: usize = 0;
        for (r.logits, 0..) |l, i| if (l > r.logits[best]) {
            best = i;
        };
        return @intCast(best);
    }

    fn serve(r: *Runner, job: *Job) !struct { api.Reason, []const u8 } {
        const req = job.req;
        const n: u32 = @intCast(req.prompt.len);
        if (n == 0) return .{ .failed, "empty prompt" };
        if (n + 1 >= r.ctx_len) return .{ .length, "the prompt fills the context" };
        r.tokens.clearRetainingCapacity();
        try r.tokens.appendSlice(r.gpa, req.prompt);
        var next_pos = try r.positions(req);
        // pictures through the tower first, each into its own rows of the image buffer
        var spans: std.ArrayList(gm.ImageSpan) = .empty;
        defer spans.deinit(r.gpa);
        if (req.images.len > 0) {
            const tw = &(r.m.tower orelse return .{ .failed, "this server was started without --vision" });
            var used: u32 = 0;
            var scratch = std.heap.ArenaAllocator.init(r.gpa);
            defer scratch.deinit();
            for (req.images) |im| {
                if (used + im.tokens > r.img_tokens) return .{ .failed, "the request's pictures exceed --vision-image-tokens" };
                const dst = r.img_buf.?.ptr + @as(u64, used) * r.c.text.hidden * 4;
                try gv.forward(scratch.allocator(), r.ctx_(), tw, &r.vwork.?, im.pixels, .{ .h = im.gh, .w = im.gw }, dst);
                try spans.append(r.gpa, .{ .src = dst, .row = im.at, .rows = im.tokens });
                used += im.tokens;
            }
        }
        // the prompt in chunks; image spans as they cross each chunk
        var p0: u32 = 0;
        var cut: std.ArrayList(gm.ImageSpan) = .empty;
        defer cut.deinit(r.gpa);
        while (p0 < n) {
            if (cancelled(job)) return .{ .cancelled, "" };
            const t = @min(chunk_rows, n - p0);
            cut.clearRetainingCapacity();
            for (spans.items) |sp| {
                const lo = @max(sp.row, p0);
                const hi = @min(sp.row + sp.rows, p0 + t);
                if (lo >= hi) continue;
                try cut.append(r.gpa, .{ .src = sp.src + @as(u64, lo - sp.row) * r.c.text.hidden * 4, .row = lo - p0, .rows = hi - lo });
            }
            try r.forward(p0, t, cut.items);
            p0 += t;
        }
        job.sink.event(job.sink.ctx, job.id, &.{ .prefilled = 0 });
        var made: u32 = 0;
        while (true) {
            const tok = try r.sample(req, r.tokens.items.len);
            try r.tokens.append(r.gpa, tok);
            made += 1;
            const one = [1]u32{tok};
            job.sink.event(job.sink.ctx, job.id, &.{ .tokens = &one });
            if (std.mem.indexOfScalar(u32, req.eos, tok) != null) return .{ .stop, "" };
            if (req.stop) |st| if (st.check(st.ctx, r.tokens.items[n..])) return .{ .stop, "" };
            if (made >= req.max_tokens) return .{ .length, "" };
            if (cancelled(job)) return .{ .cancelled, "" };
            const at: u32 = @intCast(r.tokens.items.len - 1);
            if (at + 1 >= r.ctx_len) return .{ .length, "" };
            for (0..3) |p| r.pos[p][at] = next_pos;
            next_pos += 1;
            try r.forward(at, 1, &.{});
        }
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
