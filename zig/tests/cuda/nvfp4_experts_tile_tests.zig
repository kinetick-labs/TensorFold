//! The NVFP4 expert tile against the decode kernel it replaces on prompt chunks, byte for byte (synthetic experts).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const ex_ = cuda.experts;
const fx4 = ex_.Nvfp4Experts;
const tile_ = cuda.nvfp4_tile;

const dims: usize = 2560;
const slots: usize = 11;
const top: usize = 10;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Packed blocks: random code words; moderate e4m3 scale bytes, or (`special`) any byte but the NaN encodings.
fn fillBlocks(rng: std.Random, w: []u32, special: bool) void {
    for (w, 0..) |*v, i| {
        if (i % fx4.words < 128) {
            v.* = rng.int(u32);
            continue;
        }
        var x: u32 = 0;
        for (0..4) |q| {
            var b: u8 = if (special) rng.int(u8) else rng.intRangeAtMost(u8, 0x18, 0x5F) | (if (rng.uintLessThan(u8, 8) == 0) @as(u8, 0x80) else 0);
            if (b & 0x7f == 0x7f) b ^= 1;
            x |= @as(u32, b) << @intCast(8 * q);
        }
        v.* = x;
    }
}

/// Ten distinct experts a row and the shared slot (`experts`, skipped); `hot`: seven in ten picks from the first eight.
fn pickRow(rng: std.Random, out: []i32, experts: usize, hot: bool) void {
    var n: usize = 0;
    while (n < top) {
        const c: i32 = @intCast(if (hot and rng.uintLessThan(u8, 10) < 7) rng.uintLessThan(u32, 8) else rng.uintLessThan(u32, @intCast(experts)));
        if (std.mem.indexOfScalar(i32, out[0..n], c) != null) continue;
        out[n] = c;
        n += 1;
    }
    out[top] = @intCast(experts);
}

const Weights = struct {
    up: cuda.DeviceBuffer,
    down: cuda.DeviceBuffer,
    up_scale: cuda.DeviceBuffer,
    down_scale: cuda.DeviceBuffer,
    layer: ex_.Layer,

    fn make(gpu: Gpu, rng: std.Random, experts: usize, ni: usize, special: bool) !Weights {
        const a = gpu.gpa;
        const up_words = experts * (ni / 32) * (dims / 32) * 2 * fx4.words;
        const down_words = experts * (dims / 32) * (ni / 32) * fx4.words;
        const buf = try a.alloc(u32, @max(up_words, down_words));
        defer a.free(buf);
        fillBlocks(rng, buf[0..up_words], special);
        var up = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(buf[0..up_words]));
        errdefer up.free();
        fillBlocks(rng, buf[0..down_words], special);
        var down = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(buf[0..down_words]));
        errdefer down.free();
        const scales = try a.alloc(f32, 3 * experts);
        defer a.free(scales);
        for (scales) |*v| v.* = 0.002 + rng.float(f32) * 0.02;
        var us = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales[0 .. 2 * experts]));
        errdefer us.free();
        var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales[2 * experts ..]));
        errdefer ds.free();
        return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .layer = .{ .format = .nvfp4, .up = up.ptr, .down = down.ptr, .up_scale = us.ptr, .down_scale = ds.ptr, .width = ni, .dims = dims, .experts = experts } };
    }

    fn free(w: *Weights) void {
        w.up.free();
        w.down.free();
        w.up_scale.free();
        w.down_scale.free();
    }
};

/// A plan's scratch for `pairs` pairs over `classes` experts at any tile.
const PlanBufs = struct {
    bufs: [5]cuda.DeviceBuffer,

    fn make(gpu: Gpu, pairs: usize, classes: usize) !PlanBufs {
        const nblk = (pairs + 1023) / 1024;
        const sizes = [5]usize{ pairs * 4, grouped.maxItems(pairs, classes, 16) * 12, 8, pairs * 4, nblk * classes * 4 };
        var p: PlanBufs = undefined;
        var made: usize = 0;
        errdefer for (p.bufs[0..made]) |*b| b.free();
        for (sizes, &p.bufs) |n, *b| {
            b.* = try cuda.DeviceBuffer.alloc(gpu.d, @max(n, 16));
            made += 1;
        }
        return p;
    }

    fn free(p: *PlanBufs) void {
        for (&p.bufs) |*b| b.free();
    }

    fn plan(p: PlanBufs) grouped.Plan {
        return .{ .members = p.bufs[0].ptr, .items = p.bufs[1].ptr, .counts = p.bufs[2].ptr, .rank = p.bufs[3].ptr, .hist = p.bufs[4].ptr };
    }
};

const Combo = struct { pairs: usize, gu: usize, down: usize };

/// The default (64 pairs, gate/up 4 tiles 4 stages, down 2 tiles 3 stages) first, then the other instances and tiles.
const combos = [_]Combo{
    .{ .pairs = 64, .gu = 1, .down = 2 },
    .{ .pairs = 64, .gu = 0, .down = 0 },
    .{ .pairs = 64, .gu = 2, .down = 1 },
    .{ .pairs = 64, .gu = 3, .down = 2 },
    .{ .pairs = 64, .gu = 4, .down = 0 },
    .{ .pairs = 32, .gu = 2, .down = 2 },
    .{ .pairs = 48, .gu = 4, .down = 2 },
    .{ .pairs = 128, .gu = 1, .down = 1 },
    .{ .pairs = 16, .gu = 2, .down = 2 },
};

/// One chunk: the decode kernel on 16-pair items, then each combo's tile kernels, gate/up and down alike.
fn oneChunk(gpu: Gpu, router: grouped.Router, ex: *const fx4, w: Weights, sms: usize, s: cuda.Stream, rng: std.Random, rows: usize, hot: bool, compare: bool, timing: bool) !void {
    const a = gpu.gpa;
    const pairs = rows * slots;
    const classes = w.layer.experts + 1;
    const ni = w.layer.width;
    const skip: i32 = @intCast(w.layer.experts);
    const picks = try a.alloc(i32, pairs);
    defer a.free(picks);
    for (0..rows) |r| pickRow(rng, picks[r * slots ..][0..slots], w.layer.experts, hot);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer dpicks.free();
    const x = try a.alloc(u16, rows * dims);
    defer a.free(x);
    for (x) |*v| v.* = bf16(rng.float(f32) * 4 - 2);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var plan16 = try PlanBufs.make(gpu, pairs, classes);
    defer plan16.free();
    var plan_t = try PlanBufs.make(gpu, pairs, classes);
    defer plan_t.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * ni * 2);
    defer act.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * dims * 4);
    defer y.free();
    var act_t = try cuda.DeviceBuffer.alloc(gpu.d, pairs * ni * 2);
    defer act_t.free();
    var y_t = try cuda.DeviceBuffer.alloc(gpu.d, pairs * dims * 4);
    defer y_t.free();
    const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = dims, .slots = slots };
    const down_in: ex_.Rows = .{ .x = act.ptr, .stride = ni };

    // the decode kernel's bytes, unwritten pairs (the shared slot) left at 0xff
    try act.fill8(0xff, s.handle);
    try y.fill8(0xff, s.handle);
    try router.route(s, dpicks.ptr, pairs, classes, 16, plan16.plan());
    const items16 = grouped.maxItems(pairs, classes, 16);
    try ex.run(s, .up, up_in, w.layer, plan16.plan(), act.ptr, items16, skip);
    try ex.run(s, .down_f32, down_in, w.layer, plan16.plan(), y.ptr, items16, skip);
    try s.synchronize();
    const want_act = if (compare) try check.download(gpu, act) else &.{};
    defer if (compare) a.free(want_act);
    const want_y = if (compare) try check.download(gpu, y) else &.{};
    defer if (compare) a.free(want_y);

    for (if (compare) combos[0..] else combos[0..0]) |c| {
        var t = try tile_.Tile.load(gpu.d, sms, .{ .pairs = c.pairs, .gu = c.gu, .down = c.down });
        defer t.unload();
        try router.route(s, dpicks.ptr, pairs, classes, c.pairs, plan_t.plan());
        const items = grouped.maxItems(pairs, classes, c.pairs);
        try act_t.fill8(0xff, s.handle);
        try y_t.fill8(0xff, s.handle);
        try t.run(s, .up, up_in, w.layer, plan_t.plan(), act_t.ptr, items, skip);
        // the tile's down on the decode kernel's activations, so a gate/up difference cannot hide in it
        try t.run(s, .down_f32, down_in, w.layer, plan_t.plan(), y_t.ptr, items, skip);
        try s.synchronize();
        const got_act = try check.download(gpu, act_t);
        defer a.free(got_act);
        try check.sameBytes("tile gate-up", got_act, want_act);
        const got_y = try check.download(gpu, y_t);
        defer a.free(got_y);
        try check.sameBytes("tile down", got_y, want_y);
    }
    if (compare) check.pass("nvfp4 expert tile: {d} rows ({s}), width {d}, {d} tile configurations equal the decode kernel's bytes", .{ rows, if (hot) "hot experts" else "uniform", ni, combos.len });
    if (timing) try time(gpu, router, ex, w, sms, s, up_in, down_in, dpicks.ptr, pairs, act.ptr, y.ptr, plan16, plan_t);
}

/// The layer's time (plan, gate/up, down) on the decode kernel and on the env's tile configuration, median of seven.
fn time(gpu: Gpu, router: grouped.Router, ex: *const fx4, w: Weights, sms: usize, s: cuda.Stream, up_in: ex_.Rows, down_in: ex_.Rows, picks: u64, pairs: usize, act: u64, y: u64, plan16: PlanBufs, plan_t: PlanBufs) !void {
    const classes = w.layer.experts + 1;
    const skip: i32 = @intCast(w.layer.experts);
    var t = try tile_.Tile.load(gpu.d, sms, try tile_.Config.fromEnv());
    defer t.unload();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var ms: [2][7]f64 = undefined;
    const items16 = grouped.maxItems(pairs, classes, 16);
    const items64 = grouped.maxItems(pairs, classes, t.cfg.pairs);
    for (0..8) |rep| for (0..2) |which| {
        try start.record(s);
        if (which == 0) {
            try router.route(s, picks, pairs, classes, 16, plan16.plan());
            try ex.run(s, .up, up_in, w.layer, plan16.plan(), act, items16, skip);
            try ex.run(s, .down_f32, down_in, w.layer, plan16.plan(), y, items16, skip);
        } else {
            try router.route(s, picks, pairs, classes, t.cfg.pairs, plan_t.plan());
            try t.run(s, .up, up_in, w.layer, plan_t.plan(), act, items64, skip);
            try t.run(s, .down_f32, down_in, w.layer, plan_t.plan(), y, items64, skip);
        }
        try end.record(s);
        try end.synchronize();
        if (rep > 0) ms[which][rep - 1] = try cuda.Event.elapsedMs(start, end);
    };
    const old = check.median(&ms[0]);
    const new = check.median(&ms[1]);
    std.debug.print("INFO nvfp4 expert tile {d},{d},{d}: {d} pairs, width {d}: decode kernel {d:.3} ms, tile {d:.3} ms a layer ({d:.2}x)\n", .{ t.cfg.pairs, t.cfg.gu, t.cfg.down, pairs, w.layer.width, old, new, old / new });
}

/// `nvfp4-experts-tile [ROWS[,ROWS] [EXPERTS [time]]]`: synthetic; default rows 64,129,1024,1536,2049 and 96 experts.
pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    var rows_list: std.ArrayList(usize) = .empty;
    defer rows_list.deinit(gpu.gpa);
    var it = std.mem.tokenizeScalar(u8, if (rest.len > 0) rest[0] else "64,129,1024,1536,2049", ',');
    while (it.next()) |tok| try rows_list.append(gpu.gpa, try std.fmt.parseInt(usize, tok, 10));
    const experts: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 96;
    if (experts < 16 or rows_list.items.len == 0) return error.BadArguments;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try fx4.load(gpu.d, sms);
    defer ex.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var s = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x4e56_4650);
    const rng = prng.random();
    // `time`: the layer's time alone, at the single rank's width
    if (rest.len > 2 and std.mem.eql(u8, rest[2], "time")) {
        var w = try Weights.make(gpu, rng, experts, 640, false);
        defer w.free();
        for (rows_list.items) |rows| try oneChunk(gpu, router, &ex, w, sms, s, rng, rows, false, false, true);
        return;
    }
    // widths: the single rank's 640 and a tensor-parallel rank's 320; the second set holds every scale byte but NaN
    for ([_]bool{ false, true }) |special| for ([_]usize{ 640, 320 }) |ni| {
        var w = try Weights.make(gpu, rng, experts, ni, special);
        defer w.free();
        for (rows_list.items) |rows| for ([_]bool{ false, true }) |hot| {
            if (special and hot) continue;
            try oneChunk(gpu, router, &ex, w, sms, s, rng, rows, hot, true, !special and ni == 640 and !hot and rows == 1536);
        };
        if (special) check.pass("nvfp4 expert tile: width {d} with every scale byte but NaN, rows {any}", .{ ni, rows_list.items });
    };
}
