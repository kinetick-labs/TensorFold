//! A runner hold's parts on real replies through the served runner (runner_open, TF_RUNNER* knobs), no GPU.
const std = @import("std");
const runner = @import("runner.zig");
const ropen = @import("runner_open.zig");
const route = @import("runner_graph_route.zig");
const graph = @import("runner_graph.zig");
const cost = @import("runner_cost_bench.zig");
const dsv41 = @import("dsv41_tokenizer");
const linux = std.os.linux;

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

/// The engine thread inside a round: busy on its CPU (a CUDA sync spins) until `until`.
fn spinUntil(until: u64) void {
    while (now() < until) std.atomic.spinLoopHint();
}

/// The process environment main was given, for the knob lookup a runner config parses with.
var environ: ?*const std.process.Environ.Map = null;

fn env(name: []const u8) ?[]const u8 {
    const v = (environ orelse return null).get(name) orelse return null;
    return if (v.len == 0) null else v;
}

/// One owned or abstained hold: its stages' times (ns from the trigger) and what the reply was forced.
const Fire = struct {
    owned: bool,
    graph: bool,
    feed: u64, // the trigger token's feed: scan, the cell's open, its translation thread's spawn
    translated: u64,
    started: u64,
    done: u64,
    settled: u64,
    last_tick: u64, // the tick that returned the tokens: the result's poll and tokenize
    exec: u64, // the job's own time on its thread (boot and run)
    translate: u64 = 0, // on this thread, alone: the frontend's translation
    lower: u64 = 0, // graph route: the calc program to one bank packet
    bind: u64 = 0, // Machine.boot: the packet's plans bound
    eval: u64 = 0, // Machine.run
    step_ps: u64 = 0, // Machine.run's picoseconds per step
    ticks: u32 = 1,
};

/// Every reply's tokens through one session per reply; at each hold the engine loop's ticks until the result.
const Arm = struct {
    name: []const u8,
    first_ns: u64, // the trigger round's rest before the loop's first tick (0: ticks spin, no cadence)
    round_ns: u64,
    pin: bool, // the engine thread pinned to one CPU and busy between ticks
    hogs: bool = false, // busy threads on the engine's and the runner's CPUs
};

fn pass(gpa: std.mem.Allocator, r: *runner.Runner, replies: []const cost.Reply, arm: Arm, fires: *std.ArrayList(Fire), digest: *std.hash.Wyhash) !void {
    for (replies) |rep| {
        const s = (try r.session(.{})).?;
        defer s.destroy();
        const hook = s.hook();
        for (rep.tokens, 0..) |tok, i| {
            const t0 = now();
            const held = hook.token(hook.ptr, tok, i < rep.think_end, true, false);
            const t1 = now();
            if (!held) continue;
            const cell = s.cells.items[s.waiting.?.index];
            var f: Fire = .{ .owned = false, .graph = false, .feed = t1 - t0, .translated = 0, .started = 0, .done = 0, .settled = 0, .last_tick = 0, .exec = 0 };
            const ids = while (true) {
                if (arm.first_ns == 0) {
                    if (f.translated == 0 and cell.translated.load(.acquire)) f.translated = now() - t1;
                } else spinUntil(t1 + arm.first_ns + @as(u64, f.ticks - 1) * arm.round_ns);
                const k0 = now();
                const got = try s.tick();
                const k1 = now();
                if (f.started == 0 and (cell.job != null or cell.result != null)) f.started = k1 - t1;
                if (f.done == 0 and (cell.result != null or cell.abstained)) f.done = k1 - t1;
                if (got) |x| {
                    f.settled = k1 - t1;
                    f.last_tick = k1 - k0;
                    break x;
                }
                f.ticks += 1;
                if (arm.first_ns == 0) sleepNs(20 * std.time.ns_per_us);
            };
            defer gpa.free(ids);
            digest.update(std.mem.sliceAsBytes(ids));
            digest.update(&.{0xff});
            f.owned = ids.len > 0;
            if (cell.result) |res| f.exec = res.exec_ns;
            if (cell.translation == .program) f.graph = cell.translation.program.kind == .graph;
            if (arm.first_ns == 0 and f.owned) try direct(gpa, r, cell, &f);
            try fires.append(gpa, f);
        }
    }
}

/// The owned cell's stages again on this thread alone: frontend, graph lowering, plan bind, evaluation.
fn direct(gpa: std.mem.Allocator, r: *runner.Runner, cell: anytype, f: *Fire) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const inner = @import("runner_frontend.zig").frontend();
    const t0 = now();
    const t = try inner.translate(inner.ctx, a, .{ .form = cell.form, .lang = cell.lang, .code = cell.code });
    f.translate = now() - t0;
    if (!f.graph or t != .program) return;
    const routed: *route.Routed = @ptrCast(@alignCast(r.frontend.ctx.?));
    const t1 = now();
    const bytes = try route.lower(a, routed.bank, t.program.bytes) orelse return;
    f.lower = now() - t1;
    const t2 = now();
    var m = try graph.Machine.boot(a, routed.bank, bytes);
    defer m.deinit();
    f.bind = now() - t2;
    var out: std.ArrayList(u8) = .empty;
    const t3 = now();
    const end = try m.run(r.cfg.max_steps, &out, null);
    f.eval = now() - t3;
    if (end.steps > 0) f.step_ps = f.eval * 1000 / end.steps;
}

/// Other threads busy on the engine's CPU and the runner's (server threads, a busy host).
const Hogs = struct {
    stop: std.atomic.Value(bool) = .init(false),
    threads: std.ArrayList(std.Thread) = .empty,

    fn spin(h: *Hogs, cpu: u32) void {
        pinTo(cpu);
        while (!h.stop.load(.monotonic)) std.atomic.spinLoopHint();
    }

    fn start(h: *Hogs, gpa: std.mem.Allocator, cpus: []const u32, per: u32) !void {
        h.stop.store(false, .release);
        for (cpus) |cpu| for (0..per) |_| try h.threads.append(gpa, try std.Thread.spawn(.{}, spin, .{ h, cpu }));
    }

    fn end(h: *Hogs) void {
        h.stop.store(true, .release);
        for (h.threads.items) |t| t.join();
        h.threads.clearRetainingCapacity();
    }
};

fn pinTo(cpu: u32) void {
    var set: [16]u64 = @splat(0);
    set[cpu / 64] = @as(u64, 1) << @intCast(cpu % 64);
    _ = linux.syscall3(.sched_setaffinity, 0, @sizeOf(@TypeOf(set)), @intFromPtr(&set));
}

fn unpin() void {
    const set: [16]u64 = @splat(std.math.maxInt(u64));
    _ = linux.syscall3(.sched_setaffinity, 0, @sizeOf(@TypeOf(set)), @intFromPtr(&set));
}

fn quantiles(gpa: std.mem.Allocator, xs: []u64) ![3]f64 {
    if (xs.len == 0) return .{ 0, 0, 0 };
    const v = try gpa.dupe(u64, xs);
    defer gpa.free(v);
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    const at = struct {
        fn q(s: []const u64, p: f64) f64 {
            const i: usize = @intFromFloat(@floor(p * @as(f64, @floatFromInt(s.len - 1)) + 0.5));
            return @as(f64, @floatFromInt(s[i])) / 1e3;
        }
    };
    return .{ at.q(v, 0.5), at.q(v, 0.9), at.q(v, 1.0) };
}

/// One stage's line in microseconds over the owned holds: p50, p90, max.
fn row(gpa: std.mem.Allocator, name: []const u8, fires: []const Fire, comptime field: []const u8, graph_only: bool) !void {
    var xs: std.ArrayList(u64) = .empty;
    defer xs.deinit(gpa);
    for (fires) |f| if (f.owned and (!graph_only or f.graph)) try xs.append(gpa, @field(f, field));
    const q = try quantiles(gpa, xs.items);
    std.debug.print("| {s} | {d} | {d:.0} | {d:.0} | {d:.0} |\n", .{ name, xs.items.len, q[0], q[1], q[2] });
}

/// A stage's span between two marks (ns from the trigger), as a Fire-like field for `row`.
fn span(gpa: std.mem.Allocator, name: []const u8, fires: []const Fire, comptime from: []const u8, comptime to: []const u8) !void {
    var xs: std.ArrayList(u64) = .empty;
    defer xs.deinit(gpa);
    for (fires) |f| if (f.owned) try xs.append(gpa, @field(f, to) -| @field(f, from));
    const q = try quantiles(gpa, xs.items);
    std.debug.print("| {s} | {d} | {d:.0} | {d:.0} | {d:.0} |\n", .{ name, xs.items.len, q[0], q[1], q[2] });
}

/// runner-hold-bench <tokenizer.json> <replies.jsonl> <engine-cpu> [first-ms round-ms settle-us hog-cpus per-cpu]
pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    environ = init.environ_map;
    if (argv.len < 4) {
        std.debug.print("usage: runner-hold-bench <tokenizer.json> <replies.jsonl> <engine-cpu> [first-ms round-ms settle-us hog-cpus per-cpu] (TF_RUNNER* or TF_DSV41_RUNNER* knobs)\n", .{});
        return 2;
    }
    const cpu = try std.fmt.parseInt(u32, argv[3], 10);
    const first_ms = if (argv.len > 4) try std.fmt.parseInt(u64, argv[4], 10) else 12;
    const round_ms = if (argv.len > 5) try std.fmt.parseInt(u64, argv[5], 10) else 50;
    const settle_us = if (argv.len > 6) try std.fmt.parseInt(u32, argv[6], 10) else 3000;
    var hog_cpus: std.ArrayList(u32) = .empty;
    if (argv.len > 7) {
        var it = std.mem.splitScalar(u8, argv[7], ',');
        while (it.next()) |x| try hog_cpus.append(gpa, try std.fmt.parseInt(u32, x, 10));
    }
    const per = if (argv.len > 8) try std.fmt.parseInt(u32, argv[8], 10) else 2;
    var hogs: Hogs = .{};
    const tok = try dsv41.Tokenizer.load(gpa, init.io, argv[1]);
    var table: cost.Table = .{ .tok = tok };
    const replies = try cost.load(gpa, init.io, &table, argv[2]);
    var cfg = (try ropen.bindings.dsv41.parse(&env)) orelse {
        std.debug.print("TF_RUNNER and TF_DSV41_RUNNER are off\n", .{});
        return 2;
    };
    // the engine thread opens the runner: unplaced job threads inherit its pin, as served
    pinTo(cpu);
    cfg.settle_us = 0;
    const old = try ropen.open(gpa, cfg);
    cfg.settle_us = settle_us;
    const new = try ropen.open(gpa, cfg);
    for ([_]*runner.Runner{ old, new }) |r| r.text = .{ .ctx = &table, .bytes = cost.Table.bytes, .encode = cost.Table.encode };
    const first = first_ms * std.time.ns_per_ms;
    const round = round_ms * std.time.ns_per_ms;
    const Case = struct { arm: Arm, r: *runner.Runner, settle: u32 };
    const cases = [_]Case{
        .{ .arm = .{ .name = "stages: ticks spin, engine unpinned (SETTLE_US=0)", .first_ns = 0, .round_ns = 0, .pin = false }, .r = old, .settle = 0 },
        .{ .arm = .{ .name = "served cadence, pinned engine: SETTLE_US=0 (one stage a tick)", .first_ns = first, .round_ns = round, .pin = true }, .r = old, .settle = 0 },
        .{ .arm = .{ .name = "served cadence, pinned engine: SETTLE_US", .first_ns = first, .round_ns = round, .pin = true }, .r = new, .settle = settle_us },
        .{ .arm = .{ .name = "served cadence, pinned engine, busy CPUs: SETTLE_US=0", .first_ns = first, .round_ns = round, .pin = true, .hogs = true }, .r = old, .settle = 0 },
        .{ .arm = .{ .name = "served cadence, pinned engine, busy CPUs: SETTLE_US", .first_ns = first, .round_ns = round, .pin = true, .hogs = true }, .r = new, .settle = settle_us },
    };
    var want: ?u64 = null;
    for (cases) |c| {
        if (c.arm.hogs and hog_cpus.items.len == 0) continue;
        if (c.arm.hogs) try hogs.start(gpa, hog_cpus.items, per);
        defer if (c.arm.hogs) hogs.end();
        c.r.cfg.settle_us = c.settle;
        if (c.arm.pin) pinTo(cpu) else unpin();
        var fires: std.ArrayList(Fire) = .empty;
        defer fires.deinit(gpa);
        var digest = std.hash.Wyhash.init(0);
        try pass(gpa, c.r, replies, c.arm, &fires, &digest);
        const d = digest.final();
        const same = if (want) |w| w == d else blk: {
            want = d;
            break :blk true;
        };
        var owned: u32 = 0;
        var on_graph: u32 = 0;
        var by_ticks: [5]u32 = @splat(0);
        for (fires.items) |f| if (f.owned) {
            owned += 1;
            on_graph += @intFromBool(f.graph);
            by_ticks[@min(f.ticks, 5) - 1] += 1;
        };
        std.debug.print("\n## {s}\n{d} holds, {d} owned ({d} graph), owned tokens digest {x} ({s})\n", .{ c.arm.name, fires.items.len, owned, on_graph, d, if (same) "identical" else "DIFFERS" });
        std.debug.print("owned holds by ticks 1/2/3/4/5+: {d}/{d}/{d}/{d}/{d}\n", .{ by_ticks[0], by_ticks[1], by_ticks[2], by_ticks[3], by_ticks[4] });
        std.debug.print("| stage (us, owned holds) | n | p50 | p90 | max |\n|---|---|---|---|---|\n", .{});
        if (c.arm.first_ns == 0) {
            try row(gpa, "scan close + cell open (trigger token's feed)", fires.items, "feed", false);
            try row(gpa, "frontend translate, alone", fires.items, "translate", false);
            try row(gpa, "graph lower (calc to bank packet), alone", fires.items, "lower", true);
            try row(gpa, "graph plan bind (Machine.boot), alone", fires.items, "bind", true);
            try row(gpa, "evaluate (Machine.run), alone", fires.items, "eval", true);
            try row(gpa, "evaluate ns per step (x1000)", fires.items, "step_ps", true);
            try row(gpa, "trigger to translated (thread)", fires.items, "translated", false);
            try span(gpa, "translated to job started (next tick)", fires.items, "translated", "started");
            try span(gpa, "job started to done (thread)", fires.items, "started", "done");
            try row(gpa, "job's own time (exec_ns)", fires.items, "exec", false);
            try row(gpa, "result tokenize (the returning tick)", fires.items, "last_tick", false);
        }
        try row(gpa, "held: trigger to tokens", fires.items, "settled", false);
        if (!same) return 1;
    }
    return 0;
}
