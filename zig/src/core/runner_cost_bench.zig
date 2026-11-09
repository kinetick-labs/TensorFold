//! Runner host cost on real replies, no GPU: ns/token off/scan/inline/firing and the hold wait (usage: main).
const std = @import("std");
const runner = @import("runner.zig");
const rapi = @import("runner_api.zig");
const frontend = @import("runner_frontend.zig");
const dsv41 = @import("dsv41_tokenizer");

pub const Reply = struct { tokens: []u32, think_end: usize };

pub const Table = struct {
    tok: *dsv41.Tokenizer,
    pub fn bytes(ctx: ?*anyopaque, token: u32) []const u8 {
        const t: *Table = @ptrCast(@alignCast(ctx.?));
        if (token >= t.tok.vocabSize() or t.tok.isSpecial(token)) return "";
        return t.tok.bytes[t.tok.offsets[token]..t.tok.offsets[token + 1]];
    }
    pub fn encode(ctx: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
        const t: *Table = @ptrCast(@alignCast(ctx.?));
        const e = try t.tok.acquire();
        defer t.tok.release(e);
        var out: std.ArrayList(u32) = .empty;
        try e.encode(a, text, &out);
        return out.toOwnedSlice(a);
    }
};

/// A backend whose every run is done at its first poll with a one-line value (the circuit's time is not host cost).
const Instant = struct {
    fn backend() rapi.Backend {
        return .{ .ctx = @ptrFromInt(8), .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(_: *anyopaque, _: u64, _: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        return 1;
    }
    fn poll(_: *anyopaque, _: rapi.JobId) rapi.Poll {
        return .{ .done = .{ .status = .ok, .out = "1267650600228229401496703205376\n", .steps = 1 } };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
};

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

const Mode = enum { off, scan_cr, scan_inline, fire };

const Totals = struct { tokens: u64 = 0, feed_ns: u64 = 0, fires: u64 = 0, tick_ns: u64 = 0, settle_ns: u64 = 0, sink: u64 = 0, receipt_ns: u64 = 0, receipt_bytes: u64 = 0, replies: u64 = 0 };

/// One pass over every reply; `poll_ns`: the idle wait between ticks while held; `forced`: results fed after holds.
fn pass(gpa: std.mem.Allocator, r: *runner.Runner, replies: []const Reply, mode: Mode, poll_ns: u64, forced: bool) !Totals {
    var t: Totals = .{};
    for (replies) |rep| {
        if (mode == .off) {
            const t0 = now();
            for (rep.tokens) |tok| t.sink +%= r.text.?.bytes(r.text.?.ctx, tok).len;
            t.feed_ns += now() - t0;
            t.tokens += rep.tokens.len;
            continue;
        }
        const s = (try r.session(.{})).?;
        defer s.destroy();
        const hook = s.hook();
        for (rep.tokens, 0..) |tok, i| {
            const t0 = now();
            const held = hook.token(hook.ptr, tok, i < rep.think_end, mode == .fire, false);
            t.feed_ns += now() - t0;
            t.tokens += 1;
            if (!held) continue;
            // held: tick as the engine loop does, waiting poll_ns between ticks, until the result is settled
            const h0 = now();
            while (true) {
                const k0 = now();
                const got = try s.tick();
                t.tick_ns += now() - k0;
                if (got) |ids| {
                    defer gpa.free(ids);
                    if (forced) for (ids) |f| {
                        const f0 = now();
                        _ = hook.token(hook.ptr, f, i < rep.think_end, false, false);
                        t.feed_ns += now() - f0;
                        t.tokens += 1;
                    };
                    break;
                }
                if (poll_ns > 0) sleepNs(poll_ns);
            }
            t.settle_ns += now() - h0;
            t.fires += 1;
        }
        const r0 = now();
        t.receipt_bytes += s.receipt().len;
        t.receipt_ns += now() - r0;
        t.replies += 1;
    }
    return t;
}

/// replies.jsonl's lines ({"reasoning": "...", "content": "..."}) as token ids; never freed (a bench's whole run).
pub fn load(gpa: std.mem.Allocator, io: std.Io, table: *Table, path: []const u8) ![]Reply {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    var replies: std.ArrayList(Reply) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var total: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const p = try std.json.parseFromSlice(struct { reasoning: []const u8, content: []const u8 }, gpa, line, .{ .ignore_unknown_fields = true });
        const a = try Table.encode(table, gpa, p.value.reasoning);
        const b = try Table.encode(table, gpa, p.value.content);
        const all = try std.mem.concat(gpa, u32, &.{ a, b });
        try replies.append(gpa, .{ .tokens = all, .think_end = a.len });
        total += all.len;
    }
    std.debug.print("{d} replies, {d} tokens\n", .{ replies.items.len, total });
    return replies.items;
}

/// runner-cost-bench <tokenizer.json> <replies.jsonl> [repeats]; a line is {"reasoning": "...", "content": "..."}.
pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: runner-cost-bench <tokenizer.json> <replies.jsonl> [repeats]\n", .{});
        return 2;
    }
    const repeats = if (argv.len > 3) try std.fmt.parseInt(u32, argv[3], 10) else 3;
    const tok = try dsv41.Tokenizer.load(gpa, init.io, argv[1]);
    var table: Table = .{ .tok = tok };
    const replies = try load(gpa, init.io, &table, argv[2]);
    var ctx: frontend.Context = .{ .io = init.io };
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "bench" }, .frontend = ctx.frontend(), .backend = Instant.backend(), .text = .{ .ctx = &table, .bytes = Table.bytes, .encode = Table.encode } };
    const Case = struct { name: []const u8, mode: Mode, inline_math: bool, poll_ns: u64, forced: bool = false, cells: bool = false };
    const cases = [_]Case{
        .{ .name = "runner off (byte lookup only)", .mode = .off, .inline_math = false, .poll_ns = 0 },
        .{ .name = "on, fence/repl scan, no holds", .mode = .scan_cr, .inline_math = false, .poll_ns = 0 },
        .{ .name = "on, +inline scan, no holds", .mode = .scan_inline, .inline_math = true, .poll_ns = 0 },
        .{ .name = "on, +inline, firing, tick spin", .mode = .fire, .inline_math = true, .poll_ns = 0 },
        .{ .name = "on, +inline, firing, 1 ms idle", .mode = .fire, .inline_math = true, .poll_ns = std.time.ns_per_ms },
        .{ .name = "on, +inline, firing, 20 ms idle", .mode = .fire, .inline_math = true, .poll_ns = 20 * std.time.ns_per_ms },
        .{ .name = "firing, forced fed, records off", .mode = .fire, .inline_math = true, .poll_ns = 0, .forced = true },
        .{ .name = "firing, forced fed, records on", .mode = .fire, .inline_math = true, .poll_ns = 0, .forced = true, .cells = true },
    };
    for (cases) |c| {
        r.cfg.scan.inline_math = c.inline_math;
        r.cfg.cells = c.cells;
        var best: ?Totals = null;
        for (0..repeats) |_| {
            const t = try pass(gpa, &r, replies, c.mode, c.poll_ns, c.forced);
            if (best == null or t.feed_ns < best.?.feed_ns) best = t;
        }
        const t = best.?;
        const per = @as(f64, @floatFromInt(t.feed_ns)) / @as(f64, @floatFromInt(t.tokens));
        std.debug.print("{s:34} {d:8.1} ns/token feed", .{ c.name, per });
        if (t.fires > 0) {
            const f: f64 = @floatFromInt(t.fires);
            std.debug.print(", {d} fires: tick {d:.1} us/fire, hold-to-result {d:.2} ms/fire", .{ t.fires, @as(f64, @floatFromInt(t.tick_ns)) / f / 1e3, @as(f64, @floatFromInt(t.settle_ns)) / f / 1e6 });
        }
        if (t.replies > 0) std.debug.print(", receipt {d:.2} us/reply ({d} bytes/reply)", .{ @as(f64, @floatFromInt(t.receipt_ns)) / @as(f64, @floatFromInt(t.replies)) / 1e3, t.receipt_bytes / t.replies });
        std.debug.print("\n", .{});
    }
    return 0;
}
