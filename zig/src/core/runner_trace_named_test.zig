//! TF_DSV41_RUNNER_STATED named problems (S1 R4 d_mirror rank 1b): isqrt and edit distance over literals.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const ropen = @import("runner_open.zig");
const trace = @import("runner_trace.zig");
const gpa = std.testing.allocator;

/// The last line's stated computation as display and run, after every line went through one tracer (null: none).
fn expect(whole: bool, lines: []const []const u8, want: ?[2][]const u8) !void {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.whole = whole;
    var got: ?trace.Found = null;
    for (lines) |l| got = try t.line(l);
    if (want) |w| {
        const f = got orelse {
            std.debug.print("nothing stated (whole {}): {s}\n", .{ whole, lines[lines.len - 1] });
            return error.TestExpectedStated;
        };
        try std.testing.expectEqualStrings(w[0], f.display);
        try std.testing.expectEqualStrings(w[1], f.run);
    } else if (got) |f| {
        std.debug.print("stated (whole {}): {s} from {s}\n", .{ whole, f.display, lines[lines.len - 1] });
        return error.TestUnexpectedStated;
    }
}

fn edit(comptime x: []const u8, comptime y: []const u8) [2][]const u8 {
    return .{ "edit_distance(\"" ++ x ++ "\", \"" ++ y ++ "\")", std.fmt.comptimePrint(trace.edit_program, .{ x, y }) };
}

/// The live rank-1b lines (sqrt-a, sqrt-b, edit, edit-b) and the other spellings of the same statements.
const live = [_]struct { line: []const u8, want: [2][]const u8 }{
    .{ .line = "We need answer. Need compute integer sqrt of 98765432109876543210. Need use Python.", .want = .{ "isqrt(98765432109876543210)", "from math import isqrt\nprint(isqrt(98765432109876543210))" } },
    .{ .line = "We need answer integer sqrt of N=123456789876543210123456. Need compute exactly.", .want = .{ "isqrt(123456789876543210123456)", "from math import isqrt\nprint(isqrt(123456789876543210123456))" } },
    .{ .line = "We need answer compute Levenshtein distance between \"intention\" and \"execution\". Need be careful.", .want = edit("intention", "execution") },
    .{ .line = "We need answer compute Levenshtein distance between \"algorithm\" and \"logarithm\". Need return distance.", .want = edit("algorithm", "logarithm") },
    .{ .line = "Compute the edit distance between 'kitten' and 'sitting'.", .want = edit("kitten", "sitting") },
    .{ .line = "We need the Levenshtein distance from “flaw” to “lawn” exactly.", .want = edit("flaw", "lawn") },
    .{ .line = "Need isqrt(1,000,000,007).", .want = .{ "isqrt(1000000007)", "from math import isqrt\nprint(isqrt(1000000007))" } },
    .{ .line = "We need the floor square root of 10,000,000,000,000,000.", .want = .{ "isqrt(10000000000000000)", "from math import isqrt\nprint(isqrt(10000000000000000))" } },
};

test "runner trace stated: R4 rank-1b named problems are stated with TF_DSV41_RUNNER_STATED and not without it" {
    for (live) |c| {
        try expect(true, &.{c.line}, c.want);
        try expect(false, &.{c.line}, null);
    }
}

test "runner trace stated: near misses of a named problem refuse" {
    for ([_][]const u8{
        "The edit distance is roughly 5.",
        "Need edit distance between 'kitten' and 'sitting' is about 3.",
        "Need edit distance between 'kitten' and 'sitting' maybe.",
        "Need edit distance between 'kitten' and 'sitting'?",
        "We need compute Levenshtein kitten to sitting.", // bare words: g2-edit, not literals
        "Need Levenshtein algorithm to compute.",
        "Need edit distance between s and t.",
        "Need edit distance between 'kitten' and t.",
        "Need edit distance between 'kit\\ten' and 'sitting'.",
        "Need edit distance between 'abcdefghijklmnopqrstuvwxyzabcdefg' and 'b'.", // 33 bytes
        "Need sqrt of 98765432109876543210.", // the real root, maybe: no "integer" or "floor"
        "Need integer sqrt of N.",
        "Need integer sqrt of 0123.",
        "Need integer sqrt of 2^64.",
        "Need integer sqrt of 98765432109876543210.5.",
        "Need integer sqrt of 98765432109876543210 roughly.",
        "Need isqrt(98765432109876543210 + 1).",
        "Need integer sqrt of 1234567890123456789012345678901234567890123456789012345678901234.", // 64 digits
    }) |line| try expect(true, &.{line}, null);
}

fn python(code: []const u8) ![]u8 {
    const r = try std.process.run(gpa, std.testing.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", code }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    defer gpa.free(r.stderr);
    errdefer gpa.free(r.stdout);
    try std.testing.expect(r.term.success());
    return r.stdout;
}

/// The engine's runner (runner_open: Dispatch over the bank and circuit) and a routed frontend beside it.
const Engine = struct {
    runner: *ropen.Runner,
    bank: graph.Bank,
    context: @import("runner_frontend.zig").Context,
    job: u64 = 0,

    var one: ?*Engine = null;

    /// One per test binary; null without the circuit, bank and clang (TF_RUNNER_CLANG) this needs.
    fn open() !?*Engine {
        if (one) |e| return e;
        const circuit = std.c.getenv("TF_RUNNER_CIRCUIT") orelse return null;
        const bank = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return null;
        const clang = std.c.getenv("TF_RUNNER_CLANG") orelse return null;
        // a runner lives as long as the engine: never freed, as in compute_circuit_proof_test
        const p = std.heap.page_allocator;
        const e = try p.create(Engine);
        e.* = .{
            .runner = try ropen.open(p, .{ .circuit = std.mem.span(circuit), .backend = .graph, .graph_bank = std.mem.span(bank), .workers = 1 }),
            .bank = try graph.Bank.load(p, std.mem.span(bank)),
            .context = .{ .io = std.testing.io, .compiler = .{ .clang = std.mem.span(clang), .linker = if (std.c.getenv("TF_RUNNER_LLD")) |l| std.mem.span(l) else null } },
        };
        one = e;
        return e;
    }

    /// The trace cell for `run` (a repl cell, no prior) at frontend step `budget`; null: abstained.
    fn translate(e: *Engine, run: []const u8, budget: usize) !?api.Program {
        e.context.compiler.step_budget = budget;
        var routed: route.Routed = .{ .inner = e.context.frontend(), .bank = &e.bank };
        const f = routed.frontend();
        const tr = try f.translate(f.ctx, gpa, .{ .form = .repl, .lang = "python", .code = run });
        return if (tr == .program) tr.program else null;
    }

    fn exec(e: *Engine, p: api.Program) ![]u8 {
        const b = e.runner.backend;
        e.job += 1;
        const job = try b.start(e.job, p, e.runner.cfg.max_steps);
        const res = while (true) switch (b.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        defer b.release(e.job);
        try std.testing.expectEqual(api.Status.ok, res.status);
        return gpa.dupe(u8, res.out);
    }
};

test "runner proof: trace stated: isqrt runs on the bank's plan and prints CPython's root; a mutated input follows (TF_RUNNER_GRAPH_BANK)" {
    const e = try Engine.open() orelse return error.SkipZigTest;
    for ([_][2][]const u8{ .{ live[0].want[1], "98765432109876543211" }, .{ live[1].want[1], "123456789876543210123457" }, .{ live[6].want[1], "" } }) |c| {
        const p = (try e.translate(c[0], 10522)) orelse return error.TestExpectedGraph;
        defer gpa.free(p.bytes);
        try std.testing.expect(p.kind == .graph);
        const want = try python(c[0]);
        defer gpa.free(want);
        const out = try e.exec(p);
        defer gpa.free(out);
        try std.testing.expectEqualStrings(want, out);
        if (c[1].len == 0) continue;
        // the circuit owns the root: the input changed in the packet prints CPython's root of the changed input
        const at = std.mem.indexOf(u8, c[0], "isqrt(").? + 6;
        const n = c[0][at .. c[0].len - 2];
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{n});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{c[1]});
        defer gpa.free(to);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.bytes, from));
        const mutated = try std.mem.replaceOwned(u8, gpa, p.bytes, from, to);
        defer gpa.free(mutated);
        const src = try std.fmt.allocPrint(gpa, "from math import isqrt\nprint(isqrt({s}))", .{c[1]});
        defer gpa.free(src);
        const want2 = try python(src);
        defer gpa.free(want2);
        const out2 = try e.exec(.{ .kind = .graph, .bytes = mutated });
        defer gpa.free(out2);
        try std.testing.expectEqualStrings(want2, out2);
    }
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Deterministic printable strings (no `"` or `\`) for the larger goldens: a small alphabet plus a few edge bytes.
fn text(rand: std.Random, buf: []u8) []const u8 {
    const alphabet = " ~{}'abcdeABC019";
    for (buf) |*c| c.* = alphabet[rand.intRangeLessThan(usize, 0, alphabet.len)];
    return buf;
}

test "runner proof: trace stated: edit distance runs on the bank as a DP packet at the default budget, up to 32x32; a mutated character follows (TF_RUNNER_GRAPH_BANK)" {
    const e = try Engine.open() orelse return error.SkipZigTest;
    const packet = @import("runner_graph_packet.zig");
    const dp = @import("runner_graph_edit.zig");
    var prng = std.Random.DefaultPrng.init(20261103);
    const rand = prng.random();
    var bufs: [7][2][dp.max_len]u8 = undefined;
    const sizes = [_][2]usize{ .{ 1, 1 }, .{ 1, 32 }, .{ 32, 1 }, .{ 17, 31 }, .{ 25, 32 }, .{ 28, 28 }, .{ 32, 32 } };
    const pairs = [_][2][]const u8{ .{ "kitten", "sitting" }, .{ "flaw", "lawn" }, .{ "intention", "execution" }, .{ "algorithm", "logarithm" }, .{ "abc", "yabd" } };
    var cases: [pairs.len + sizes.len][2][]const u8 = undefined;
    @memcpy(cases[0..pairs.len], &pairs);
    for (sizes, 0..) |s, k| cases[pairs.len + k] = .{ text(rand, bufs[k][0][0..s[0]]), text(rand, bufs[k][1][0..s[1]]) };
    // the engine's own frontend (runner_open: routed, default compile budget) and its Dispatch backend
    const f = e.runner.frontend;
    for (cases) |c| {
        const code = try std.fmt.allocPrint(gpa, trace.edit_program, .{ c[0], c[1] });
        defer gpa.free(code);
        const want = try python(code);
        defer gpa.free(want);
        const t0 = nowNs();
        const tr = try f.translate(f.ctx, gpa, .{ .form = .repl, .lang = "python", .code = code });
        const t1 = nowNs();
        if (c[0].len * c[1].len * dp.ops_per_cell > packet.max_ops) {
            // past a packet's ops: never the graph (the RV32 frontend keeps the cell, and refuses this size)
            if (tr == .program) {
                defer gpa.free(tr.program.bytes);
                try std.testing.expect(tr.program.kind != .graph);
            }
            std.debug.print("EDWALL\t{d}x{d}\tover max_ops {d}: not graph\n", .{ c[0].len, c[1].len, packet.max_ops });
            continue;
        }
        if (tr != .program or tr.program.kind != .graph) {
            std.debug.print("not a graph program: {d}x{d}\n", .{ c[0].len, c[1].len });
            return error.TestExpectedGraph;
        }
        defer gpa.free(tr.program.bytes);
        const out = try e.exec(tr.program);
        defer gpa.free(out);
        const t2 = nowNs();
        try std.testing.expectEqualStrings(want, out);
        std.debug.print("EDWALL\t{d}x{d}\ttranslate_ms={d:.1}\trun_ms={d:.1}\tbytes={d}\tout={s}", .{ c[0].len, c[1].len, @as(f64, @floatFromInt(t1 - t0)) / 1e6, @as(f64, @floatFromInt(t2 - t1)) / 1e6, tr.program.bytes.len, out });
        const display = try std.fmt.allocPrint(gpa, "edit_distance(\"{s}\", \"{s}\")", .{ c[0], c[1] });
        defer gpa.free(display);
        const line = (try trace.owned(gpa, "[exact: {}]", display, false, out, false)) orelse return error.TestExpectedOwned;
        defer gpa.free(line);
        const golden = try std.fmt.allocPrint(gpa, "[exact: {s} = {s}]\n", .{ display, std.mem.trimEnd(u8, want, "\n") });
        defer gpa.free(golden);
        try std.testing.expectEqualStrings(golden, line);
        // the circuit compares the characters: the first one changed in the packet prints CPython's distance for it
        if (c[0].len == 0) continue;
        const swap: u8 = if (c[0][0] == 'q') 'r' else 'q';
        const from = try std.fmt.allocPrint(gpa, "\"values\":[\"{d}\"", .{c[0][0]});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"values\":[\"{d}\"", .{swap});
        defer gpa.free(to);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, tr.program.bytes, from));
        const mutated = try std.mem.replaceOwned(u8, gpa, tr.program.bytes, from, to);
        defer gpa.free(mutated);
        const changed = try gpa.dupe(u8, c[0]);
        defer gpa.free(changed);
        changed[0] = swap;
        const src = try std.fmt.allocPrint(gpa, trace.edit_program, .{ changed, c[1] });
        defer gpa.free(src);
        const want2 = try python(src);
        defer gpa.free(want2);
        const out2 = try e.exec(.{ .kind = .graph, .bytes = mutated });
        defer gpa.free(out2);
        try std.testing.expectEqualStrings(want2, out2);
    }
}
