//! Form trace's `divmod(a, b)` cells (S1 R4) on the bank, run as the engine runs packets and sequences (Dispatch).
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const trace = @import("runner_trace.zig");
const gpa = std.testing.allocator;

/// The R4 cells the stated audit ran on a bare graph backend, which takes no sequence (all 13 distinct ones).
const cells = [_][]const u8{
    "divmod(1010101010101018, 112233445566770)",
    "divmod(10**24, 97)",
    "divmod(10**30-1, 97)",
    "divmod(123456789012345, 8913585)",
    "divmod(23777868243178464, 41832496)",
    "divmod(314159265358979323846264, 271828183)",
    "divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)",
    "divmod(32685677580, -4826)",
    "divmod(33311225865831424, 998244353)",
    "divmod(40092776, 233)",
    "divmod(45949729863572161, 1000000007)",
    "divmod(4852937748780900, 1000000007)",
    "divmod(987654321098765432109876, 1000003)",
};

/// The circuit: no divmod cell may reach it.
const Never = struct {
    fn backend() api.Backend {
        return .{ .ctx = undefined, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(_: *anyopaque, _: u64, _: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
        return error.Busy;
    }
    fn poll(_: *anyopaque, _: api.JobId) api.Poll {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
};

fn python(code: []const u8) ![]u8 {
    const src = try std.fmt.allocPrint(gpa, "print({s})", .{code});
    defer gpa.free(src);
    const r = try std.process.run(gpa, std.testing.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", src }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    defer gpa.free(r.stderr);
    errdefer gpa.free(r.stdout);
    try std.testing.expect(r.term.success());
    return r.stdout;
}

const Bank = struct {
    bank: graph.Bank,
    graph: api.Backend,
    dispatch: route.Dispatch,
    job: u64 = 0,

    /// The cell as trace hands it to the runner (a repl line, no prior); it must stay on the graph backend.
    fn translate(b: *Bank, code: []const u8) ![]const u8 {
        var routed: route.Routed = .{ .inner = @import("runner_frontend.zig").frontend(), .bank = &b.bank };
        const f = routed.frontend();
        const tr = try f.translate(f.ctx, gpa, .{ .form = .repl, .lang = "python", .code = code });
        if (tr != .program or tr.program.kind != .graph) {
            std.debug.print("not a bank program: {s}\n", .{code});
            if (tr == .program) gpa.free(tr.program.bytes);
            return error.TestExpectedGraph;
        }
        return tr.program.bytes;
    }

    fn run(b: *Bank, bytes: []const u8) ![]u8 {
        const d = b.dispatch.backend();
        b.job += 1;
        const job = try d.start(b.job, .{ .kind = .graph, .bytes = bytes }, 1 << 32);
        const res = while (true) switch (d.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        defer d.release(b.job);
        try std.testing.expectEqual(api.Status.ok, res.status);
        return gpa.dupe(u8, res.out);
    }
};

fn openBank() !?*Bank {
    const path = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return null;
    const b = try gpa.create(Bank);
    errdefer gpa.destroy(b);
    b.* = .{ .bank = try graph.Bank.load(gpa, std.mem.span(path)), .graph = undefined, .dispatch = undefined };
    b.graph = try graph.backend(gpa, &b.bank, 2);
    b.dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = b.graph, .bank = &b.bank };
    return b;
}

fn closeBank(b: *Bank) void {
    b.dispatch.deinit();
    graph.destroy(b.graph);
    b.bank.deinit();
    gpa.destroy(b);
}

/// CPython's `(q, r)` as trace's owned line in the `[exact: {}]` format (pinned: the default wording may change).
fn ownedLine(code: []const u8, tuple: []const u8) ![]u8 {
    const v = std.mem.trimEnd(u8, tuple, "\n");
    const comma = std.mem.indexOf(u8, v, ", ").?;
    return std.fmt.allocPrint(gpa, "[exact: {s} = {s} remainder {s}]\n", .{ code, v[1..comma], v[comma + 2 .. v.len - 1] });
}

test "runner proof: trace divmod: each R4 divmod cell runs on the bank and owns CPython's quotient and remainder (TF_RUNNER_GRAPH_BANK)" {
    const b = try openBank() orelse return error.SkipZigTest;
    defer closeBank(b);
    for (cells) |code| {
        const want = try python(code);
        defer gpa.free(want);
        const bytes = try b.translate(code);
        defer gpa.free(bytes);
        const out = try b.run(bytes);
        defer gpa.free(out);
        if (!std.mem.eql(u8, want, out)) std.debug.print("{s}: '{s}', CPython '{s}'\n", .{ code, out, want });
        try std.testing.expectEqualStrings(want, out);
        const line = (try trace.owned(gpa, "[exact: {}]", code, true, out, false)) orelse return error.TestExpectedOwned;
        defer gpa.free(line);
        const golden = try ownedLine(code, want);
        defer gpa.free(golden);
        try std.testing.expectEqualStrings(golden, line);
    }
}

test "runner proof: trace divmod: the circuit owns both halves: a mutated dividend prints CPython's pair (TF_RUNNER_GRAPH_BANK)" {
    const b = try openBank() orelse return error.SkipZigTest;
    defer closeBank(b);
    for ([_][4][]const u8{
        .{ cells[5], "314159265358979323846264", "314159265358979323846265", "divmod(314159265358979323846265, 271828183)" },
        .{ cells[7], "32685677580", "32685677581", "divmod(32685677581, -4826)" },
    }) |m| {
        const bytes = try b.translate(m[0]);
        defer gpa.free(bytes);
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[1]});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[2]});
        defer gpa.free(to);
        // A single divmod packet binds once for both outputs; split sequences may bind it in both pieces.
        if (std.mem.count(u8, bytes, from) < 1) {
            std.debug.print("dividend {s} not bound in the graph program: {s}\n", .{ m[1], bytes });
            return error.TestExpectedInput;
        }
        const mutated = try std.mem.replaceOwned(u8, gpa, bytes, from, to);
        defer gpa.free(mutated);
        const want = try python(m[3]);
        defer gpa.free(want);
        const out = try b.run(mutated);
        defer gpa.free(out);
        if (!std.mem.eql(u8, want, out)) std.debug.print("host-owned: {s} -> {s}: '{s}', CPython '{s}'\n", .{ m[1], m[2], out, want });
        try std.testing.expectEqualStrings(want, out);
    }
}
