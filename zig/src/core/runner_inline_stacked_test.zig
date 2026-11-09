//! STATED with inline math: a stacked column (`264,278,…` / `+771,565,…` / `= ?`) opens a cell at its lone `=`.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const scan = @import("runner_scan.zig");
const t = std.testing;

const on: scan.Config = .{ .inline_math = true, .trace = true, .trace_whole = true };

/// The reply's lines before the `=` line, the `=` line as the model starts it, the cell code, and CPython's value.
const Golden = struct { []const u8, []const u8, []const u8, []const u8 };

/// W7 (T4 imr-measure-20261103T015417Z) and S2 R5 replies that hand-added these columns, each then wrote `= ?`.
pub const goldens = [_]Golden{
    // a-s2 add30-22
    .{ "We need answer exact integer sum. Need compute carefully. Need add:\n264,278,028,996,235,349,740,985,471,434\n+771,565,915,835,769,397,012,014,492,734\n", "=", "264278028996235349740985471434+771565915835769397012014492734", "1035843944832004746752999964168" },
    // rt-s3 add30-23, a space after the `=`
    .{ "Compute:\n201,353,929,467,531,947,324,387,649,049\n+878,008,121,024,580,933,149,930,071,041\n", "= ", "201353929467531947324387649049+878008121024580933149930071041", "1079362050492112880474317720090" },
    // a-s2 bignum-10
    .{ "Subtract:\n94,514,111,220,054,011,613\n-11,232,705,916,289,774,554\n", "=", "94514111220054011613-11232705916289774554", "83281405303764237059" },
    // R5 g4-hex-a s1, the operand aligned under the first
    .{ "Align:\n12,379,813,733,990,400,000\n+                 4,886,718,345\n", "=", "12379813733990400000+4886718345", "12379813738877118345" },
    .{ "Multiply:\n123,456,789\n× 987,654,321\n", "=", "123456789*987654321", "121932631112635269" },
    .{ "Need exact sum:\n\n522246220506633173639720\n+738473471932250682253300\n", "=", "522246220506633173639720+738473471932250682253300", "1260719692438883855893020" },
};

/// A column the `=` would not sum as one reading, and lines that are not a column.
const refused = [_][2][]const u8{
    .{ "Add:\n111,111\n+222,222\n+333,333\n", "=" }, // three lines: the `=` sums them all, or the last two
    .{ "Add:\n111,111\n222,222\n+333,333\n", "=" },
    .{ "  123\n×  45\n-----\n", "=" }, // a multiplication table's rule
    .{ "Sum:\n264,278\n+771,565\n", "=?" }, // the model asks in the same token
    .{ "Sum:\n264,278\n+771,565\n", "= 1035843" },
    .{ "Sum:\n264,278\n+0,000,771,565\n", "=" }, // a zero-padded group
    .{ "Sum:\n012\n+771,565\n", "=" },
    .{ "Sum:\n1.5\n+2\n", "=" },
    .{ "Sum:\nA\n+B\n", "=" },
    .{ "Sum:\n-12\n+5\n", "=" },
    .{ "Sum:\n123\nx 45\n", "=" }, // `x` may be a name
    .{ "Pow:\n2\n**64\n", "=" },
    .{ "Sum:\n12,3456\n+1\n", "=" },
    .{ "Sum:\n264,278\n+771,565 maybe\n", "=" },
    .{ "Sum: 264,278\n+771,565\n", "=" },
    .{ "Sum:\n264,278\n+771,565\nthen\n", "=" },
};

/// The inline cell the `=` line opens after `lines`, fed a line at a time (null: none).
fn cellOf(a: std.mem.Allocator, lines: []const u8, eq: []const u8, cfg: scan.Config) !?scan.Closed {
    var s: scan.Scanner = .{ .cfg = cfg };
    defer s.deinit(a);
    var it = std.mem.splitScalar(u8, lines[0 .. lines.len - 1], '\n');
    while (it.next()) |l| {
        _ = try s.feed(a, l, true);
        _ = try s.feed(a, "\n", true);
    }
    const e = try s.feed(a, eq, true);
    const c = e.closed orelse return null;
    if (c.form != .inline_expr or e.trigger == null or e.trigger.?.inline_space == null) return null;
    return .{ .index = c.index, .form = c.form, .lang = "", .code = try a.dupe(u8, c.code), .own_context = c.own_context };
}

test "runner scan: STATED reads a stacked column at its lone `=`; a longer column, a table or a non-literal refuses" {
    for (goldens) |g| {
        const c = try cellOf(t.allocator, g[0], g[1], on) orelse {
            std.debug.print("no cell: {s}{s}\n", .{ g[0], g[1] });
            return error.TestExpectedInline;
        };
        defer t.allocator.free(c.code);
        try t.expectEqualStrings(g[2], c.code);
        try t.expect(!c.own_context);
    }
    for (refused) |r| if (try cellOf(t.allocator, r[0], r[1], on)) |c| {
        defer t.allocator.free(c.code);
        std.debug.print("opened, must not: {s}{s} -> {s}\n", .{ r[0], r[1], c.code });
        return error.TestUnexpectedInline;
    };
    // after a join cue the two lines may be digit groups: the cell keeps the model's value
    const j = try cellOf(t.allocator, "Concatenate the groups:\n1119\n+392\n", "=", on) orelse return error.TestExpectedInline;
    defer t.allocator.free(j.code);
    try t.expect(j.own_context);
}

test "runner scan: a stacked column opens nothing without STATED, or with STATED but no inline math" {
    for (goldens) |g| {
        try t.expectEqual(@as(?scan.Closed, null), try cellOf(t.allocator, g[0], g[1], .{ .inline_math = true, .trace = true }));
        try t.expectEqual(@as(?scan.Closed, null), try cellOf(t.allocator, g[0], g[1], .{ .trace = true, .trace_whole = true }));
    }
    try t.expect(!(scan.Config{}).trace_whole);
}

fn bankPath() ![]const u8 {
    const p = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return std.mem.span(p);
}

fn run(bank: *graph.Bank, bytes: []const u8) ![]u8 {
    var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, bank, bytes);
    defer machine.deinit();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(t.allocator);
    const end = try machine.run(4_000_000, &out, null);
    try t.expect(end.end == .halted);
    return out.toOwnedSlice(t.allocator);
}

test "runner proof: stacked columns run on the graph bank and print CPython's value; a mutated operand prints its own (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-inline-stacked-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for (goldens, 0..) |g, k| {
        const c = try cellOf(t.allocator, g[0], g[1], on) orelse return error.TestExpectedInline;
        defer t.allocator.free(c.code);
        const result = try f.translate(f.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = c.code });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        if (result != .program or result.program.kind != .graph) {
            std.debug.print("not on the graph: {s}: {any}\n", .{ c.code, result });
            return error.TestExpectedGraph;
        }
        const out = try run(&bank, result.program.bytes);
        defer t.allocator.free(out);
        std.testing.expectEqualStrings(g[3], std.mem.trimEnd(u8, out, "\n")) catch |e| {
            std.debug.print("stacked value: {s}\n", .{c.code});
            return e;
        };
        if (k != 0) continue;
        // the bank computes the column: one operand changed in the packet prints CPython's value of the new sum
        const from = "\"771565915835769397012014492734\"";
        if (std.mem.count(u8, result.program.bytes, from) != 1) {
            std.debug.print("operand not one packet value: {s}\n", .{result.program.bytes});
            return error.TestExpectedInput;
        }
        const mutated = try std.mem.replaceOwned(u8, t.allocator, result.program.bytes, from, "\"771565915835769397012014492735\"");
        defer t.allocator.free(mutated);
        const got = try run(&bank, mutated);
        defer t.allocator.free(got);
        try t.expectEqualStrings("1035843944832004746752999964169", std.mem.trimEnd(u8, got, "\n"));
    }
}
