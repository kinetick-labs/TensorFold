//! Bare hex literal cells (S1 round 4 d_mirror hex-a and hex): the scanner closes `0x… =`; the bank does the radix.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const scan = @import("runner_scan.zig");
const t = std.testing;

/// The inline cell `text` closes on one scanner has code `want` (null: no inline cell).
fn closes(cfg: scan.Config, text: []const u8, want: ?[]const u8) !void {
    var s: scan.Scanner = .{ .cfg = cfg };
    defer s.deinit(t.allocator);
    const e = try s.feed(t.allocator, text, true);
    const got: ?[]const u8 = if (e.closed) |c| (if (c.form == .inline_expr and e.trigger != null) c.code else null) else null;
    if (want == null and got != null) std.debug.print("hex cell closed: {s} -> {s}\n", .{ text, got.? });
    if (want) |w| try t.expectEqualStrings(w, got orelse "<none>") else try t.expect(got == null);
}

test "runner scan: a bare `0x…` literal ending its line at `=` is a whole inline cell; operands and glued words are not" {
    const on: scan.Config = .{ .inline_math = true };
    // live R4 lines, then partial conversions the same replies wrote
    try closes(on, "Known decimal? 0xABCDEF0123456789 =", "0xABCDEF0123456789");
    try closes(on, "I recall 0xFEDCBA9876543210 =", "0xFEDCBA9876543210");
    try closes(on, "Compute using Python mentally or derive. 0xFFFFFFFFFFFFFFFF =", "0xFFFFFFFFFFFFFFFF");
    try closes(on, "Low 0x23456789. Compute: 0x23=", "0x23");
    try closes(on, "0xabcdef01 = ", "0xabcdef01");
    for ([_][]const u8{
        "Hex = 0x123456789ABCDEF012345 =", "#0xFF =",        "-0xFF =",     "x0xFF =",    "0xF =",         "0x =",
        "0xFG =",                          "200 mod 0xFF =", "17 % 0x10 =", "0x1F + 1 =", "0xFF == 255 =",
    }) |line| try closes(on, line, null);
    // an operand of the expression the model is closing: the replay's live 0633, 0492 and 0506 lines
    for ([_][]const u8{
        "Let's add: 0xFFFFFFFFFFFFFFFF - 0xFEDCBA9876543210 =",
        "[exact: 0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF =",
        "Add 0xFF =",
        "2 × 0xFF =",
        "(0xFF =",
    }) |line| try closes(on, line, null);
    try closes(.{ .inline_math = true, .inline_arith = false }, "I recall 0xFEDCBA9876543210 =", null);
}

fn bankPath() ![]const u8 {
    return std.mem.span(std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest);
}

fn python(code: []const u8) ![]u8 {
    const r = try std.process.run(t.allocator, t.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", code }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    defer t.allocator.free(r.stderr);
    errdefer t.allocator.free(r.stdout);
    try t.expect(r.term.success());
    return r.stdout;
}

fn run(bank: *const graph.Bank, bytes: []const u8) ![]u8 {
    var m = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, bank, bytes);
    defer m.deinit();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(t.allocator);
    const end = try m.run(4_000_000, &out, null);
    try t.expect(end.end == .halted);
    return out.toOwnedSlice(t.allocator);
}

const ff64 = "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

test "runner proof: a bare hex literal cell runs on the bank and prints CPython's int (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-hex-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for ([_][]const u8{ "0xFEDCBA9876543210", "0xABCDEF0123456789", "0x123456789ABCDEF012345", "0xFFFFFFFFFFFFFFFF", "0x23", "0xabcdef01", ff64 }) |hex| {
        const code = try std.fmt.allocPrint(t.allocator, "print({s})", .{hex});
        defer t.allocator.free(code);
        const want = try python(code);
        defer t.allocator.free(want);
        for ([_]api.Form{ .inline_expr, .repl }) |form| {
            // the route's own Horner program, not T1's binder (which would read the literal on the host)
            const own = try route.hexLiteral(t.allocator, &bank, .{ .form = form, .lang = "python", .code = hex });
            defer if (own) |b| t.allocator.free(b);
            try t.expect(own != null);
            const r = try f.translate(f.ctx, t.allocator, .{ .form = form, .lang = "python", .code = hex });
            defer if (r == .program) t.allocator.free(r.program.bytes);
            try t.expect(r == .program and r.program.kind == .graph);
            try t.expectEqualStrings(own.?, r.program.bytes);
            const out = try run(&bank, r.program.bytes);
            defer t.allocator.free(out);
            try t.expectEqualStrings(want, out);
        }
    }
    for ([_]api.Cell{
        .{ .form = .inline_expr, .lang = "python", .code = "0x" },
        .{ .form = .inline_expr, .lang = "python", .code = "0x7" },
        .{ .form = .inline_expr, .lang = "python", .code = "0xG1" },
        .{ .form = .inline_expr, .lang = "python", .code = ">>> 0xff" },
        .{ .form = .inline_expr, .lang = "python", .code = "0x1F+1" },
        .{ .form = .inline_expr, .lang = "python", .code = "0x11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" },
        .{ .form = .fence, .lang = "python", .code = "0xff" },
        .{ .form = .inline_expr, .lang = "c", .code = "0xff" },
    }) |c| try t.expectEqual(@as(?[]u8, null), try route.hexLiteral(t.allocator, &bank, c));
}

test "runner proof: the circuit owns the radix: a mutated digit or base input prints CPython's mutated int (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    const bytes = (try route.hexLiteral(t.allocator, &bank, .{ .form = .inline_expr, .lang = "python", .code = "0xFEDCBA9876543210" })).?;
    defer t.allocator.free(bytes);
    const digits = "[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]";
    for ([_][3][]const u8{
        .{ "\"13\"", "\"4\"", "print(0xFE4CBA9876543210)" }, // the digit D read as 4
        .{ "\"16\"", "\"10\"", "a = 0\nfor d in " ++ digits ++ ":\n    a = a * 10 + d\nprint(a)" }, // the base read as 10
    }) |m| {
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, m[0]));
        const mutated = try std.mem.replaceOwned(u8, t.allocator, bytes, m[0], m[1]);
        defer t.allocator.free(mutated);
        const want = try python(m[2]);
        defer t.allocator.free(want);
        const out = try run(&bank, mutated);
        defer t.allocator.free(out);
        if (!std.mem.eql(u8, want, out)) std.debug.print("host-owned: {s} -> {s}: '{s}', CPython '{s}'\n", .{ m[0], m[1], out, want });
        try t.expectEqualStrings(want, out);
    }
}
