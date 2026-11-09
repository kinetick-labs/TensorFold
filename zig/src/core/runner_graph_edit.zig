//! A literal edit-distance cell as one graph packet: the Levenshtein DP on the bank's compare, min, add and select.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const packet = @import("runner_graph_packet.zig");
const Allocator = std.mem.Allocator;

/// Levenshtein distance as the trace states it (T5's census program); `{s}` are the two strings, CPython checks it.
pub const program =
    \\def edit_distance(a,b):
    \\    if len(a) > len(b):
    \\        a,b = b,a
    \\    m = len(a)
    \\    dp = list(range(m+1))
    \\    for i, ch in enumerate(b, 1):
    \\        prev = dp[0]
    \\        dp[0] = i
    \\        for j in range(1, m+1):
    \\            old = dp[j]
    \\            cost = 0 if a[j-1] == ch else 1
    \\            dp[j] = min(old + 1, dp[j-1] + 1, prev + cost)
    \\            prev = old
    \\    return dp[m]
    \\print(edit_distance("{s}", "{s}"))
;

/// The longest string on either side.
pub const max_len = 32;
/// Graph ops per DP cell: the compare, two mins, the +1 and the select.
pub const ops_per_cell = 5;

const first = std.mem.indexOf(u8, program, "{s}").?;
const second = std.mem.lastIndexOf(u8, program, "{s}").?;
const head = program[0..first];
const middle = program[first + 3 .. second];
const tail = program[second + 3 ..];

/// The two strings when `code` is `program` over them: each at most max_len printable ASCII bytes, no `"` or `\`.
pub fn parse(code: []const u8) ?[2][]const u8 {
    if (code.len < head.len + middle.len + tail.len or !std.mem.startsWith(u8, code, head) or !std.mem.endsWith(u8, code, tail))
        return null;
    const inner = code[head.len .. code.len - tail.len];
    const end = std.mem.indexOfScalar(u8, inner, '"') orelse return null;
    if (!std.mem.startsWith(u8, inner[end..], middle)) return null;
    const pair: [2][]const u8 = .{ inner[0..end], inner[end + middle.len ..] };
    for (pair) |s| {
        if (s.len > max_len) return null;
        for (s) |c| if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') return null;
    }
    return pair;
}

/// The DP row by row, each character a packet value so the circuit compares; null: other code or too large.
pub fn cell(a: Allocator, bank: *const graph.Bank, c: api.Cell) Allocator.Error!?[]u8 {
    if ((c.form != .repl and c.form != .fence) or c.prior.len != 0 or !std.mem.eql(u8, c.lang, "python")) return null;
    const pair = parse(std.mem.trim(u8, c.code, "\n")) orelse return null;
    const x, const y = pair;
    if (x.len * y.len * ops_per_cell > packet.max_ops) return null;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var values: std.ArrayList([]const u8) = .empty;
    for (x) |ch| try values.append(arena, try std.fmt.allocPrint(arena, "{d}", .{ch}));
    for (y) |ch| try values.append(arena, try std.fmt.allocPrint(arena, "{d}", .{ch}));
    // the row and column labels 0..max(len), and 1 for the +1
    const labels: u32 = @intCast(values.items.len);
    for (0..@max(x.len, y.len, 1) + 1) |k| try values.append(arena, try std.fmt.allocPrint(arena, "{d}", .{k}));
    const one = labels + 1;
    // character codes are below 127 (width 4); a distance and its +1 at most 33 (width 2)
    const cw = 4;
    const dw = 2;
    var ops: std.ArrayList(packet.Op) = .empty;
    var up: [max_len + 1]u32 = undefined;
    var row: [max_len + 1]u32 = undefined;
    for (0..y.len + 1) |j| up[j] = labels + @as(u32, @intCast(j));
    for (0..x.len) |i| {
        row[0] = labels + @as(u32, @intCast(i + 1));
        for (0..y.len) |j| {
            const at: u32 = @intCast(values.items.len + ops.items.len);
            try ops.append(arena, .{ .op = "compare", .width = cw, .a = @intCast(i), .b = @intCast(x.len + j), .checked = true });
            try ops.append(arena, .{ .op = "min", .width = dw, .a = up[j + 1], .b = row[j], .checked = true });
            try ops.append(arena, .{ .op = "min", .width = dw, .a = at + 1, .b = up[j], .checked = true });
            try ops.append(arena, .{ .op = "add", .width = dw, .a = at + 2, .b = one, .checked = true });
            // neighbours differ by at most 1, so equal characters give the diagonal: select's test is compare's digit
            try ops.append(arena, .{ .op = "select", .width = dw, .args = try arena.dupe(u32, &.{ at + 3, up[j], at }), .checked = true });
            row[j + 1] = at + 4;
        }
        @memcpy(up[0 .. y.len + 1], row[0 .. y.len + 1]);
    }
    const result = up[y.len];
    _ = bank.validate(values.items.len, ops.items, result) catch return null;
    return packet.Builder.program(a, values.items, ops.items, &.{.{ .ref = result }}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
}

test "runner graph edit: the trace's program over two literals parses; anything else refuses" {
    var buf: [2048]u8 = undefined;
    const ok = try std.fmt.bufPrint(&buf, program, .{ "kitten", "sitting" });
    const got = parse(ok).?;
    try std.testing.expectEqualStrings("kitten", got[0]);
    try std.testing.expectEqualStrings("sitting", got[1]);
    const empty = parse(try std.fmt.bufPrint(&buf, program, .{ "", "abc" })).?;
    try std.testing.expectEqualStrings("", empty[0]);
    const long: [max_len]u8 = @splat('z');
    try std.testing.expect(parse(try std.fmt.bufPrint(&buf, program, .{ &long, &long })) != null);
    var refuse: std.ArrayList([]const u8) = .empty;
    defer {
        for (refuse.items) |r| std.testing.allocator.free(r);
        refuse.deinit(std.testing.allocator);
    }
    const g = std.testing.allocator;
    const too_long: [max_len + 1]u8 = @splat('z');
    try refuse.append(g, try std.fmt.allocPrint(g, program, .{ &too_long, "a" }));
    try refuse.append(g, try std.fmt.allocPrint(g, program, .{ "a\\nb", "a" }));
    try refuse.append(g, try std.fmt.allocPrint(g, program, .{ "caf\xc3\xa9", "cafe" }));
    try refuse.append(g, try std.fmt.allocPrint(g, program, .{ "a\tb", "ab" }));
    try refuse.append(g, try std.fmt.allocPrint(g, program, .{ "a\", \"b", "c" })); // a third argument
    // the body changed: a substitution costs 2, so this is not Levenshtein
    const twice = try std.fmt.allocPrint(g, program, .{ "kitten", "sitting" });
    defer g.free(twice);
    const at = std.mem.indexOf(u8, twice, "else 1").?;
    try refuse.append(g, try std.mem.concat(g, u8, &.{ twice[0..at], "else 2", twice[at + 6 ..] }));
    try refuse.append(g, try std.mem.concat(g, u8, &.{ twice, "\nprint(1)" }));
    try refuse.append(g, try g.dupe(u8, "print(edit_distance(\"kitten\", \"sitting\"))"));
    for (refuse.items) |r| if (parse(r)) |p| {
        std.debug.print("parsed {s} / {s} from:\n{s}\n", .{ p[0], p[1], r });
        return error.TestUnexpectedParse;
    };
}
