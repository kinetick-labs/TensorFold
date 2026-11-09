const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const packet = @import("runner_graph_packet.zig");
const graph = @import("runner_graph.zig");
const Allocator = std.mem.Allocator;

// Inline notation alone treats caret as power and a separate mod word as remainder.
pub fn normalize(a: Allocator, source: []const u8, inline_form: bool) Allocator.Error!?[]u8 {
    if (source.len == 0 or source.len > 1024 or std.mem.indexOfAny(u8, source, "\r\n\x00") != null) return null;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '^') {
            if (!inline_form) return null;
            try out.appendSlice(a, "**");
        } else if (inline_form and std.mem.startsWith(u8, source[i..], "mod") and
            i > 0 and std.ascii.isWhitespace(source[i - 1]) and
            i + 3 < source.len and std.ascii.isWhitespace(source[i + 3]))
        {
            try out.append(a, '%');
            i += 2;
        } else try out.append(a, source[i]);
    }
    return try out.toOwnedSlice(a);
}

// Proof values bound admission; only literal radix conversion uses them in the packet's inputs.
pub fn lower(a: Allocator, bank: *const graph.Bank, source: []const u8, inline_form: bool) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const text = try normalize(arena, source, inline_form) orelse return null;
    if (inline_form and std.mem.indexOfScalar(u8, source, '%') != null and
        try @import("runner_frontend_integer.zig").ambiguousPercent(arena, text)) return null;
    var p: ast.Parser = .{ .source = text };
    const root = p.parse() catch return null;
    var live: [64]bool = @splat(false);
    live[root] = true;
    var i = p.count;
    while (i > 0) {
        i -= 1;
        if (!live[i]) continue;
        const n = p.nodes[i];
        switch (n.op) {
            .literal => {},
            .neg => live[n.l] = true,
            .add, .sub, .mul, .pow, .powmod => {
                live[n.l] = true;
                live[n.r] = true;
                if (n.op == .powmod) live[n.k] = true;
            },
            else => return null,
        }
    }
    var values: std.ArrayList([]const u8) = .empty;
    var refs: [64]u32 = undefined;
    for (p.nodes[0..p.count], 0..) |n, j| if (live[j] and n.op == .literal) {
        refs[j] = @intCast(values.items.len);
        try values.append(arena, try literalText(arena, n) orelse return null);
    };
    const zero: u32 = @intCast(values.items.len);
    try values.append(arena, "0");
    const one: u32 = @intCast(values.items.len);
    try values.append(arena, "1");
    var ops: std.ArrayList(packet.Op) = .empty;
    var count: u32 = @intCast(values.items.len);
    for (p.nodes[0..p.count], 0..) |n, j| {
        if (!live[j] or n.op == .literal) continue;
        if (n.op == .pow) {
            const e = p.nodes[n.r];
            if (e.op != .literal or e.value < 0 or e.value > 999) return null;
            var exponent: u16 = @intCast(e.value);
            var base = refs[n.l];
            var result = one;
            while (exponent != 0) {
                if (exponent & 1 != 0) result = try operation(arena, &ops, &count, "mul", &.{ result, base });
                exponent >>= 1;
                if (exponent != 0) base = try operation(arena, &ops, &count, "mul", &.{ base, base });
            }
            refs[j] = result;
            continue;
        }
        refs[j] = if (n.op == .powmod)
            try operation(arena, &ops, &count, "powmod", &.{ refs[n.l], refs[n.r], refs[n.k] })
        else if (n.op == .neg)
            try operation(arena, &ops, &count, "sub", &.{ zero, refs[n.l] })
        else
            try operation(arena, &ops, &count, @tagName(n.op), &.{ refs[n.l], refs[n.r] });
    }
    _ = bank.validate(values.items.len, ops.items, refs[root]) catch return null;
    return packet.Builder.encode(a, .{ .values = values.items, .ops = ops.items, .result = refs[root] }) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

fn operation(a: Allocator, ops: *std.ArrayList(packet.Op), count: *u32, name: []const u8, args: []const u32) Allocator.Error!u32 {
    try ops.append(a, .{ .op = name, .width = 62, .args = try a.dupe(u32, args), .checked = true });
    const result = count.*;
    count.* += 1;
    return result;
}

// Bind original literal digits without instantiating a wide-integer output formatter.
fn literalText(a: Allocator, n: ast.Node) Allocator.Error!?[]const u8 {
    var digits: [62]u8 = @splat(0);
    var count: usize = 1;
    for (n.text) |c| {
        var carry: u16 = switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => return null,
        };
        if (carry >= n.base) return null;
        for (digits[0..count]) |*digit| {
            carry += @as(u16, digit.*) * n.base;
            digit.* = @intCast(carry % 10);
            carry /= 10;
        }
        while (carry != 0) {
            if (count == digits.len) return null;
            digits[count] = @intCast(carry % 10);
            carry /= 10;
            count += 1;
        }
    }
    const sign: usize = @intFromBool(n.value < 0);
    const text = try a.alloc(u8, count + sign);
    if (sign != 0) text[0] = '-';
    for (0..count) |i| text[sign + i] = '0' + digits[count - 1 - i];
    return text;
}
