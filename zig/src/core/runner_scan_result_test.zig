const std = @import("std");
const scan = @import("runner_scan.zig");
const t = std.testing;
const Chunk = struct { id: u32, bytes: []const u8, think: bool };

const gcd = [_]Chunk{
    .{ .id = 9854, .bytes = "```", .think = false },
    .{ .id = 36490, .bytes = "python", .think = false },
    .{ .id = 201, .bytes = "\n", .think = false },
    .{ .id = 1897, .bytes = "import", .think = false },
    .{ .id = 7704, .bytes = " math", .think = false },
    .{ .id = 201, .bytes = "\n", .think = false },
    .{ .id = 67, .bytes = "a", .think = false },
    .{ .id = 31, .bytes = "=", .think = false },
    .{ .id = 6895, .bytes = "123", .think = false },
    .{ .id = 18009, .bytes = "456", .think = false },
    .{ .id = 25744, .bytes = "789", .think = false },
    .{ .id = 16993, .bytes = "012", .think = false },
    .{ .id = 18014, .bytes = "345", .think = false },
    .{ .id = 27183, .bytes = "678", .think = false },
    .{ .id = 2225, .bytes = "90", .think = false },
    .{ .id = 201, .bytes = "\n", .think = false },
    .{ .id = 68, .bytes = "b", .think = false },
    .{ .id = 31, .bytes = "=", .think = false },
    .{ .id = 30515, .bytes = "987", .think = false },
    .{ .id = 27395, .bytes = "654", .think = false },
    .{ .id = 17561, .bytes = "321", .think = false },
    .{ .id = 36458, .bytes = "098", .think = false },
    .{ .id = 28782, .bytes = "765", .think = false },
    .{ .id = 20208, .bytes = "432", .think = false },
    .{ .id = 553, .bytes = "10", .think = false },
    .{ .id = 201, .bytes = "\n", .think = false },
    .{ .id = 3098, .bytes = "print", .think = false },
    .{ .id = 127637, .bytes = "(math", .think = false },
    .{ .id = 3588, .bytes = ".g", .think = false },
    .{ .id = 6621, .bytes = "cd", .think = false },
    .{ .id = 6036, .bytes = "(a", .think = false },
    .{ .id = 16181, .bytes = ",b", .think = false },
    .{ .id = 5203, .bytes = "))\n", .think = false },
    .{ .id = 20759, .bytes = "```\n\n", .think = false },
    .{ .id = 10610, .bytes = "Result", .think = false },
    .{ .id = 28, .bytes = ":", .think = false },
    .{ .id = 223, .bytes = " ", .think = false },
    .{ .id = 553, .bytes = "10", .think = false },
    .{ .id = 271, .bytes = "\n\n", .think = false },
    .{ .id = 40, .bytes = "F", .think = false },
    .{ .id = 66393, .bytes = "INAL", .think = false },
    .{ .id = 65, .bytes = "_", .think = false },
};

const modpow = [_]Chunk{
    .{ .id = 9854, .bytes = "```", .think = false },
    .{ .id = 36490, .bytes = "python", .think = false },
    .{ .id = 201, .bytes = "\n", .think = false },
    .{ .id = 3098, .bytes = "print", .think = false },
    .{ .id = 7093, .bytes = "(p", .think = false },
    .{ .id = 359, .bytes = "ow", .think = false },
    .{ .id = 10, .bytes = "(", .think = false },
    .{ .id = 25, .bytes = "7", .think = false },
    .{ .id = 14, .bytes = ",", .think = false },
    .{ .id = 223, .bytes = " ", .think = false },
    .{ .id = 6131, .bytes = "83", .think = false },
    .{ .id = 14, .bytes = ",", .think = false },
    .{ .id = 223, .bytes = " ", .think = false },
    .{ .id = 1457, .bytes = "100", .think = false },
    .{ .id = 1320, .bytes = "000", .think = false },
    .{ .id = 1320, .bytes = "000", .think = false },
    .{ .id = 25, .bytes = "7", .think = false },
    .{ .id = 5203, .bytes = "))\n", .think = false },
    .{ .id = 20759, .bytes = "```\n\n", .think = false },
    .{ .id = 10610, .bytes = "Result", .think = false },
    .{ .id = 28, .bytes = ":", .think = false },
    .{ .id = 223, .bytes = " ", .think = false },
    .{ .id = 19913, .bytes = "640", .think = false },
    .{ .id = 22024, .bytes = "025", .think = false },
    .{ .id = 29438, .bytes = "692", .think = false },
    .{ .id = 271, .bytes = "\n\n", .think = false },
    .{ .id = 40, .bytes = "F", .think = false },
};

test "runner scan: exact S1 gcd and modpow token boundaries trigger before the guessed result" {
    for ([_][]const Chunk{ &gcd, &modpow }) |chunks| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        var closes: usize = 0;
        var fires: usize = 0;
        for (chunks) |chunk| {
            const event = try s.feed(t.allocator, chunk.bytes, chunk.think);
            if (event.closed) |cell| {
                closes += 1;
                try t.expectEqualStrings("python", cell.lang);
                try t.expectEqual(@as(u32, 0), cell.index);
            }
            if (event.trigger) |trigger| {
                fires += 1;
                try t.expectEqual(@as(u32, 28), chunk.id);
                try t.expectEqual(@as(u32, 0), trigger.index);
                try t.expect(!trigger.fenced and trigger.inline_space == null and !trigger.trace);
            }
        }
        try t.expectEqual(@as(usize, 1), closes);
        try t.expectEqual(@as(usize, 1), fires);
    }
}

test "runner scan: complete result headers require a token boundary and pending cell" {
    for ([_][]const u8{ "Result: 10", "Result: we need O(n) ", "A Result: ", "Result : 10 ", "Result:\n10" }) |text| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        _ = try s.feed(t.allocator, "```python\nprint(1)\n```\n", false);
        try t.expect((try s.feed(t.allocator, text, false)).trigger == null);
    }
    for ([_][]const u8{ "Result: ", "**Output:**\t", "  Running this gives: " }) |text| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        try t.expect((try s.feed(t.allocator, text, false)).trigger == null);
        _ = try s.feed(t.allocator, "\n```python\nprint(1)\n```\n", false);
        try t.expect((try s.feed(t.allocator, text, false)).trigger != null);
    }
    var s: scan.Scanner = .{ .cfg = .{ .reasoning_only = true } };
    defer s.deinit(t.allocator);
    _ = try s.feed(t.allocator, "```python\nprint(1)\n```\n", true);
    try t.expect((try s.feed(t.allocator, "Result: ", false)).trigger == null);
}

test "runner scan: terminal closed fences finish only at an explicit EOS" {
    for ([_][]const u8{ "```python\nprint(2)\n```", "```python\nprint(2)\n```\n", "```python\nprint(2)\n```\n \t" }) |text| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        var closed: usize = 0;
        for (text) |byte| {
            const event = try s.feed(t.allocator, &.{byte}, false);
            closed += @intFromBool(event.closed != null);
            try t.expect(event.trigger == null);
        }
        try t.expect((try s.feed(t.allocator, "", false)).trigger == null);
        const event = try s.finishEos(t.allocator, false);
        closed += @intFromBool(event.closed != null);
        try t.expectEqual(@as(usize, 1), closed);
        try t.expect(event.trigger != null and event.trigger.?.terminal);
        try t.expectEqual(@as(u32, 0), event.trigger.?.index);
        try t.expect((try s.finishEos(t.allocator, false)).trigger == null);
    }
}

test "runner scan: EOF declines unfinished fences stale cells prose and excluded spans" {
    for ([_][]const u8{
        "```python\nprint(2)",
        "```python\nprint(2)\n```\nSome explanation.\n",
        "```python\nprint(2)\n```\n42",
        "There is a Result:",
        "```python\n# Result:\n",
    }) |text| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        _ = try s.feed(t.allocator, text, false);
        try t.expect((try s.finishEos(t.allocator, false)).trigger == null);
    }
    var s: scan.Scanner = .{ .cfg = .{ .reasoning_only = true } };
    defer s.deinit(t.allocator);
    _ = try s.feed(t.allocator, "```python\nprint(2)\n```", false);
    try t.expect((try s.finishEos(t.allocator, false)).trigger == null);
}

test "runner scan: a result label in prose or inside code never fires" {
    for ([_][]const u8{ "Prose with Result:", "`Result:`", "The explanation mentions Result:" }) |text| {
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        _ = try s.feed(t.allocator, "```python\nprint(2)\n```\n", false);
        try t.expect((try s.feed(t.allocator, text, false)).trigger == null);
    }
    var s: scan.Scanner = .{ .cfg = .{} };
    defer s.deinit(t.allocator);
    _ = try s.feed(t.allocator, "```python\n", false);
    try t.expect((try s.feed(t.allocator, "Result:", false)).trigger == null);
}

test "runner scan: five T4 native token streams hand off before values or at terminal EOS" {
    const Native = struct {
        key: []const u8,
        original_sha256_match: bool,
        chunks: []const struct { completion_index: usize, token_id: u32, bytes_hex: []const u8, think_open: bool },
    };
    const parsed = try std.json.parseFromSlice([]const Native, t.allocator, @embedFile("fixtures/runner/t4-handoff.json"), .{});
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 5), parsed.value.len);
    const expected = [_][]const usize{ &.{ 902, 944 }, &.{ 430, 467 }, &.{113}, &.{74}, &.{127} };
    for (parsed.value, 0..) |row, i| {
        try t.expect(row.original_sha256_match);
        var s: scan.Scanner = .{ .cfg = .{} };
        defer s.deinit(t.allocator);
        var fires: usize = 0;
        var was_thinking = true;
        for (row.chunks, 0..) |chunk, index| {
            try t.expectEqual(index, chunk.completion_index);
            const ended = was_thinking and !chunk.think_open;
            was_thinking = chunk.think_open;
            const bytes = try t.allocator.alloc(u8, chunk.bytes_hex.len / 2);
            defer t.allocator.free(bytes);
            _ = try std.fmt.hexToBytes(bytes, chunk.bytes_hex);
            const event = if (chunk.token_id == 1)
                try s.finishEos(t.allocator, chunk.think_open)
            else
                try s.feed(t.allocator, if (ended) "" else bytes, chunk.think_open);
            if (event.trigger) |trigger| {
                try t.expect(fires < expected[i].len);
                try t.expectEqual(expected[i][fires], index);
                fires += 1;
                if (i < 3) {
                    try t.expect(!trigger.terminal and trigger.newline_before);
                    try t.expectEqualStrings(":", bytes);
                } else {
                    try t.expect(trigger.terminal);
                    try t.expectEqual(@as(u32, 1), chunk.token_id);
                }
            }
        }
        try t.expectEqual(expected[i].len, fires);
    }
}
