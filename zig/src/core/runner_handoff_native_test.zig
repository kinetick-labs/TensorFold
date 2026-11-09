//! Live handoff-rg tokens, including the structural think end and terminal EOS, through the real stream hook.
const std = @import("std");
const lanes = @import("lanes");
const scan = @import("runner_scan.zig");
const a = std.testing.allocator;
const t = std.testing;
const Native = struct {
    key: []const u8,
    original_sha256_match: bool,
    sha256: []const u8,
    chunks: []const struct { completion_index: usize, token_id: u32, bytes_hex: []const u8, think_open: bool },
};

const Replay = struct {
    scanner: scan.Scanner = .{ .cfg = .{} },
    bytes: []const u8 = "",
    closed: usize = 0,
    fired: usize = 0,
    language: std.ArrayList(u8) = .empty,
    source: std.ArrayList(u8) = .empty,
    fn event(r: *@This(), e: scan.Events) !bool {
        if (e.closed) |c| {
            r.closed += 1;
            r.language.clearRetainingCapacity();
            try r.language.appendSlice(a, c.lang);
            r.source.clearRetainingCapacity();
            try r.source.appendSlice(a, c.code);
        }
        r.fired += @intFromBool(e.trigger != null);
        return e.trigger != null;
    }
    fn token(ptr: *anyopaque, _: u32, thinking: bool, _: bool, ended: bool) bool {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        _ = r.event(r.scanner.feed(a, if (ended) "" else r.bytes, thinking) catch unreachable) catch unreachable;
        return false;
    }
    fn eos(ptr: *anyopaque, thinking: bool, _: bool) bool {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        return r.event(r.scanner.finishEos(a, thinking) catch unreachable) catch unreachable;
    }
    fn deinit(r: *@This()) void {
        r.scanner.deinit(a);
        r.language.deinit(a);
        r.source.deinit(a);
    }
};

test "runner: live content-terminal fences survive an unbudgeted think boundary" {
    const parsed = try std.json.parseFromSlice([]const Native, a, @embedFile("fixtures/runner/handoff-rg.json"), .{});
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 3), parsed.value.len);
    for (parsed.value, 0..) |row, index| {
        try t.expect(row.original_sha256_match);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var buffer: [16]u8 = undefined;
        for (row.chunks, 0..) |chunk, i| {
            if (i > 0) hash.update(",");
            hash.update(try std.fmt.bufPrint(&buffer, "{d}", .{chunk.token_id}));
        }
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        try t.expectEqualStrings(row.sha256, &std.fmt.bytesToHex(digest, .lower));
        for ([_]bool{ false, true }) |marker| {
            var replay: Replay = .{};
            defer replay.deinit();
            var stream = try lanes.Stream.init(a, .{
                .id = row.key,
                .prompt = &.{2},
                .max_new = 32768,
                .eos = &.{1},
                .think_end = if (marker) 128822 else -1,
                .runner = .{ .ptr = &replay, .token = Replay.token, .eos = Replay.eos },
            });
            defer stream.deinit(a);
            for (row.chunks, 0..) |chunk, i| {
                try t.expectEqual(i, chunk.completion_index);
                const bytes = try a.alloc(u8, chunk.bytes_hex.len / 2);
                defer a.free(bytes);
                _ = try std.fmt.hexToBytes(bytes, chunk.bytes_hex);
                replay.bytes = bytes;
                try t.expectEqual(@as(usize, 1), try stream.commit(a, &.{chunk.token_id}, &.{}));
            }
            try t.expectEqual(@as(usize, if (marker or index == 2) 1 else 0), replay.closed);
            try t.expectEqual(@as(usize, if (marker or index == 2) 1 else 0), replay.fired);
            try t.expectEqualStrings(if (marker) "python" else "", replay.language.items);
            try t.expectEqual(marker and index < 2, stream.runner_eos_pending);
            try t.expectEqual(@as(u32, 0), stream.think_budget);
            try t.expect(!stream.loop_guard);
            if (marker and index == 0) try t.expectEqualStrings(
                "a = 282668092539785370248997280888\nb = 816005819957362936377102597457\nprint(a * b)\n",
                replay.source.items,
            );
        }
    }
}
