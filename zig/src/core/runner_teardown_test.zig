//! Session.destroy joins every translation before it frees a cell: a later cell's prior borrows earlier code.
const std = @import("std");
const rapi = @import("runner_api.zig");
const runner = @import("runner.zig");

const gpa = std.testing.allocator;

fn bytes(_: ?*anyopaque, token: u32) []const u8 {
    const all = comptime blk: {
        var t: [128]u8 = undefined;
        for (&t, 0..) |*c, i| c.* = i;
        break :blk t;
    };
    return if (token < 128) all[token..][0..1] else "";
}

fn encode(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |c, *t| t.* = c;
    return out;
}

const first = "x = 12345678901234567890\nprint(x)\n";

/// The later cell's translation is held until destroy has begun, then watches its borrowed prior for a while.
const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    destroying: std.atomic.Value(bool) = .init(false),
    changed: std.atomic.Value(bool) = .init(false),

    fn translate(ctx: ?*anyopaque, _: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        if (cell.prior.len == 0) return .{ .abstain = .unsupported };
        g.entered.store(true, .release);
        while (!g.destroying.load(.acquire)) std.Thread.yield() catch {};
        // freed history reads back as the testing allocator's poison: the same bytes for 200 ms mean nothing freed them
        var i: u32 = 0;
        while (i < 200) : (i += 1) {
            if (!std.mem.eql(u8, cell.prior[0], first)) {
                g.changed.store(true, .release);
                break;
            }
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return .{ .abstain = .unsupported };
    }
};

const Back = struct {
    fn backend() rapi.Backend {
        return .{ .ctx = undefined, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release, .now = now } };
    }
    fn start(_: *anyopaque, _: u64, _: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        return error.Busy;
    }
    fn poll(_: *anyopaque, _: rapi.JobId) rapi.Poll {
        return .{ .done = .{ .status = .fault } };
    }
    fn now(_: *anyopaque, _: rapi.Program, _: u64, _: u64) ?rapi.Result {
        return null;
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
};

test "runner: destroy joins a later cell's translation before it frees the earlier cell that translation borrows" {
    var gate: Gate = .{};
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test" }, .frontend = .{ .ctx = &gate, .translate = Gate.translate }, .backend = Back.backend(), .text = .{ .bytes = bytes, .encode = encode } };
    const s = (try r.session(.{})).?;
    const hook = s.hook();
    // the second fence closes on the reply's last token: can_hold false (length cap), then the stream ends
    const reply = "```python\n" ++ first ++ "```\nThen:\n```python\nprint(x + 1)\n```\n";
    for (reply) |ch| _ = hook.token(hook.ptr, ch, false, false, false);
    try std.testing.expectEqual(2, s.cells.items.len);
    while (!gate.entered.load(.acquire)) std.Thread.yield() catch {};
    gate.destroying.store(true, .release);
    s.destroy();
    try std.testing.expect(!gate.changed.load(.acquire));
}
