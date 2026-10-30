//! A stream held at its EOS: other streams decode meanwhile, a resume forces the result once, the reply goes on exact.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");
const p1 = shared.p1;
const p2 = shared.p2;

const gpa = std.testing.allocator;
const suffix = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88, 12, 23, 34, 45, 56, 67, 78, 89, 13, 24 };

/// The fake target's first draw after `prompt` (index 2 or later) that no earlier draw repeats: a reachable EOS.
fn eosAfter(prompt: []const u32) !struct { token: u32, at: usize } {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, prompt);
    var seen: [fake.vocab]bool = @splat(false);
    for (0..64) |i| {
        const t = fake.next(h.items, null, h.items.len);
        if (i >= 2 and !seen[t]) return .{ .token = t, .at = i };
        seen[t] = true;
        try h.append(gpa, t);
    }
    return error.NoEos;
}

/// Each token of `emitted` from `from` on is the fake target's draw after the prompt and everything before it.
fn expectDecoded(prompt: []const u32, emitted: []const u32, from: usize) !void {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, prompt);
    try h.appendSlice(gpa, emitted[0..from]);
    for (emitted[from..]) |t| {
        try std.testing.expectEqual(fake.next(h.items, null, h.items.len), t);
        try h.append(gpa, t);
    }
}

const Run = struct {
    engine: Engine,
    target: fake.Fake,
    clock: fake.FixedClock = .{},
    cfg: @import("config.zig").Config,

    fn init(r: *Run, mtp: bool) !void {
        r.cfg = try shared.modelWith(mtp);
        r.target = .{ .gpa = gpa };
        r.clock = .{};
        r.engine = Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
    }

    fn deinit(r: *Run) void {
        r.engine.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }

    fn until(r: *Run, s: *const sm.Stream, comptime field: []const u8) !void {
        var steps: usize = 0;
        while (!@field(s, field) and steps < 400) : (steps += 1) try r.engine.step();
        try std.testing.expect(@field(s, field));
    }
};

/// The peer's reply alone, for the shared-round comparison.
fn solo(mtp: bool, drafts: bool) ![]u32 {
    var r: Run = undefined;
    try r.init(mtp);
    defer r.deinit();
    var peer = try sm.Stream.init(gpa, .{ .id = "peer", .prompt = &p2, .max_new = 40, .drafts = drafts });
    defer peer.deinit(gpa);
    try r.engine.addStream(&peer);
    try r.until(&peer, "finished");
    return gpa.dupe(u32, peer.emitted());
}

test "compute hold: a held EOS keeps its stream while another decodes; one resume forces the suffix and the reply goes on exact" {
    const eos = try eosAfter(&p1);
    for ([_]bool{ true, false }) |mtp| for ([_]bool{ true, false }) |drafts| {
        var r: Run = undefined;
        try r.init(mtp);
        defer r.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "held", .prompt = &p1, .max_new = 70, .eos = &.{eos.token}, .drafts = drafts, .hold_eos = true });
        defer s.deinit(gpa);
        var peer = try sm.Stream.init(gpa, .{ .id = "peer", .prompt = &p2, .max_new = 40, .drafts = drafts });
        defer peer.deinit(gpa);
        try r.engine.addStream(&s);
        try r.engine.addStream(&peer);
        try r.until(&s, "held");
        try std.testing.expect(s.paused and !s.finished);
        try std.testing.expectEqual(eos.at + 1, s.emitted().len);
        try std.testing.expectEqual(eos.token, s.emitted()[eos.at]);
        // held: the stream draws nothing while the peer keeps its rounds
        const peer_before = peer.emitted().len;
        for (0..3) |_| try r.engine.step();
        try std.testing.expectEqual(eos.at + 1, s.emitted().len);
        try std.testing.expect(peer.finished or peer.emitted().len > peer_before);
        try std.testing.expectEqual(hold.Outcome.resumed, try hold.resumeHeld(&s, gpa, &suffix, false));
        try std.testing.expectEqual(hold.Outcome.not_held, try hold.resumeHeld(&s, gpa, &suffix, false));
        try r.until(&s, "finished");
        try std.testing.expect(s.reason == .length);
        const out = s.emitted();
        try std.testing.expectEqualSlices(u32, &suffix, out[eos.at + 1 .. eos.at + 1 + suffix.len]);
        // the model's draws after the suffix are the fake target's over the whole history, EOS and suffix included
        try expectDecoded(&p1, out, eos.at + 1 + suffix.len);
        try expectDecoded(&p1, out[0 .. eos.at + 1], 0);
        try r.until(&peer, "finished");
        const alone = try solo(mtp, drafts);
        defer gpa.free(alone);
        try std.testing.expectEqualSlices(u32, alone, peer.emitted());
    };
}

test "compute hold: an empty resume ends the held reply at its EOS, as a stream without a hold ends" {
    const eos = try eosAfter(&p1);
    var r: Run = undefined;
    try r.init(true);
    defer r.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "held", .prompt = &p1, .max_new = 70, .eos = &.{eos.token}, .hold_eos = true });
    defer s.deinit(gpa);
    var plain = try sm.Stream.init(gpa, .{ .id = "plain", .prompt = &p1, .max_new = 70, .eos = &.{eos.token} });
    defer plain.deinit(gpa);
    try r.engine.addStream(&s);
    try r.until(&s, "held");
    try std.testing.expectEqual(hold.Outcome.ended, try hold.resumeHeld(&s, gpa, &.{}, false));
    try std.testing.expect(s.finished and s.reason == .stop and !s.held);
    try r.engine.step();
    try std.testing.expectEqual(@as(usize, 0), r.engine.live.items.len);
    try r.engine.addStream(&plain);
    try r.until(&plain, "finished");
    try std.testing.expectEqualSlices(u32, plain.emitted(), s.emitted());
}

test "compute hold: a resume past the reply's budget ends it at the EOS" {
    const eos = try eosAfter(&p1);
    var r: Run = undefined;
    try r.init(true);
    defer r.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "held", .prompt = &p1, .max_new = @intCast(eos.at + 1 + 4), .eos = &.{eos.token}, .hold_eos = true });
    defer s.deinit(gpa);
    try r.engine.addStream(&s);
    try r.until(&s, "held");
    try std.testing.expectEqual(hold.Outcome.no_room, try hold.resumeHeld(&s, gpa, suffix[0..4], false));
    try std.testing.expect(s.finished and s.reason == .stop);
    try std.testing.expectEqual(eos.at + 1, s.emitted().len);
}

test "compute hold: a held stream is cancelled like any other: discarded, released, out of the live list" {
    const eos = try eosAfter(&p1);
    var r: Run = undefined;
    try r.init(false);
    defer r.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "held", .prompt = &p1, .max_new = 70, .eos = &.{eos.token}, .hold_eos = true });
    defer s.deinit(gpa);
    try r.engine.addStream(&s);
    try r.until(&s, "held");
    try std.testing.expectEqual(@as(usize, 0), hold.runnableCount(r.engine.live.items));
    try std.testing.expectEqual(@as(usize, 1), r.engine.activeCount());
    r.engine.discard(&s);
    try std.testing.expect(s.finished and s.reason == .cancelled);
    try std.testing.expectEqual(@as(usize, 0), r.engine.live.items.len);
    try std.testing.expectEqual(hold.Outcome.not_held, try hold.resumeHeld(&s, gpa, &suffix, false));
}
