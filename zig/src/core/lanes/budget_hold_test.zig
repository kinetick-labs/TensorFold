//! The thinking budget beside a held tool result: the result's forced rows land whole, the close comes after them.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");

const gpa = std.testing.allocator;
const close = [_]u32{ 90, 91, 92 };
const think_end: i64 = 91;
const base_result = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88, 12, 23, 34, 45, 56, 67, 78, 89, 13, 24 };
const eos_row = 2; // the result's own template EOS (ChatML's <|im_end|> closing the tool turn) sits on this row

/// The tool result a hold resumes with: its row `eos_row` is the reply's EOS, as a ChatML tool turn's rows hold one.
fn resultWith(eos: ?u32) [base_result.len]u32 {
    var r = base_result;
    if (eos) |e| r[eos_row] = e;
    return r;
}

/// A fresh non-close draw (index 2 on) before any think end: an EOS the stream reaches with its think block open.
fn eosInThink(prompt: []const u32) !struct { token: u32, at: usize } {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, prompt);
    var seen: [fake.vocab]bool = @splat(false);
    for (0..64) |i| {
        const t = fake.next(h.items, null, h.items.len);
        if (@as(i64, t) == think_end) return error.ThinkEndFirst;
        if (i >= 2 and !seen[t] and std.mem.indexOfScalar(u32, &close, t) == null) return .{ .token = t, .at = i };
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

const Spec = struct {
    prompt: []const u32 = &shared.p1,
    mtp: bool = true,
    drafts: bool = true,
    budget: u32 = 0,
    think_open: ?bool = null,
    eos: ?u32 = null,
    max_new: u32 = 70,
    /// held at its EOS: resumed with `result`, the think block open after it or not
    resume_think: bool = true,
};

const Out = struct { tokens: []u32, held_at: ?usize, reason: sm.Reason };

/// One stream on the fake target, stepped until it finishes; a hold at its EOS resumes once with `result`.
fn run(spec: Spec) !Out {
    var cfg = try shared.modelWith(spec.mtp);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const eos: []const u32 = if (spec.eos) |*e| e[0..1] else &.{};
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = spec.prompt, .max_new = spec.max_new, .eos = eos, .drafts = spec.drafts, .think_budget = spec.budget, .think_close = &close, .think_end = think_end, .think_open = spec.think_open, .hold_eos = spec.eos != null });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    var held_at: ?usize = null;
    var steps: usize = 0;
    while (!s.finished and steps < 600) : (steps += 1) {
        if (s.held and held_at == null) {
            held_at = s.emitted().len - 1;
            const result = resultWith(spec.eos);
            try std.testing.expectEqual(hold.Outcome.resumed, try hold.resumeHeld(&s, gpa, &result, spec.resume_think));
        }
        try engine.step();
    }
    try std.testing.expect(s.finished);
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .held_at = held_at, .reason = s.reason };
}

test "budget hold: a cut that would land inside a held result waits: every result row lands, then the close" {
    const eos = try eosInThink(&shared.p1);
    const result = resultWith(eos.token);
    for ([_]bool{ true, false }) |mtp| for ([_]bool{ true, false }) |drafts| for ([_]usize{ 1, eos_row + 1, 4, result.len - 1 }) |into| {
        const budget: u32 = @intCast(eos.at + 1 + into); // the budget's cut falls on the result's row `into`
        const r = try run(.{ .mtp = mtp, .drafts = drafts, .budget = budget, .eos = eos.token });
        defer gpa.free(r.tokens);
        try std.testing.expectEqual(@as(?usize, eos.at), r.held_at);
        const after = eos.at + 1;
        try std.testing.expectEqualSlices(u32, &result, r.tokens[after..][0..result.len]);
        // the close starts at the first position the model draws after the result
        try std.testing.expectEqualSlices(u32, &close, r.tokens[after + result.len ..][0..close.len]);
        try expectDecoded(&shared.p1, r.tokens, after + result.len + close.len);
    };
}

test "budget hold: with no lane call the close lands at the budget, drafted and plain alike" {
    for ([_]bool{ true, false }) |mtp| {
        const drafted = try run(.{ .prompt = &shared.p2, .mtp = mtp, .budget = 9, .max_new = 40 });
        defer gpa.free(drafted.tokens);
        const plain = try run(.{ .prompt = &shared.p2, .mtp = mtp, .drafts = false, .budget = 9, .max_new = 40 });
        defer gpa.free(plain.tokens);
        try std.testing.expectEqualSlices(u32, plain.tokens, drafted.tokens);
        try std.testing.expectEqualSlices(u32, &close, drafted.tokens[8..11]);
        try expectDecoded(&shared.p2, drafted.tokens[0..8], 0);
    }
}

test "budget hold: a lane call right after the forced close lands its result whole and the reply goes on exact" {
    const budget: u32 = 6;
    const ref = try run(.{ .budget = budget, .max_new = 60 });
    defer gpa.free(ref.tokens);
    try std.testing.expectEqualSlices(u32, &close, ref.tokens[budget - 1 ..][0..close.len]);
    // an EOS the reply first draws after its close
    var at: ?usize = null;
    for (ref.tokens, 0..) |t, i| if (i >= budget - 1 + close.len and std.mem.indexOfScalar(u32, ref.tokens[0..i], t) == null and t != 96) {
        at = i;
        break;
    };
    const j = at orelse return error.NoEosAfterClose;
    const r = try run(.{ .budget = budget, .eos = ref.tokens[j], .resume_think = false, .max_new = 60 });
    defer gpa.free(r.tokens);
    const result = resultWith(ref.tokens[j]);
    try std.testing.expectEqual(@as(?usize, j), r.held_at);
    try std.testing.expectEqualSlices(u32, ref.tokens[0 .. j + 1], r.tokens[0 .. j + 1]);
    try std.testing.expectEqualSlices(u32, &result, r.tokens[j + 1 ..][0..result.len]);
    try expectDecoded(&shared.p1, r.tokens, j + 1 + result.len);
}

test "budget hold: thinking off, a budget changes nothing: the held result lands and the reply equals the unbudgeted one" {
    const eos = try eosInThink(&shared.p1);
    for ([_]bool{ true, false }) |drafts| {
        const budgeted = try run(.{ .drafts = drafts, .budget = @intCast(eos.at + 1 + 4), .think_open = false, .eos = eos.token, .resume_think = false });
        defer gpa.free(budgeted.tokens);
        const unbudgeted = try run(.{ .drafts = drafts, .think_open = false, .eos = eos.token, .resume_think = false });
        defer gpa.free(unbudgeted.tokens);
        try std.testing.expectEqualSlices(u32, unbudgeted.tokens, budgeted.tokens);
        const result = resultWith(eos.token);
        try std.testing.expectEqualSlices(u32, &result, budgeted.tokens[eos.at + 1 ..][0..result.len]);
        try std.testing.expect(std.mem.indexOfScalar(u32, budgeted.tokens, close[0]) == null or
            std.mem.indexOfScalar(u32, unbudgeted.tokens, close[0]) != null);
    }
}
