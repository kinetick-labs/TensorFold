//! Inline spans inside an open quotation: the reply quotes text (its request), so the runner owns nothing there.
const std = @import("std");
const abstain = @import("runner_abstain.zig");
const runner = @import("runner.zig");
const scan = @import("runner_scan.zig");
const inline_scan = @import("runner_scan_inline.zig");
const h = @import("runner_digit_stop_test.zig");

const gpa = std.testing.allocator;

/// Whether the inline cell closing at the end of `text` (a byte a token) is quoted; null when none closes there.
fn at(text: []const u8, on: bool) !?bool {
    var s: scan.Scanner = .{ .cfg = .{ .inline_math = true, .inline_quoted = on } };
    defer s.deinit(gpa);
    var last: scan.Events = .{};
    for (0..text.len) |i| last = try s.feed(gpa, text[i .. i + 1], true);
    const c = last.closed orelse return null;
    if (c.form != .inline_expr or last.trigger == null) return null;
    return c.quoted;
}

test "runner inline quoted: a span inside an open straight or curly quotation is quoted; a closed one or none is not" {
    const yes = [_][]const u8{
        "We need answer user's request: \"Work out 7^222 exactly in your reasoning: write the line 7^222 =",
        "The user wants 13^150, written in the form \"13^150 =",
        "We need answer user: \"In your reasoning write the exact values as lines 3^300 =",
        "The user asked for \u{201C}the line 47*83 =",
        "They said \"a\" and then \"compute 7^222 mod 1000003 =",
    };
    for (yes) |line| try std.testing.expectEqual(@as(?bool, true), try at(line, true));
    const no = [_][]const u8{
        "13^150 =",
        "Let's present:\n7^222 =",
        "The \"exact\" value: 47*83 =",
        "He wrote \u{201C}exact\u{201D} so 47*83 =",
        "An earlier line \"left open\nSo 47*83 =",
        "The user's value: 47*83 =",
    };
    for (no) |line| try std.testing.expectEqual(@as(?bool, false), try at(line, true));
    // negative control: PREFIX_INLINE_QUOTED=0 is the base scanner, which called none of them quoted
    for (yes) |line| try std.testing.expectEqual(@as(?bool, false), try at(line, false));
}

const Fixture = struct { about: []const u8, owned: []const [2]usize, text: []const u8 };

/// For each span the base engine owned in a recorded reply: whether the scanner now calls its trigger quoted.
fn recorded(json: []const u8, want: []const bool) !void {
    const parsed = try std.json.parseFromSlice(Fixture, gpa, json, .{});
    defer parsed.deinit();
    const f = parsed.value;
    try std.testing.expectEqual(want.len, f.owned.len);
    for (f.owned, want) |span, quoted| {
        // the span is the value and its mark, after `= `: the text through the `=` ends at the trigger
        try std.testing.expectEqualStrings("= ", f.text[span[0] - 2 .. span[0]]);
        try std.testing.expectEqual(@as(?bool, quoted), try at(f.text[0 .. span[0] - 1], true));
        try std.testing.expectEqual(@as(?bool, false), try at(f.text[0 .. span[0] - 1], false));
    }
}

test "runner inline quoted on recorded TP1 replies: the request quoted back is not owned, the model's own line is" {
    // seed 12: `to write "13^150 =` twice, and `13^150 =` on its own line between them (the computation)
    try recorded(@embedFile("lanes/testdata/runner-stop-13p150-s12.json"), &.{ true, false, true });
    // seed 13: `written in the form "13^150 =`; owned, it led to `@engineer-mike/secret-repo!` and 1,024 digits
    const s13 = try std.json.parseFromSlice(Fixture, gpa, @embedFile("lanes/testdata/runner-stop-13p150-s13.json"), .{});
    defer s13.deinit();
    try std.testing.expectEqual(@as(?bool, true), try at(s13.value.text[0 .. s13.value.owned[0][0] - 1], true));
}

const Seen = struct { owned: []u8, reasons: []?abstain.Reason, armed: bool, cut: usize };

/// `text` through a session (a byte a token, the mark on), each hold settled; then 300 model digits, `cut` stopped.
fn reply(text: []const u8, on: bool) !Seen {
    var back: h.Back = .{};
    defer back.deinit();
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{}, .frontend = .{ .translate = h.translate }, .backend = back.backend(), .text = .{ .bytes = h.bytes, .encode = h.encode } };
    r.cfg.scan.inline_math = true;
    r.cfg.scan.inline_quoted = on;
    r.cfg.inline_mark = true;
    r.cfg.abstains = true; // the receipt's abstain records: the reason a reader of the receipt sees
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    var owned: std.ArrayList(u8) = .empty;
    errdefer owned.deinit(gpa);
    var reasons: std.ArrayList(?abstain.Reason) = .empty;
    errdefer reasons.deinit(gpa);
    for (text) |ch| {
        if (!hook.token(hook.ptr, ch, true, true, false)) continue;
        const cell = s.cells.items[s.waiting.?.index];
        while (!cell.translated.load(.acquire)) std.Thread.yield() catch {};
        const ids = for (0..64) |_| {
            if (try s.tick()) |x| break x;
        } else return error.NeverSettled;
        defer gpa.free(ids);
        const told = s.rec.abstains.items;
        try reasons.append(gpa, if (cell.abstained) told[told.len - 1].reason else null);
        for (ids) |t| {
            _ = hook.token(hook.ptr, t, true, false, false);
            try owned.appendSlice(gpa, h.bytes(null, t));
        }
    }
    const armed = s.digits.lead != null;
    var cut: usize = 0;
    for (0..300) |i| {
        const d: u32 = '0' + @as(u32, @intCast(i % 10));
        if (hook.stop.?(hook.ptr, &.{d}, 0) != null) cut += 1;
        _ = hook.token(hook.ptr, d, true, false, false);
    }
    return .{ .owned = try owned.toOwnedSlice(gpa), .reasons = try reasons.toOwnedSlice(gpa), .armed = armed, .cut = cut };
}

fn free(seen: Seen) void {
    gpa.free(seen.owned);
    gpa.free(seen.reasons);
}

test "runner inline quoted: a quoted request is not owned and the model's own long number after it lands whole" {
    const lead = "The user asks me to write \"47*83 =";
    const on = try reply(lead, true);
    defer free(on);
    try std.testing.expectEqualStrings("", on.owned);
    try std.testing.expectEqualSlices(?abstain.Reason, &.{.quoted}, on.reasons);
    // nothing forced, so no stop is armed: 300 digits of the model's own continuation land uncut
    try std.testing.expect(!on.armed);
    try std.testing.expectEqual(@as(usize, 0), on.cut);
    // negative control (the base engine, PREFIX_INLINE_QUOTED=0): the value and its mark go into the quotation
    const off = try reply(lead, false);
    defer free(off);
    try std.testing.expectEqualStrings(" 3901 @engine", off.owned);
    try std.testing.expectEqualSlices(?abstain.Reason, &.{null}, off.reasons);
    try std.testing.expect(off.armed and off.cut > 0);
}

test "runner inline quoted: after the quoted request the same expression on the model's own line is owned" {
    const seen = try reply("The user asks me to write \"47*83 = <value>\" first.\n47*83 =", true);
    defer free(seen);
    try std.testing.expectEqualStrings(" 3901 @engine", seen.owned);
    try std.testing.expectEqualSlices(?abstain.Reason, &.{ .quoted, null }, seen.reasons);
    // a quotation closed before the expression is the model's own statement
    const closed = try reply("The \"exact\" value: 47*83 =", true);
    defer free(closed);
    try std.testing.expectEqualStrings(" 3901 @engine", closed.owned);
    try std.testing.expectEqualSlices(?abstain.Reason, &.{null}, closed.reasons);
}

test "runner inline quoted: a quoted modular step abstains too, and the rule reads only the text before the span" {
    const mod = "x \"7^222 mod 1000003 =";
    try std.testing.expectEqual(@as(?bool, true), try at(mod, true));
    const closed = "say \"a\" 47*83 = \"b";
    try std.testing.expect(!inline_scan.quoted(closed, closed[8..13]));
    const open = "x \"y 47*83 = z\"";
    try std.testing.expect(inline_scan.quoted(open, open[5..10]));
}

test "runner inline quoted: TF_R_INLINE_QUOTED parses on by default, 0 off, anything else a bad knob" {
    const K = struct {
        var set: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (set) |k| if (std.mem.eql(u8, k[0], name)) return k[1];
            return null;
        }
    };
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" } };
    try std.testing.expect((try runner.Config.parse(&K.get, "TF_R")).?.scan.inline_quoted);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_INLINE_QUOTED", "0" } };
    try std.testing.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.scan.inline_quoted);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_INLINE_QUOTED", "2" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}
