//! A shared round's draft rows priced apart from its streams' rows (Model.draft_row_ms): the window allocation keeps the drafts
//! the table at total rows would cut, and prices as before when the setting is off.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const fake = @import("fake.zig");
const t = std.testing;
const gpa = t.allocator;

/// Three streams' windows of 6 held drafts each allocated under `draft_row_ms`; the drafts each keeps.
fn grants(draft_row_ms: f64) ![3]u32 {
    // a round table rising 10 ms a row from 30 ms at one: a complete linear fixture, not a measured device table
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = 20 + 10 * @as(f64, @floatFromInt(row)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 6, .hidden_rows = true, .batch_rows = 16, .max_streams = 8, .shared_costs = &costs, .draft_row_ms = draft_row_ms }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [3]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (&streams) |*s| {
        s.* = try Stream.init(gpa, .{ .id = "rows", .prompt = &.{1}, .max_new = 1000, .eos = &.{} });
        made += 1;
    }
    var plans: [3]win.Plan = undefined;
    for (&plans, &streams) |*p, *s| p.* = .{ .stream = s, .position = 0, .kind = .head, .held = 6 };
    try win.allocate(&e, &plans);
    var out: [3]u32 = undefined;
    for (&out, plans) |*o, p| o.* = @intCast(p.count());
    return out;
}

test "draft rows priced apart are kept where the table at total rows cuts them" {
    const by_rows = try grants(0.0);
    const apart = try grants(1.0);
    var rows_total: u32 = 0;
    var apart_total: u32 = 0;
    for (by_rows, apart) |a, b| {
        rows_total += a;
        apart_total += b;
        try t.expect(b >= a);
    }
    // 10 ms a row against drafts that land at 0.85, 0.64, 0.45 ...: a few rows pay; at 1 ms a row nearly all six do
    try t.expect(rows_total <= 9);
    try t.expect(apart_total > rows_total);
    try t.expect(apart_total >= 12);
    // dear draft rows cut at least as many as the table does
    const dear = try grants(30.0);
    var dear_total: u32 = 0;
    for (dear) |d| dear_total += d;
    try t.expect(dear_total <= rows_total);
}
