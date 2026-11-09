//! Drafts past a lone stream's forced tail (a runner's or tool's result): the last fed row's drafts fill its window.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const Plan = @import("windows.zig").Plan;
const accept = @import("accept.zig");
const depth = @import("depth.zig");

fn on(e: *const Engine, s: *const sm.Stream) bool {
    return e.cfg.forced_drafts and e.alone and s.drafts and s.grammar == null;
}

fn width(e: *const Engine) usize {
    return @min(e.cfg.base_width, e.cfg.batch_rows);
}

/// Forced tokens still queued that the next window takes whole, with room for drafts after them (0: none, or off).
pub fn tail(e: *const Engine, s: *const sm.Stream) u32 {
    const n = s.force.items.len;
    if (!on(e, s) or n == 0 or n + 1 >= width(e)) return 0;
    return @intCast(n);
}

/// The head's next drafts for `s`: past a forced tail the next window takes whole, the tail's own positions too.
pub fn headDepth(e: *Engine, s: *const sm.Stream, w: *depth.Who, budget: ?i64, probe: depth.Probe) !i64 {
    const n = tail(e, s);
    if (n == 0) return e.rule.headDepth(w, budget, probe);
    var free = w.*;
    free.forced = false; // the depth it would draft unforced; no copy is ahead past a forced tail
    const d = try e.rule.headDepth(&free, budget, .{ .ptr = e, .copy = noCopy });
    const most = @min(@as(i64, e.rule.most_drafts), @max(1, w.draft_room - 1));
    if (d == 0 or n >= most) return 0;
    return @min(d + n, most);
}

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

/// A window holding the whole forced tail: the held drafts past it as host tokens (drafted from the same pending row).
pub fn extend(e: *Engine, s: *sm.Stream, p: *Plan, queued: ?sm.Held) !void {
    const q = queued orelse return;
    if (!on(e, s) or s.force.items.len > 0) return;
    if (q.parents) |parents| if (!accept.isChain(parents)) return;
    const count: usize = if (q.tokens) |t| t.len else q.count;
    const n = @min(count, width(e) - 1);
    if (n <= p.forced.len) return;
    const a = e.arena.allocator();
    const drafts = if (q.tokens) |t| t else read: {
        const read = e.backend.vtable.held orelse return;
        const out = try a.alloc(u32, count);
        try read(e.backend.ptr, s, out);
        break :read out;
    };
    p.tokens = try std.mem.concat(a, u32, &.{ p.forced, drafts[p.forced.len..n] });
}

/// A forced window's path: forced rows land whole, drafts past them up to the first the target did not pick.
pub fn conclude(e: *Engine, s: *sm.Stream, p: Plan, tokens: []const u32, sampled: []const u32, committed: *std.ArrayList(u32)) ![]u32 {
    const a = e.arena.allocator();
    const fed = p.forced.len + 1;
    var n = fed;
    while (n < tokens.len and tokens[n] == sampled[n - 1]) n += 1;
    const path = try a.alloc(u32, n);
    for (path, 0..) |*r, i| r.* = @intCast(i);
    try committed.appendSlice(a, p.forced);
    try committed.appendSlice(a, tokens[fed..n]);
    try committed.append(a, s.popForce() orelse sampled[n - 1]);
    if (tokens.len > fed) {
        e.drafted += tokens.len - fed;
        e.accepted += n - fed;
        s.drafted += tokens.len - fed;
        s.accepted += n - fed;
    }
    return path;
}
