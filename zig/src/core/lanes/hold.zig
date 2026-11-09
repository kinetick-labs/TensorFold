//! A stream held at its EOS (Stream.hold_eos) for a server tool's result: its tokens forced after the EOS, or the end.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Stream = @import("stream.zig").Stream;

/// Streams the next step decodes: a paused (held) stream waits outside the rounds.
pub fn runnableCount(live: []const *Stream) usize {
    var n: usize = 0;
    for (live) |s| n += @intFromBool(!s.finished and !s.paused);
    return n;
}

/// A server's resume of a held stream: the tokens forced after its EOS (empty: it ends), and whether they open a think.
pub const Resume = struct { tokens: []const u32, think_open: bool = false };

pub const Outcome = enum {
    /// The tokens follow the EOS as forced rows; the model draws again after the last.
    resumed,
    /// No tokens: the reply ends as the EOS would have ended it.
    ended,
    /// The stream was not held (already resumed, never held or finished): nothing changed.
    not_held,
    /// The tokens and one draw do not fit the reply's budget: it ends at the EOS instead.
    no_room,
};

/// Applies one resume on the engine thread between rounds; a stream takes at most one per hold.
pub fn resumeHeld(s: *Stream, gpa: Allocator, tokens: []const u32, think_open: bool) Allocator.Error!Outcome {
    if (!s.held or s.finished or s.awaiting) return .not_held;
    // the hold's EOS was drawn, so no forced token was waiting before it
    std.debug.assert(s.force.items.len == 0);
    const room = tokens.len > 0 and @as(i64, @intCast(tokens.len)) < s.budgetLeft();
    if (!room) {
        s.held = false;
        s.paused = false;
        s.finished = true;
        s.reason = .stop;
        return if (tokens.len == 0) .ended else .no_room;
    }
    try s.force.appendSlice(gpa, tokens);
    s.resumed_rows = tokens.len;
    s.held = false;
    s.paused = false;
    s.think_open = think_open;
    return .resumed;
}

/// Whether the token landing now is a resumeHeld result's row: the template's, so its EOS neither holds nor ends.
pub fn resultRow(s: *Stream) bool {
    defer s.resumed_rows -|= 1;
    return s.resumed_rows > 0;
}

/// Continues a stream held at a runner trigger: `tokens` (cut to the room left) are forced next, then the model draws.
pub fn resumeAwaiting(s: *Stream, gpa: Allocator, tokens: []const u32) Allocator.Error!bool {
    if (!s.awaiting or !s.held or s.finished) return false;
    if (s.runner_eos_pending) return @import("runner_eos.zig").resumeResult(s, gpa, tokens);
    const room: usize = @intCast(@max(0, s.budgetLeft() - 1));
    try s.force.appendSlice(gpa, tokens[0..@min(tokens.len, room)]);
    s.held = false;
    s.paused = false;
    s.awaiting = false;
    return true;
}

/// At a runner trigger in `commit`: forces the hook's settled result (cut to the room left) instead of a hold.
pub fn settle(s: *Stream, gpa: Allocator, r: @import("stream.zig").RunnerHook) Allocator.Error!bool {
    const ids = (r.settled orelse return false)(r.ptr) orelse return false;
    const room: usize = @intCast(@max(0, s.budgetLeft() - 1));
    try s.force.appendSlice(gpa, ids[0..@min(ids.len, room)]);
    if (ids.len > 0 and room > 0) s.settle = .matching;
    return s.settle == .matching;
}

/// A later token of the settling commit: false cuts the commit there (not the result's next token), else it lands.
pub fn settledNext(s: *Stream, t: u32) bool {
    if (s.force.items.len == 0 or s.force.items[0] != t) {
        s.settle = .cut;
        return false;
    }
    _ = s.force.orderedRemove(0);
    if (s.force.items.len == 0) s.settle = .none;
    return true;
}
