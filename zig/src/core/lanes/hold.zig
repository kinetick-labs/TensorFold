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
    if (!s.held or s.finished) return .not_held;
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
    s.held = false;
    s.paused = false;
    s.think_open = think_open;
    return .resumed;
}
