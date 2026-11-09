//! A runner's terminal hand-off keeps the model EOS internal until its owned result settles.
const std = @import("std");

pub fn hold(s: anytype, token: u32, stopped: anytype) bool {
    if (!s.isEos(token)) return false;
    const hook = s.runner orelse return false;
    const eos = hook.eos orelse return false;
    if (s.hold_eos or s.grammar != null or s.force.items.len > 0 or s.budgetLeft() <= 1) return false;
    if (stopped(s) or !eos(hook.ptr, s.think_open, true)) return false;
    s.runner_eos_pending = true;
    s.held = true;
    s.awaiting = true;
    s.paused = true;
    return true;
}

/// The caller flushes the visible prefix first and skips the internal EOS only after a successful insertion.
pub fn resumeResult(s: anytype, gpa: std.mem.Allocator, tokens: []const u32) std.mem.Allocator.Error!bool {
    const room: usize = @intCast(@max(0, s.budgetLeft()));
    if (tokens.len == 0 or room <= 1) {
        s.finished = true;
        s.reason = .stop;
    } else {
        std.debug.assert(s.force.items.len == 0);
        const n = @min(tokens.len, room);
        const eos = s.emitted()[s.emitted().len - 1];
        try s.force.ensureUnusedCapacity(gpa, n + 1);
        s.force.appendSliceAssumeCapacity(tokens[0..n]);
        s.force.appendAssumeCapacity(eos);
        // A clipped result fills the budget before the queued EOS can commit, reporting length.
    }
    s.runner_eos_pending = false;
    s.held = false;
    s.awaiting = false;
    s.paused = false;
    return true;
}

test {
    _ = @import("runner_eos_test.zig");
}
