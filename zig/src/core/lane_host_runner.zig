//! The in-model runner on the engine thread: each round advances the reply sessions and continues held streams.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Advances every started job's runner session; a settled result is forced after the trigger it waited at.
pub fn tick(h: *LaneHost) void {
    if (h.runner == null) return;
    var ended = false;
    for (h.admitted.items) |job| {
        if (!job.started) continue;
        const s = job.runner orelse continue;
        const tokens = s.tick() catch {
            fail(job, "out of memory for a runner result");
            ended = true;
            continue;
        } orelse continue;
        defer h.gpa.free(tokens);
        const ok = resumeResult(h, job, tokens) catch {
            fail(job, "out of memory for a runner result");
            ended = true;
            continue;
        };
        ended = ended or job.stream.finished;
        if (!ok) std.log.warn("lane host: a runner result for request {d}, which is not held, was dropped", .{job.id});
    }
    if (ended) h.core.sweep() catch {};
}

/// An idle loop's wait: 1 ms while a stream is held at a runner trigger (its run settles off-thread, waking nothing).
pub fn idleWait(h: *LaneHost, idle_ms: i64) i64 {
    if (h.runner == null) return idle_ms;
    for (h.admitted.items) |job| {
        if (!job.started) continue;
        const s = job.runner orelse continue;
        if (s.awaiting()) return @min(idle_ms, held_poll_ms);
    }
    return idle_ms;
}

const held_poll_ms = 1;

fn fail(job: anytype, words: []const u8) void {
    const st = &job.stream;
    st.held = false;
    st.paused = false;
    st.awaiting = false;
    st.finished = true;
    st.reason = .@"error";
    st.problem = words;
}

/// Flushes before releasing a terminal hold; a successful insertion skips its internal EOS exactly once.
pub fn resumeResult(h: *LaneHost, job: *LaneHost.Job, tokens: []const u32) std.mem.Allocator.Error!bool {
    const terminal = job.stream.runner_eos_pending;
    if (terminal) h.send(job);
    const ok = try lanes.hold.resumeAwaiting(&job.stream, h.gpa, tokens);
    if (ok and terminal and !job.stream.finished) job.delivered += 1;
    return ok;
}

test {
    _ = @import("lane_host_runner_eos_test.zig");
}
