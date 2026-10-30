//! Resumes the server queues for replies held at their EOS (Request.hold_eos), applied on the engine thread.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// A resume waiting for the engine thread, with its own copy of the tokens.
pub const Pending = struct { id: api.Id, tokens: []u32, think_open: bool };

/// Queues a resume from any thread; the engine thread applies it before its next round.
pub fn resumeFn(ctx: *anyopaque, id: api.Id, r: api.hold.Resume) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    const tokens = h.gpa.dupe(u32, r.tokens) catch null;
    h.lock();
    defer h.unlock();
    // without memory for the result the reply cannot go on: it is cancelled rather than left held
    const copy = tokens orelse return cancelLocked(h, id);
    h.resumes.append(h.gpa, .{ .id = id, .tokens = copy, .think_open = r.think_open }) catch {
        h.gpa.free(copy);
        return cancelLocked(h, id);
    };
    h.wake.signal(h.io);
}

fn cancelLocked(h: *LaneHost, id: api.Id) void {
    h.cancels.append(h.gpa, id) catch {};
    h.wake.signal(h.io);
}

/// Applies the queued resumes to their held streams; a job no longer admitted drops its resume.
pub fn take(h: *LaneHost) void {
    h.lock();
    var list = h.resumes;
    h.resumes = .empty;
    h.unlock();
    defer {
        for (list.items) |p| h.gpa.free(p.tokens);
        list.deinit(h.gpa);
    }
    var ended = false;
    for (list.items) |p| {
        const job = admitted(h, p.id) orelse continue;
        const s = &job.stream;
        const out = lanes.hold.resumeHeld(s, h.gpa, p.tokens, p.think_open) catch blk: {
            s.held = false;
            s.paused = false;
            s.finished = true;
            s.reason = .@"error";
            s.problem = "out of memory for a tool result";
            break :blk .no_room;
        };
        switch (out) {
            .resumed => {},
            .ended, .no_room => ended = true,
            .not_held => std.log.warn("lane host: a resume for request {d}, which is not held, was ignored", .{p.id}),
        }
    }
    // an ended stream leaves the live list now: the round loop may not step again before its job is delivered
    if (ended) h.core.sweep() catch {};
}

fn admitted(h: *LaneHost, id: api.Id) ?*Job {
    h.lock();
    defer h.unlock();
    for (h.admitted.items) |job| if (job.id == id and job.started) return job;
    return null;
}

/// Frees resumes no round took (the host is stopping).
pub fn close(h: *LaneHost) void {
    for (h.resumes.items) |p| h.gpa.free(p.tokens);
    h.resumes.deinit(h.gpa);
}
