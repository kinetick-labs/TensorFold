//! Bounded graph workers over a boot-loaded bank; the engine thread never waits for a running job.
const std = @import("std");
const api = @import("runner_api.zig");
const place = @import("runner_place.zig");
pub const Bank = @import("runner_graph_bank.zig").Bank;
pub const Machine = @import("runner_graph_machine.zig").Machine;
pub const Packet = @import("runner_graph_packet.zig").Packet;
pub const Op = @import("runner_graph_packet.zig").Op;
pub const Plan = @import("runner_graph_packet.zig").Plan;
pub const Print = @import("runner_graph_packet.zig").Print;
pub const PrintStyle = @import("runner_graph_packet.zig").Style;
pub const PrintEnding = @import("runner_graph_packet.zig").Ending;
pub const Builder = @import("runner_graph_packet.zig").Builder;
pub const id = "graph";
pub const bindCell = @import("runner_graph_bind.zig").cell;
pub const bindExpression = @import("runner_graph_bind.zig").expression;
pub const bindScript = @import("runner_graph_bind.zig").script;
pub const directFrontend = @import("runner_graph_frontend.zig").frontend;

const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.JobId,
    stream: u64,
    program: []u8,
    graph: bool,
    budget: u64,
    thread: ?std.Thread = null,
    done: std.atomic.Value(bool) = .init(false),
    cancel: std.atomic.Value(bool) = .init(false),
    result: api.Result = .{ .status = .abstain, .abstain = .unsupported },
    out: std.ArrayList(u8) = .empty,
};

const Cpu = struct {
    gpa: Allocator,
    bank: *const Bank,
    workers: u8,
    placer: ?*place.Placer = null,
    next_id: api.JobId = 1,
    jobs: std.ArrayList(*Job) = .empty,
    dropped: std.ArrayList(*Job) = .empty,
    delivered: std.ArrayList(*Job) = .empty,
    now_out: std.ArrayList(u8) = .empty, // the last run on the engine thread's output

    fn live(c: *Cpu) usize {
        return c.jobs.items.len + c.dropped.items.len;
    }

    fn free(c: *Cpu, j: *Job) void {
        if (j.thread) |t| t.join();
        j.out.deinit(c.gpa);
        c.gpa.free(j.program);
        c.gpa.destroy(j);
    }

    /// Joins and frees cancelled jobs whose worker has stopped.
    fn reap(c: *Cpu) void {
        var i: usize = 0;
        while (i < c.dropped.items.len) {
            const j = c.dropped.items[i];
            if (!j.done.load(.acquire)) {
                i += 1;
                continue;
            }
            _ = c.dropped.swapRemove(i);
            c.free(j);
        }
    }

    /// `out` of a stream's previous result stays valid only until the next call for that stream.
    fn forget(c: *Cpu, stream: u64) void {
        var i: usize = 0;
        while (i < c.delivered.items.len) {
            if (c.delivered.items[i].stream != stream) {
                i += 1;
                continue;
            }
            c.free(c.delivered.swapRemove(i));
        }
    }

    fn find(c: *Cpu, job_id: api.JobId) ?usize {
        for (c.jobs.items, 0..) |j, i| if (j.id == job_id) return i;
        return null;
    }

    fn drop(c: *Cpu, i: usize) void {
        const j = c.jobs.swapRemove(i);
        j.cancel.store(true, .release);
        c.dropped.appendAssumeCapacity(j);
    }
};

fn work(c: *Cpu, j: *Job) void {
    if (c.placer) |p| p.enter();
    var began: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &began);
    defer {
        var ended: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ended);
        const ns = (ended.sec - began.sec) * std.time.ns_per_s + ended.nsec - began.nsec;
        j.result.exec_ns = @intCast(@max(0, ns));
        j.done.store(true, .release);
    }
    if (!j.graph) return;
    var m = Machine.boot(c.gpa, c.bank, j.program) catch return;
    defer m.deinit();
    const end = m.run(j.budget, &j.out, &j.cancel) catch {
        j.out.clearRetainingCapacity();
        j.result = .{ .status = .abstain, .abstain = .unsupported, .steps = m.steps };
        return;
    };
    j.result = resultOf(end);
}

fn resultOf(end: api.RunEnd) api.Result {
    return .{
        .status = switch (end.end) {
            .halted => .ok,
            .budget => .budget,
            .fault => .fault,
            .cancelled => .abstain,
        },
        .steps = end.steps,
        .abstain = if (end.end == .cancelled) .unsupported else null,
    };
}

/// A job's run here, capped at `quick` steps: one that ends sooner is the job's own result; else (or an error) null.
fn now(ctx: *anyopaque, program: api.Program, budget: u64, quick: u64) ?api.Result {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    if (!std.mem.eql(u8, @tagName(program.kind), id) or program.bytes.len > @import("runner_graph_packet.zig").max_packet) return null;
    c.now_out.clearRetainingCapacity();
    var t0: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t0);
    var m = Machine.boot(c.gpa, c.bank, program.bytes) catch return null;
    defer m.deinit();
    m.decode = false; // the engine thread: a template's first decode (0.1-0.4 s wide) is the job's, in a hold
    const cap = @min(budget, quick);
    const end = m.run(cap, &c.now_out, null) catch return null;
    if (end.end == .budget and cap < budget) return null;
    var t1: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t1);
    var r = resultOf(end);
    r.out = c.now_out.items;
    r.exec_ns = @intCast(@max(0, (t1.sec - t0.sec) * std.time.ns_per_s + t1.nsec - t0.nsec));
    return r;
}

fn start(ctx: *anyopaque, stream: u64, program: api.Program, budget: u64) error{ OutOfMemory, Busy }!api.JobId {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    c.reap();
    c.forget(stream);
    if (c.live() >= c.workers) return error.Busy;
    const j = try c.gpa.create(Job);
    errdefer c.gpa.destroy(j);
    const valid = std.mem.eql(u8, @tagName(program.kind), id) and
        program.bytes.len <= @import("runner_graph_packet.zig").max_packet;
    const bytes = try c.gpa.dupe(u8, if (valid) program.bytes else "");
    errdefer c.gpa.free(bytes);
    j.* = .{ .id = c.next_id, .stream = stream, .program = bytes, .graph = valid, .budget = budget };
    try c.jobs.ensureUnusedCapacity(c.gpa, 1);
    j.thread = std.Thread.spawn(.{}, work, .{ c, j }) catch return error.Busy;
    c.jobs.appendAssumeCapacity(j);
    c.next_id += 1;
    return j.id;
}

fn poll(ctx: *anyopaque, job_id: api.JobId) api.Poll {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    c.reap();
    const i = c.find(job_id) orelse return .{ .done = .{ .status = .abstain, .abstain = .unsupported } };
    const j = c.jobs.items[i];
    if (!j.done.load(.acquire)) return .running;
    _ = c.jobs.swapRemove(i);
    if (j.thread) |t| t.join();
    j.thread = null;
    c.forget(j.stream);
    c.delivered.append(c.gpa, j) catch {
        c.free(j);
        return .{ .done = .{ .status = .abstain, .abstain = .unsupported } };
    };
    var r = j.result;
    r.out = j.out.items;
    return .{ .done = r };
}

fn cancel(ctx: *anyopaque, job_id: api.JobId) void {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    if (c.find(job_id)) |i| c.drop(i);
    c.reap();
}

fn release(ctx: *anyopaque, stream: u64) void {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    var i: usize = 0;
    while (i < c.jobs.items.len) {
        if (c.jobs.items[i].stream == stream) c.drop(i) else i += 1;
    }
    c.forget(stream);
    c.reap();
}

const vtable: api.Backend.VTable = .{ .start = start, .poll = poll, .cancel = cancel, .release = release, .now = now };

/// Runs at most workers jobs over a bank the caller keeps alive; gpa must support worker-thread allocations.
pub fn backend(gpa: Allocator, bank: *const Bank, workers: u8) !api.Backend {
    return placed(gpa, bank, workers, null);
}

/// As `backend`, each job's thread placed by `placer` (which the caller keeps alive) before it boots a machine.
pub fn placed(gpa: Allocator, bank: *const Bank, workers: u8, placer: ?*place.Placer) !api.Backend {
    if (workers == 0) return error.NoWorkers;
    const c = try gpa.create(Cpu);
    errdefer gpa.destroy(c);
    c.* = .{ .gpa = gpa, .bank = bank, .workers = workers, .placer = placer };
    try c.dropped.ensureTotalCapacity(gpa, workers);
    errdefer c.dropped.deinit(gpa);
    try c.jobs.ensureTotalCapacity(gpa, workers);
    return .{ .ctx = c, .vtable = &vtable };
}

/// Cancels every job, waits for the workers and frees the backend.
pub fn destroy(b: api.Backend) void {
    const c: *Cpu = @ptrCast(@alignCast(b.ctx));
    for (c.jobs.items) |j| j.cancel.store(true, .release);
    for ([_]*std.ArrayList(*Job){ &c.jobs, &c.dropped, &c.delivered }) |list| {
        for (list.items) |j| c.free(j);
        list.deinit(c.gpa);
    }
    c.now_out.deinit(c.gpa);
    c.gpa.destroy(c);
}
