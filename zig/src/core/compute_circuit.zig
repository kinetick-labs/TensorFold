//! S3: a compute lane call whose code the frontend accepts runs on the circuit (in-model runner), else in the sandbox.
const std = @import("std");
pub const api = @import("runner_api.zig"); // the frontend and backend S3 drives (a server test scripts them)
const runner = @import("runner.zig");
const lane = @import("compute_lane.zig");
const Allocator = std.mem.Allocator;

/// The executor names a report gives; a number never adds the two.
pub const in_model = "in-model runner";
pub const code_tool = "code tool";

/// S3's knobs beside the runner's under the same prefix (PREFIX_WALL_MS, PREFIX_BUSY_MS, PREFIX_VERIFY, PREFIX_ELF).
pub const Config = struct {
    wall_ms: u32 = 2000, // the safety net: a run still going is cancelled and the call goes to the sandbox
    busy_ms: u32 = 2000, // how long a call waits for a free backend slot before it goes to the sandbox
    verify: bool = false, // shadow: the sandbox runs too, its result is inserted and the bodies compared
    elf: bool = false, // compile ELF programs too: off, as clang alone (80-410 ms) loses to the sandbox (~15 ms)

    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!Config {
        var c: Config = .{};
        c.wall_ms = try knob(u32, get, prefix ++ "_WALL_MS", c.wall_ms, 1, 3_600_000);
        c.busy_ms = try knob(u32, get, prefix ++ "_BUSY_MS", c.busy_ms, 0, 3_600_000);
        c.verify = try knob(u1, get, prefix ++ "_VERIFY", 0, 0, 1) == 1;
        c.elf = try knob(u1, get, prefix ++ "_ELF", 0, 0, 1) == 1;
        return c;
    }

    fn knob(comptime T: type, get: *const fn ([]const u8) ?[]const u8, comptime name: []const u8, default: T, min: T, max: T) error{BadKnob}!T {
        const text = get(name) orelse return default;
        if (text.len == 0) return default;
        const v = std.fmt.parseInt(T, text, 10) catch return bad(name, text);
        return if (v < min or v > max) bad(name, text) else v;
    }

    fn bad(name: []const u8, value: []const u8) error{BadKnob} {
        std.log.warn("{s}={s} is not a valid value", .{ name, value });
        return error.BadKnob;
    }
};

/// What the engine hands the server: the runner S3 opened for itself (its own backend) and S3's knobs.
pub const Setup = struct { runner: *runner.Runner, cfg: Config };

/// S3's switch and knobs under a family's compute lane names (TF_COMPUTE_LANE_RUNNER*, or its alias's): off by default.
pub fn binding(comptime b: @import("compute_binding.zig").Binding) @import("runner_binding.zig").Binding {
    return .{ .prefix = b.prefix ++ "_RUNNER", .alias = if (b.alias) |a| a ++ "_RUNNER" else null };
}

/// S3 when switched on beside a lane: its own runner, sharing `with`'s circuit and bank when they name the same files.
pub fn open(comptime b: @import("compute_binding.zig").Binding, gpa: Allocator, get: *const fn ([]const u8) ?[]const u8, lane_on: bool, with: ?*runner.Runner, engine: @FieldType(runner.place.Config, "engine")) !?Setup {
    const rb = comptime binding(b);
    var rc = (try rb.parse(get)) orelse return null;
    if (!lane_on) {
        std.log.warn("{s}=1 needs the compute lane ({s}=1); S3 stays off", .{ rb.active(get), b.prefix });
        return null;
    }
    const cfg = if (rb.alias != null and std.mem.eql(u8, rb.active(get), rb.alias.?)) try Config.parse(get, rb.alias.?) else try Config.parse(get, rb.prefix);
    rc.place.engine = engine; // PREFIX_CPUS=auto keeps its threads off the engine's CPUs
    rc.compile = cfg.elf;
    return .{ .runner = try @import("runner_open.zig").openSharing(gpa, rc, with), .cfg = cfg };
}

/// Why a call went to the sandbox: an abstain, an ELF with _ELF off, a non-ok run, no slot, the wall net, or gone.
pub const Decline = enum { language, unsupported, syntax, name, too_long, too_big, context, markup, abstain, elf, budget, fault, busy, wall, cancelled };
pub const Kind = @FieldType(api.Program, "kind");
/// A call the circuit ran to its halt: the lane's result body made of its stdout, and what ran it.
pub const Owned = struct { body: []const u8, kind: Kind, steps: u64, exec_ns: u64 };
pub const Outcome = union(enum) { owned: Owned, declined: Decline };

/// One server's S3 executor; request threads share it, and every backend call happens under its lock.
pub const Exec = struct {
    frontend: api.Frontend,
    backend: api.Backend,
    max_steps: u64,
    cfg: Config,
    lock: std.atomic.Value(bool) = .init(false),
    next_stream: std.atomic.Value(u64) = .init(1),

    pub fn init(s: Setup) Exec {
        return .{ .frontend = s.runner.frontend, .backend = s.runner.backend, .max_steps = s.runner.cfg.max_steps, .cfg = s.cfg };
    }

    /// The call's code as a fresh fence cell (a lane call sees no earlier call's names), run to its halt or declined.
    pub fn run(e: *Exec, a: Allocator, code: []const u8, hook: ?lane.Cancel) Allocator.Error!Outcome {
        const t = try e.frontend.translate(e.frontend.ctx, a, .{ .form = .fence, .lang = "python", .code = code });
        const program = switch (t) {
            .program => |p| p,
            .abstain => |why| return .{ .declined = from(why) },
        };
        // the no-compile frontend builds none, yet a cached ELF may come back: S3 runs one only under _ELF=1
        if (program.kind == .elf and !e.cfg.elf) return .{ .declined = .elf };
        const stream = e.next_stream.fetchAdd(1, .monotonic);
        defer e.release(stream);
        const t0 = nowNs();
        const job = while (true) {
            if (e.start(stream, program)) |j| break j else |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Busy => if (nowNs() - t0 >= @as(u64, e.cfg.busy_ms) * std.time.ns_per_ms) return .{ .declined = .busy },
            }
            if (gone(hook)) return .{ .declined = .cancelled };
            sleepMs(1);
        };
        const t1 = nowNs();
        while (true) {
            switch (e.poll(job)) {
                .running => {},
                .done => |r| return switch (r.status) {
                    // `out` is the backend's until its next call for this stream: the body is copied first
                    .ok => .{ .owned = .{ .body = try body(a, r.out), .kind = program.kind, .steps = r.steps, .exec_ns = r.exec_ns } },
                    .abstain => .{ .declined = if (r.abstain) |why| from(why) else .abstain },
                    .budget => .{ .declined = .budget },
                    .fault => .{ .declined = .fault },
                },
            }
            const late = nowNs() - t1 >= @as(u64, e.cfg.wall_ms) * std.time.ns_per_ms;
            if (late or gone(hook)) {
                e.cancel(job);
                return .{ .declined = if (late) .wall else .cancelled };
            }
            sleepMs(1);
        }
    }

    fn start(e: *Exec, stream: u64, program: api.Program) error{ OutOfMemory, Busy }!api.JobId {
        e.acquire();
        defer e.lock.store(false, .release);
        return e.backend.start(stream, program, e.max_steps);
    }

    fn poll(e: *Exec, job: api.JobId) api.Poll {
        e.acquire();
        defer e.lock.store(false, .release);
        return e.backend.poll(job);
    }

    fn cancel(e: *Exec, job: api.JobId) void {
        e.acquire();
        defer e.lock.store(false, .release);
        e.backend.vtable.cancel(e.backend.ctx, job);
    }

    fn release(e: *Exec, stream: u64) void {
        e.acquire();
        defer e.lock.store(false, .release);
        e.backend.vtable.release(e.backend.ctx, stream);
    }

    fn acquire(e: *Exec) void {
        while (e.lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
};

/// The lane's result body for a run that printed `out` and ended ok: the sandbox's rule, so the bytes match.
pub fn body(a: Allocator, out: []const u8) Allocator.Error![]u8 {
    return lane.runner.body(a, .{ .status = .ok, .stdout = out }, 0, 0);
}

fn from(why: api.Abstain) Decline {
    return switch (why) {
        inline else => |w| @field(Decline, @tagName(w)),
    };
}

fn gone(hook: ?lane.Cancel) bool {
    const c = hook orelse return false;
    return c.gone(c.ctx);
}

fn sleepMs(ms: u64) void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(ms * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
