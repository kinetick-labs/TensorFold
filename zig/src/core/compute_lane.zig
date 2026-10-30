//! The compute lane's executor: its knobs, a pool of sandbox slots and the supervisor of one sandboxed job.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const sandbox = @import("compute_sandbox.zig");
const log = std.log.scoped(.compute_lane);

pub const Config = struct {
    workers: u8 = 2, // concurrent sandboxes
    max_ms: u32 = 10_000, // wall limit per job
    max_output_bytes: u32 = 64 * 1024, // stdout+stderr cap; past it the job is killed
    max_result_tokens: u32 = 512, // result body cap, applied by the server
    max_calls: u32 = 8, // built-in calls per reply, applied by the server
    mem_mb: u32 = 1024, // RLIMIT_AS
    python: []const u8 = "/usr/bin/python3",
    cpus: []const u8 = "", // optional affinity list like "16-19" or "1,3,5-7"

    /// PREFIX "1" enables (unset, "" or "0": null); PREFIX_* knobs override defaults; a bad value is error.BadKnob.
    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!?Config {
        const on = get(prefix) orelse return null;
        if (on.len == 0 or std.mem.eql(u8, on, "0")) return null;
        if (!std.mem.eql(u8, on, "1")) return bad(prefix, on);
        var c: Config = .{};
        c.workers = try knob(u8, get, prefix ++ "_WORKERS", c.workers, 1, 64);
        c.max_ms = try knob(u32, get, prefix ++ "_MAX_MS", c.max_ms, 1, 3_600_000);
        c.max_result_tokens = try knob(u32, get, prefix ++ "_MAX_RESULT_TOKENS", c.max_result_tokens, 1, 1 << 20);
        c.max_calls = try knob(u32, get, prefix ++ "_MAX_CALLS", c.max_calls, 1, 1024);
        c.mem_mb = try knob(u32, get, prefix ++ "_MEM_MB", c.mem_mb, 64, 1 << 20);
        if (get(prefix ++ "_PYTHON")) |p| if (p.len > 0) {
            if (p[0] != '/') return bad(prefix ++ "_PYTHON", p);
            c.python = p;
        };
        if (get(prefix ++ "_CPUS")) |s| if (s.len > 0) {
            var set: [16]u64 = undefined;
            if (!sandbox.cpuList(s, &set)) return bad(prefix ++ "_CPUS", s);
            c.cpus = s;
        };
        return c;
    }

    fn knob(comptime T: type, get: *const fn ([]const u8) ?[]const u8, comptime name: []const u8, default: T, min: T, max: T) error{BadKnob}!T {
        const text = get(name) orelse return default;
        if (text.len == 0) return default;
        const v = std.fmt.parseInt(T, text, 10) catch return bad(name, text);
        if (v < min or v > max) return bad(name, text);
        return v;
    }

    fn bad(name: []const u8, value: []const u8) error{BadKnob} {
        log.warn("{s}={s} is not a valid value", .{ name, value });
        return error.BadKnob;
    }
};

pub const Status = enum { ok, exit_error, timeout, output_limit, cancelled, unavailable };
pub const Outcome = struct {
    status: Status,
    exit_code: ?u8 = null,
    signal: ?u8 = null,
    stdout: []const u8 = "",
    stderr: []const u8 = "",
    exec_ns: u64 = 0,
};
/// A caller's hook: `gone` turns true once the job's result is no longer wanted (client disconnected).
pub const Cancel = struct { ctx: *anyopaque, gone: *const fn (ctx: *anyopaque) bool };

pub const Pool = struct {
    cfg: Config,
    launcher: []const []const u8,
    busy: std.atomic.Value(u32) = .init(0),

    /// `launcher`: argv prefix that starts the trampoline, e.g. {"/proc/self/exe", "__compute_sandbox"}.
    pub fn init(cfg: Config, launcher: []const []const u8) Pool {
        return .{ .cfg = cfg, .launcher = launcher };
    }

    /// Waits for a free slot, runs `code` in the sandbox and frees the slot; failures to start are .unavailable.
    pub fn run(p: *Pool, a: Allocator, code: []const u8, cancel: ?Cancel) Allocator.Error!Outcome {
        if (builtin.os.tag != .linux) return .{ .status = .unavailable, .stderr = "compute lane: Linux only" } else {
            if (!p.acquire(cancel)) return .{ .status = .cancelled };
            defer _ = p.busy.fetchSub(1, .release);
            return supervise(p.cfg, a, p.launcher, code, cancel);
        }
    }

    /// Runs `print(1+1)`; anything but status ok with stdout "2\n" is error.SandboxUnavailable.
    pub fn selfTest(p: *Pool, a: Allocator) !void {
        const o = try p.run(a, "print(1+1)", null);
        if (o.status == .ok and std.mem.eql(u8, o.stdout, "2\n")) return;
        log.warn("compute lane self-test failed: {s}, stdout \"{s}\", stderr \"{s}\"", .{ @tagName(o.status), o.stdout, o.stderr });
        return error.SandboxUnavailable;
    }

    fn acquire(p: *Pool, cancel: ?Cancel) bool {
        while (true) {
            const n = p.busy.load(.acquire);
            if (n < p.cfg.workers) {
                if (p.busy.cmpxchgWeak(n, n + 1, .acquire, .monotonic) == null) return true;
                continue;
            }
            if (cancel) |c| if (c.gone(c.ctx)) return false;
            sleepMs(10);
        }
    }
};

const libc = struct {
    extern "c" fn posix_spawn(pid: *linux.pid_t, path: [*:0]const u8, actions: *const Opaque, attr: *const Opaque, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int;
    extern "c" fn posix_spawn_file_actions_init(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_destroy(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_adddup2(actions: *Opaque, fd: c_int, to: c_int) c_int;
    extern "c" fn posix_spawnattr_init(attr: *Opaque) c_int;
    extern "c" fn posix_spawnattr_destroy(attr: *Opaque) c_int;
    extern "c" fn posix_spawnattr_setflags(attr: *Opaque, flags: c_short) c_int;
    extern "c" fn posix_spawnattr_setsigmask(attr: *Opaque, set: *const Opaque) c_int;
    extern "c" fn posix_spawnattr_setsigdefault(attr: *Opaque, set: *const Opaque) c_int;
    extern "c" fn sigemptyset(set: *Opaque) c_int;
    extern "c" fn sigfillset(set: *Opaque) c_int;
};
/// Storage for libc's spawn attributes, file actions and sigset_t, larger than glibc's and musl's.
const Opaque = extern struct { bytes: [512]u8 align(16) = undefined };
const spawn_setsigdef: c_short = 0x04;
const spawn_setsigmask: c_short = 0x08;
const ns_per_ms = std.time.ns_per_ms;
// pipe ends: stdin read/write, stdout read/write, stderr read/write
const in_r = 0;
const in_w = 1;
const out_r = 2;
const out_w = 3;
const err_r = 4;
const err_w = 5;

fn supervise(cfg: Config, a: Allocator, launcher: []const []const u8, code: []const u8, cancel: ?Cancel) Allocator.Error!Outcome {
    if (launcher.len == 0) return unavailable(a, "launcher", @intCast(@backingInt(linux.E.INVAL)));
    const argv = try sandbox.encode(a, launcher, .{ .max_ms = cfg.max_ms, .mem_mb = cfg.mem_mb, .cpus = cfg.cpus, .python = cfg.python });
    var fds: [6]linux.fd_t = @splat(-1);
    defer for (fds) |fd| if (fd >= 0) close(fd);
    for (0..3) |i| {
        var pair: [2]linux.fd_t = undefined;
        const rc = linux.pipe2(&pair, .{ .CLOEXEC = true });
        if (linux.errno(rc) != .SUCCESS) return unavailable(a, "pipe", errno(rc));
        fds[2 * i] = pair[0];
        fds[2 * i + 1] = pair[1];
    }
    for ([_]usize{ in_w, out_r, err_r }) |i| {
        const flags = linux.fcntl(fds[i], linux.F.GETFL, 0);
        const nonblock: u32 = @bitCast(linux.O{ .NONBLOCK = true });
        const rc = linux.fcntl(fds[i], linux.F.SETFL, flags | nonblock);
        if (linux.errno(rc) != .SUCCESS) return unavailable(a, "fcntl", errno(rc));
    }
    var pid: linux.pid_t = 0;
    const failed = spawn(&pid, argv, fds[in_r], fds[out_w], fds[err_w]);
    if (failed != 0) return unavailable(a, "posix_spawn", failed);
    for ([_]usize{ in_r, out_w, err_w }) |i| {
        close(fds[i]);
        fds[i] = -1;
    }
    const start = now();
    const opened = linux.pidfd_open(pid, 0);
    if (linux.errno(opened) != .SUCCESS) {
        _ = linux.kill(pid, .KILL);
        _ = reap(pid);
        return unavailable(a, "pidfd_open", errno(opened));
    }
    const pidfd: linux.fd_t = @intCast(opened);
    defer close(pidfd);
    var io: Pipes = .{ .fds = &fds, .cap = cfg.max_output_bytes };
    if (code.len == 0) io.shut(in_w);
    const ours = io.loop(a, pidfd, code, start + @as(u64, cfg.max_ms) * ns_per_ms, cancel) catch |e| {
        _ = linux.kill(pid, .KILL);
        _ = reap(pid);
        return e;
    };
    if (ours != null) _ = linux.kill(pid, .KILL);
    const st = reap(pid);
    const o: Outcome = .{ .status = .ok, .stdout = try io.out.toOwnedSlice(a), .stderr = try io.err.toOwnedSlice(a), .exec_ns = now() - start };
    return classify(o, ours, st);
}

fn classify(o: Outcome, ours: ?Status, st: ?u32) Outcome {
    var r = o;
    if (ours) |s| {
        r.status = s;
        return r;
    }
    const w = st orelse {
        r.status = .unavailable;
        return r;
    };
    if (linux.W.IFEXITED(w)) {
        const code = linux.W.EXITSTATUS(w);
        r.status = if (code == 0) .ok else .exit_error;
        if (code == 125 and std.mem.startsWith(u8, r.stderr, "compute sandbox: ")) r.status = .unavailable;
        if (code != 0) r.exit_code = code;
    } else {
        r.status = .exit_error;
        r.signal = @intCast(@backingInt(linux.W.TERMSIG(w)));
    }
    return r;
}

/// The supervisor's side of one job's pipes: feeds stdin, collects capped stdout and stderr.
const Pipes = struct {
    fds: *[6]linux.fd_t,
    cap: u32,
    out: std.ArrayList(u8) = .empty,
    err: std.ArrayList(u8) = .empty,
    written: usize = 0,

    fn shut(io: *Pipes, i: usize) void {
        close(io.fds[i]);
        io.fds[i] = -1;
    }

    /// Polls until the child exits with both pipes drained (null), or the status the supervisor kills it for.
    fn loop(io: *Pipes, a: Allocator, pidfd: linux.fd_t, code: []const u8, deadline: u64, cancel: ?Cancel) Allocator.Error!?Status {
        var exited = false;
        while (io.fds[out_r] >= 0 or io.fds[err_r] >= 0 or !exited) {
            if (cancel) |cn| if (cn.gone(cn.ctx)) return .cancelled;
            const t = now();
            if (t >= deadline) return .timeout;
            const wait: i32 = @intCast(@min(50, (deadline - t + ns_per_ms - 1) / ns_per_ms));
            var pf = [_]linux.pollfd{
                .{ .fd = io.fds[out_r], .events = linux.POLL.IN },
                .{ .fd = io.fds[err_r], .events = linux.POLL.IN },
                .{ .fd = if (exited) -1 else pidfd, .events = linux.POLL.IN },
                .{ .fd = io.fds[in_w], .events = linux.POLL.OUT },
            };
            const rc = linux.poll(&pf, pf.len, wait);
            if (linux.errno(rc) == .INTR) continue;
            if (linux.errno(rc) != .SUCCESS) return .unavailable;
            if (pf[3].revents != 0) io.feed(code);
            if (pf[0].revents != 0 and try io.drain(a, out_r, &io.out)) return .output_limit;
            if (pf[1].revents != 0 and try io.drain(a, err_r, &io.err)) return .output_limit;
            if (pf[2].revents != 0) exited = true;
        }
        return null;
    }

    fn feed(io: *Pipes, code: []const u8) void {
        const rc = writeNoSigpipe(io.fds[in_w], code[io.written..]);
        switch (linux.errno(rc)) {
            .SUCCESS => io.written += rc,
            .AGAIN, .INTR => {},
            else => io.written = code.len, // the child closed stdin: nothing more to give it
        }
        if (io.written >= code.len) io.shut(in_w);
    }

    /// Reads what is ready from pipe `i`; true once stdout and stderr together pass the cap.
    fn drain(io: *Pipes, a: Allocator, i: usize, into: *std.ArrayList(u8)) Allocator.Error!bool {
        var buf: [16 * 1024]u8 = undefined;
        const rc = linux.read(io.fds[i], &buf, buf.len);
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .AGAIN, .INTR => return false,
            else => {
                io.shut(i);
                return false;
            },
        }
        if (rc == 0) {
            io.shut(i);
            return false;
        }
        const room = io.cap -| (io.out.items.len + io.err.items.len);
        try into.appendSlice(a, buf[0..@min(rc, room)]);
        return rc > room;
    }
};

/// Writes with SIGPIPE blocked on this thread, consuming the one an EPIPE raises.
fn writeNoSigpipe(fd: linux.fd_t, bytes: []const u8) usize {
    var set = linux.sigemptyset();
    linux.sigaddset(&set, .PIPE);
    var old: linux.sigset_t = undefined;
    _ = linux.sigprocmask(linux.SIG.BLOCK, &set, &old);
    defer _ = linux.sigprocmask(linux.SIG.SETMASK, &old, null);
    const rc = linux.write(fd, bytes.ptr, bytes.len);
    if (linux.errno(rc) == .PIPE) {
        const zero: linux.timespec = .{ .sec = 0, .nsec = 0 };
        _ = linux.syscall4(.rt_sigtimedwait, @intFromPtr(&set), 0, @intFromPtr(&zero), @sizeOf(linux.sigset_t));
    }
    return rc;
}

/// Starts the trampoline with the pipe ends as its stdio, an empty signal mask and default handlers; 0 or the errno.
fn spawn(pid: *linux.pid_t, argv: [:null]?[*:0]const u8, stdin: linux.fd_t, stdout: linux.fd_t, stderr: linux.fd_t) u16 {
    var actions: Opaque = .{};
    var attr: Opaque = .{};
    var empty: Opaque = .{};
    var all: Opaque = .{};
    var rc = libc.posix_spawn_file_actions_init(&actions);
    if (rc != 0) return spawnErrno(rc);
    defer _ = libc.posix_spawn_file_actions_destroy(&actions);
    rc = libc.posix_spawnattr_init(&attr);
    if (rc != 0) return spawnErrno(rc);
    defer _ = libc.posix_spawnattr_destroy(&attr);
    for ([_][2]linux.fd_t{ .{ stdin, 0 }, .{ stdout, 1 }, .{ stderr, 2 } }) |d| {
        if (rc == 0) rc = libc.posix_spawn_file_actions_adddup2(&actions, d[0], d[1]);
    }
    if (rc == 0) rc = libc.sigemptyset(&empty) | libc.sigfillset(&all);
    if (rc == 0) rc = libc.posix_spawnattr_setsigmask(&attr, &empty);
    if (rc == 0) rc = libc.posix_spawnattr_setsigdefault(&attr, &all);
    if (rc == 0) rc = libc.posix_spawnattr_setflags(&attr, spawn_setsigmask | spawn_setsigdef);
    if (rc == 0) rc = libc.posix_spawn(pid, argv[0].?, &actions, &attr, argv.ptr, &[_:null]?[*:0]const u8{});
    return spawnErrno(rc);
}

fn spawnErrno(rc: c_int) u16 {
    return if (rc >= 0 and rc < 4096) @intCast(rc) else @intCast(@backingInt(linux.E.INVAL));
}

/// The child's wait status, or null when it cannot be reaped.
fn reap(pid: linux.pid_t) ?u32 {
    while (true) {
        var st: i32 = 0;
        const rc = linux.wait4(pid, &st, 0, null);
        switch (linux.errno(rc)) {
            .SUCCESS => return @bitCast(st),
            .INTR => continue,
            else => return null,
        }
    }
}

fn unavailable(a: Allocator, step: []const u8, e: u16) Allocator.Error!Outcome {
    return .{ .status = .unavailable, .stderr = try std.fmt.allocPrint(a, "compute lane: {s} failed (errno {d})", .{ step, e }) };
}

fn errno(rc: usize) u16 {
    return @intCast(@backingInt(linux.errno(rc)));
}

fn close(fd: linux.fd_t) void {
    _ = linux.close(fd);
}

fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepMs(ms: u32) void {
    const ts: linux.timespec = .{ .sec = 0, .nsec = @intCast(@as(u64, ms) * ns_per_ms) };
    _ = linux.nanosleep(&ts, null);
}
