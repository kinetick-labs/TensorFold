//! The cross compiler's process by posix_spawn (Linux libc): unlike fork it never copies the runner's page tables.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;

/// A finished child's captured streams and whether it exited 0; the caller frees both streams.
pub const Ran = struct {
    stdout: []u8,
    stderr: []u8,
    ok: bool,

    pub fn deinit(r: Ran, a: Allocator) void {
        a.free(r.stdout);
        a.free(r.stderr);
    }
};

pub const Limits = struct { stdout: usize, stderr: usize, timeout: std.Io.Duration };

const spawned = builtin.link_libc and builtin.os.tag == .linux;
const spawn_setpgroup: c_short = 0x02;

/// argv run with stdin on /dev/null (PATH for a bare name); null if it cannot start, overflows a limit or times out.
pub fn run(a: Allocator, io: std.Io, argv: []const []const u8, limits: Limits) Allocator.Error!?Ran {
    if (comptime !spawned) return forked(a, io, argv, limits);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const args = try arena.allocSentinel(?[*:0]const u8, argv.len, null);
    for (argv, 0..) |arg, i| args[i] = try arena.dupeSentinel(u8, arg, 0);
    var fds: [4]linux.fd_t = @splat(-1);
    defer for (fds) |fd| if (fd >= 0) close(fd);
    for (0..2) |i| {
        var pair: [2]linux.fd_t = undefined;
        if (linux.errno(linux.pipe2(&pair, .{ .CLOEXEC = true })) != .SUCCESS) return null;
        fds[2 * i] = pair[0];
        fds[2 * i + 1] = pair[1];
    }
    const deadline = now() +| @as(u64, @intCast(@max(limits.timeout.toNanoseconds(), 0)));
    var pid: linux.pid_t = 0;
    if (!start(&pid, args, fds[1], fds[3])) return null;
    for ([_]usize{ 1, 3 }) |i| {
        close(fds[i]);
        fds[i] = -1;
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var err: std.ArrayList(u8) = .empty;
    defer err.deinit(a);
    const drained = drain(a, &fds, &out, &err, limits, deadline) catch |e| {
        kill(pid);
        return e;
    };
    if (!drained) {
        kill(pid);
        return null;
    }
    const status = switch (waitUntil(pid, deadline)) {
        .done => |status| status,
        .timed_out => {
            kill(pid);
            return null;
        },
        .lost => return null,
    };
    const stdout = try out.toOwnedSlice(a);
    errdefer a.free(stdout);
    return .{ .stdout = stdout, .stderr = try err.toOwnedSlice(a), .ok = status == 0 };
}

/// Reads both pipes to end of stream; false on a passed limit, the deadline or a failed poll.
fn drain(a: Allocator, fds: *[4]linux.fd_t, out: *std.ArrayList(u8), err: *std.ArrayList(u8), limits: Limits, deadline: u64) Allocator.Error!bool {
    var buf: [4096]u8 = undefined;
    while (fds[0] >= 0 or fds[2] >= 0) {
        const t = now();
        if (t >= deadline) return false;
        var pf = [_]linux.pollfd{
            .{ .fd = fds[0], .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = fds[2], .events = linux.POLL.IN, .revents = 0 },
        };
        const wait: i32 = @intCast(@min((deadline - t) / std.time.ns_per_ms + 1, std.math.maxInt(i32)));
        const rc = linux.poll(&pf, pf.len, wait);
        if (linux.errno(rc) == .INTR) continue;
        if (linux.errno(rc) != .SUCCESS) return false;
        for ([_]usize{ 0, 2 }, [_]*std.ArrayList(u8){ out, err }, [_]usize{ limits.stdout, limits.stderr }, 0..) |i, list, limit, p| {
            if (pf[p].revents == 0) continue;
            const n = linux.read(fds[i], &buf, buf.len);
            if (linux.errno(n) == .INTR or linux.errno(n) == .AGAIN) continue;
            if (linux.errno(n) != .SUCCESS or n == 0) {
                close(fds[i]);
                fds[i] = -1;
                continue;
            }
            if (list.items.len + n > limit) return false;
            try list.appendSlice(a, buf[0..n]);
        }
    }
    return true;
}

const libc = struct {
    extern "c" fn posix_spawnp(pid: *linux.pid_t, file: [*:0]const u8, actions: *const Opaque, attr: ?*const anyopaque, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int;
    extern "c" fn posix_spawn_file_actions_init(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_destroy(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_adddup2(actions: *Opaque, fd: c_int, to: c_int) c_int;
    extern "c" fn posix_spawn_file_actions_addopen(actions: *Opaque, fd: c_int, path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
    extern "c" fn posix_spawnattr_init(attr: *Opaque) c_int;
    extern "c" fn posix_spawnattr_destroy(attr: *Opaque) c_int;
    extern "c" fn posix_spawnattr_setpgroup(attr: *Opaque, group: linux.pid_t) c_int;
    extern "c" fn posix_spawnattr_setflags(attr: *Opaque, flags: c_short) c_int;
    extern "c" fn getsid(pid: linux.pid_t) linux.pid_t;
    extern "c" var environ: [*:null]?[*:0]u8;
};

const Opaque = extern struct { bytes: [512]u8 align(16) = undefined };

fn start(pid: *linux.pid_t, argv: [:null]?[*:0]const u8, stdout: linux.fd_t, stderr: linux.fd_t) bool {
    var attr: Opaque = .{};
    if (libc.posix_spawnattr_init(&attr) != 0) return false;
    defer _ = libc.posix_spawnattr_destroy(&attr);
    if (libc.posix_spawnattr_setpgroup(&attr, 0) != 0 or libc.posix_spawnattr_setflags(&attr, spawn_setpgroup) != 0) return false;
    var actions: Opaque = .{};
    if (libc.posix_spawn_file_actions_init(&actions) != 0) return false;
    defer _ = libc.posix_spawn_file_actions_destroy(&actions);
    var rc = libc.posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", 0, 0);
    if (rc == 0) rc = libc.posix_spawn_file_actions_adddup2(&actions, stdout, 1);
    if (rc == 0) rc = libc.posix_spawn_file_actions_adddup2(&actions, stderr, 2);
    if (rc == 0) rc = libc.posix_spawnp(pid, argv[0].?, &actions, &attr, argv.ptr, @ptrCast(libc.environ));
    return rc == 0;
}

fn kill(pid: linux.pid_t) void {
    _ = linux.kill(-pid, .KILL);
    _ = reap(pid);
}

/// Pipe EOF does not prove exit: a child that closes its streams still has the original deadline.
fn waitUntil(pid: linux.pid_t, deadline: u64) union(enum) { done: u32, timed_out, lost } {
    while (true) {
        var st: i32 = 0;
        const rc = linux.wait4(pid, &st, 1, null);
        switch (linux.errno(rc)) {
            .SUCCESS => if (rc != 0) return .{ .done = @bitCast(st) },
            .INTR => continue,
            else => return .lost,
        }
        if (now() >= deadline) return .timed_out;
        const pause: linux.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = linux.nanosleep(&pause, null);
    }
}

/// The child's wait status, or null when it cannot be reaped.
fn reap(pid: linux.pid_t) ?u32 {
    while (true) {
        var st: i32 = 0;
        switch (linux.errno(linux.wait4(pid, &st, 0, null))) {
            .SUCCESS => return @bitCast(st),
            .INTR => continue,
            else => return null,
        }
    }
}

fn close(fd: linux.fd_t) void {
    _ = linux.close(fd);
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Hosts without Linux libc (macOS builds) keep std.process.run.
fn forked(a: Allocator, io: std.Io, argv: []const []const u8, limits: Limits) Allocator.Error!?Ran {
    const r = std.process.run(a, io, .{
        .argv = argv,
        .stdout_limit = .limited(limits.stdout),
        .stderr_limit = .limited(limits.stderr),
        .timeout = .{ .duration = .{ .raw = limits.timeout, .clock = .awake } },
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    return .{ .stdout = r.stdout, .stderr = r.stderr, .ok = r.term.success() };
}

test "spawned compiler process: exit status, captured streams, limits, timeout and a missing program" {
    const t = std.testing;
    const ok = (try run(t.allocator, t.io, &.{ "/bin/sh", "-c", "printf out; printf err >&2" }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(5000) })).?;
    defer ok.deinit(t.allocator);
    try t.expect(ok.ok);
    try t.expectEqualStrings("out", ok.stdout);
    try t.expectEqualStrings("err", ok.stderr);
    const failed = (try run(t.allocator, t.io, &.{ "sh", "-c", "read x; exit 3" }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(5000) })).?;
    defer failed.deinit(t.allocator);
    try t.expect(!failed.ok);
    try t.expectEqual(@as(?Ran, null), try run(t.allocator, t.io, &.{ "/bin/sh", "-c", "head -c 200 /dev/zero" }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(5000) }));
    const t0 = now();
    try t.expectEqual(@as(?Ran, null), try run(t.allocator, t.io, &.{ "/bin/sh", "-c", "sleep 5" }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(100) }));
    try t.expect(now() - t0 < 3 * std.time.ns_per_s);
    if (comptime spawned) {
        const closed = now();
        try t.expectEqual(@as(?Ran, null), try run(t.allocator, t.io, &.{ "/bin/sh", "-c", "exec 1>/dev/null 2>&1; sleep 5" }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(100) }));
        try t.expect(now() - closed < 3 * std.time.ns_per_s);
        try descendantsStop();
    }
    try t.expectEqual(@as(?Ran, null), try run(t.allocator, t.io, &.{"/nonexistent/tf-clang"}, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(5000) }));
}

/// A timed-out compiler driver must leave no live worker in the scope, even when the worker outlives the driver.
fn descendantsStop() !void {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, "child" });
    defer t.allocator.free(path);
    try t.expectEqual(@as(?Ran, null), try run(t.allocator, t.io, &.{ "/bin/sh", "-c", "sleep 30 & echo $! > \"$1\"; wait", "compiler", path }, .{ .stdout = 64, .stderr = 64, .timeout = .fromMilliseconds(200) }));
    const bytes = try tmp.dir.readFileAlloc(t.io, "child", t.allocator, .limited(64));
    defer t.allocator.free(bytes);
    const pid = try std.fmt.parseInt(linux.pid_t, std.mem.trim(u8, bytes, " \r\n\t"), 10);
    const proc = try std.fmt.allocPrint(t.allocator, "/proc/{d}/stat", .{pid});
    defer t.allocator.free(proc);
    const settled_by = now() + std.time.ns_per_s;
    var alive = true;
    var owned = false;
    while (true) {
        const file = std.Io.Dir.cwd().openFile(t.io, proc, .{}) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer file.close(t.io);
        var buffer: [4096]u8 = undefined;
        var reader = file.readerStreaming(t.io, &buffer);
        // a stat that opens but fails to read (ESRCH) belongs to a task already gone; the open can keep succeeding
        const stat = reader.interface.allocRemaining(t.allocator, .limited(4096)) catch |err| switch (err) {
            error.ReadFailed => {
                alive = false;
                break;
            },
            else => return err,
        };
        defer t.allocator.free(stat);
        if (stat.len == 0) return;
        const end = std.mem.lastIndexOfScalar(u8, stat, ')') orelse return error.InvalidProcStat;
        var fields = std.mem.tokenizeScalar(u8, stat[end + 1 ..], ' ');
        const state = fields.next() orelse return error.InvalidProcStat;
        _ = fields.next() orelse return error.InvalidProcStat;
        _ = fields.next() orelse return error.InvalidProcStat;
        const sid = try std.fmt.parseInt(linux.pid_t, fields.next() orelse return error.InvalidProcStat, 10);
        alive = state[0] != 'Z' and state[0] != 'X';
        owned = sid == libc.getsid(0) and std.mem.indexOf(u8, stat, "(sleep)") != null;
        if (!alive or now() >= settled_by) break;
        const pause: linux.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = linux.nanosleep(&pause, null);
    }
    if (alive and owned) _ = linux.kill(pid, .KILL);
    try t.expect(!alive);
}
