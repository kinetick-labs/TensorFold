//! Clang kept started: a preloaded shim forks a child per request running clang's main; failures spawn as before.
const std = @import("std");
const builtin = @import("builtin");
const spawn = @import("runner_frontend_spawn.zig");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner_frontend);

const supported = builtin.link_libc and builtin.os.tag == .linux;

/// The shim, built at warm-up by the configured clang for the host: no headers, libc resolved in the tool.
pub const shim_source = @embedFile("runner_frontend_forkserver_shim.c");

/// Upper bound on any file a served child writes (outputs and captured streams).
const file_limit = 8 << 20;

const Server = struct {
    tool: [256]u8 = undefined,
    tool_len: usize = 0,
    pid: linux.pid_t = 0,
    fd: linux.fd_t = -1,
    busy: std.Io.Mutex = .init,
    alive: bool = false,
};

var table_lock: std.Io.Mutex = .init;
var servers: [2]Server = @splat(.{});

/// Starts a server for `tool` when none runs; true when one is ready. `dir` is an empty scratch directory.
pub fn start(a: Allocator, io: std.Io, tool: []const u8, linker: ?[]const u8, dir: []const u8) Allocator.Error!bool {
    if (comptime !supported) return false;
    if (tool.len == 0 or tool.len > 255) return false;
    table_lock.lockUncancelable(io);
    defer table_lock.unlock(io);
    for (&servers) |*s| if (s.alive and std.mem.eql(u8, s.tool[0..s.tool_len], tool)) return true;
    const slot = for (&servers) |*s| {
        if (!s.alive) break s;
    } else return false;
    const src = try std.fs.path.join(a, &.{ dir, "fs.c" });
    defer a.free(src);
    const so = try std.fs.path.join(a, &.{ dir, "fs.so" });
    defer a.free(so);
    const file = std.Io.Dir.cwd().createFile(io, src, .{ .permissions = .fromMode(0o600) }) catch return false;
    file.writeStreamingAll(io, shim_source) catch {
        file.close(io);
        return false;
    };
    file.close(io);
    const ld = if (linker) |l| try std.fmt.allocPrint(a, "--ld-path={s}", .{l}) else try a.dupe(u8, "-fuse-ld=lld");
    defer a.free(ld);
    // -nostdlib: an aarch64 compare-and-swap must be inline, not compiler-rt's outline-atomics call
    const atomics = if (builtin.cpu.arch == .aarch64) "-mno-outline-atomics" else "-O2";
    const built = try spawn.run(a, io, &.{ tool, "-shared", "-fPIC", "-nostdinc", "-nostdlib", "-O2", atomics, ld, "-o", so, src }, .{ .stdout = 4096, .stderr = 16384, .timeout = .fromMilliseconds(10_000) }) orelse return false;
    defer built.deinit(a);
    if (!built.ok) {
        log.warn("fork server shim did not build with {s}; clang starts per call", .{tool});
        return false;
    }
    var pair: [2]linux.fd_t = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0, &pair)) != .SUCCESS) return false;
    const pid = launch(a, tool, so, pair[1]) catch |e| {
        _ = linux.close(pair[0]);
        _ = linux.close(pair[1]);
        return e;
    };
    _ = linux.close(pair[1]);
    if (pid == 0) {
        _ = linux.close(pair[0]);
        return false;
    }
    var hello: [4]u8 = undefined;
    if (!exact(pair[0], &hello, 5000) or std.mem.readInt(u32, &hello, .little) != 0x52454459) {
        _ = linux.close(pair[0]);
        _ = linux.kill(pid, .KILL);
        reap(pid);
        return false;
    }
    slot.pid = pid;
    slot.fd = pair[0];
    slot.tool_len = tool.len;
    @memcpy(slot.tool[0..tool.len], tool);
    @atomicStore(bool, &slot.alive, true, .release);
    return true;
}

/// argv by argv[0]'s ready, free server (in absolute `dir`, streams in dir/.stdout, .stderr), else posix_spawn.
pub fn run(a: Allocator, io: std.Io, dir: []const u8, argv: []const []const u8, limits: spawn.Limits) Allocator.Error!?spawn.Ran {
    if (comptime supported) {
        if (find(io, argv[0])) |s| {
            defer s.busy.unlock(io);
            return served(a, io, s, dir, argv, limits);
        }
    }
    return spawn.run(a, io, argv, limits);
}

/// The ready server's process for `tool` (tests: what it holds open).
pub fn serverPid(tool: []const u8) ?linux.pid_t {
    for (&servers) |*s| if (@atomicLoad(bool, &s.alive, .acquire) and std.mem.eql(u8, s.tool[0..s.tool_len], tool)) return s.pid;
    return null;
}

/// Whether a server for `tool` is ready (tests and reports).
pub fn ready(tool: []const u8) bool {
    for (&servers) |*s| if (@atomicLoad(bool, &s.alive, .acquire) and std.mem.eql(u8, s.tool[0..s.tool_len], tool)) return true;
    return false;
}

fn find(io: std.Io, tool: []const u8) ?*Server {
    for (&servers) |*s| {
        if (!@atomicLoad(bool, &s.alive, .acquire) or !std.mem.eql(u8, s.tool[0..s.tool_len], tool)) continue;
        if (!s.busy.tryLock()) return null;
        if (s.alive) return s;
        s.busy.unlock(io);
        return null;
    }
    return null;
}

fn served(a: Allocator, io: std.Io, s: *Server, dir: []const u8, argv: []const []const u8, limits: spawn.Limits) Allocator.Error!?spawn.Ran {
    var request: std.ArrayList(u8) = .empty;
    defer request.deinit(a);
    try request.appendNTimes(a, 0, 8);
    std.mem.writeInt(u32, request.items[4..8], file_limit, .little);
    for ([_][]const []const u8{ &.{dir}, argv }) |list| for (list) |arg| {
        if (std.mem.indexOfScalar(u8, arg, 0) != null) return null;
        try request.appendSlice(a, arg);
        try request.append(a, 0);
    };
    if (request.items.len - 4 > 65535) return spawn.run(a, io, argv, limits);
    std.mem.writeInt(u32, request.items[0..4], @intCast(request.items.len - 4), .little);
    const end = now() + @as(u64, @intCast(@max(limits.timeout.toNanoseconds(), 0)));
    if (left(end) <= 0) return null;
    var reply: [4]u8 = undefined;
    // no server or a child gone before clang ran nothing: the argv spawns; 50 ms floor keeps healthy servers
    const took = send(s.fd, request.items) and exact(s.fd, &reply, std.math.clamp(left(end), 50, 2000));
    const child: linux.pid_t = if (took) @bitCast(std.mem.readInt(u32, &reply, .little)) else 0;
    if (child <= 0) {
        if (!took or child < 0) dead(s);
        if (left(end) <= 0) return null;
        var rest = limits;
        rest.timeout = .fromMilliseconds(left(end));
        return spawn.run(a, io, argv, rest);
    }
    // a status there at the deadline is the child's (it ran): no kill, as its pid may name another process now
    if (!exact(s.fd, &reply, left(end)) and !exact(s.fd, &reply, 0)) {
        _ = linux.kill(child, .KILL);
        if (!exact(s.fd, &reply, 2000)) dead(s);
        return null;
    }
    const status = std.mem.readInt(u32, &reply, .little);
    const stdout = try slurp(a, io, dir, ".stdout", limits.stdout) orelse return null;
    errdefer a.free(stdout);
    const stderr = try slurp(a, io, dir, ".stderr", limits.stderr) orelse {
        a.free(stdout);
        return null;
    };
    return .{ .stdout = stdout, .stderr = stderr, .ok = status == 0 };
}

/// A server that stopped answering is closed (it exits on end of stream); this call and later ones spawn.
fn dead(s: *Server) void {
    @atomicStore(bool, &s.alive, false, .release);
    _ = linux.close(s.fd);
    _ = linux.kill(s.pid, .KILL);
    reap(s.pid);
    log.warn("fork server for {s} stopped; clang starts per call", .{s.tool[0..s.tool_len]});
}

fn slurp(a: Allocator, io: std.Io, dir: []const u8, name: []const u8, limit: usize) Allocator.Error!?[]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit + 1)) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

/// Milliseconds before `end`, 0 once past.
fn left(end: u64) i64 {
    const t = now();
    return if (t >= end) 0 else @intCast((end - t) / std.time.ns_per_ms);
}

fn send(fd: linux.fd_t, bytes: []const u8) bool {
    var at: usize = 0;
    while (at < bytes.len) {
        // MSG_NOSIGNAL: a server gone fails the write instead of raising SIGPIPE in a host that keeps it
        const n = linux.sendto(fd, bytes[at..].ptr, bytes.len - at, linux.MSG.NOSIGNAL, null, 0);
        if (linux.errno(n) == .INTR) continue;
        if (linux.errno(n) != .SUCCESS or n == 0) return false;
        at += n;
    }
    return true;
}

/// Reads exactly buf.len bytes within `ms` milliseconds (0: only what is already there).
fn exact(fd: linux.fd_t, buf: []u8, ms: i64) bool {
    const end = now() + @as(u64, @intCast(ms)) * std.time.ns_per_ms;
    var at: usize = 0;
    while (at < buf.len) {
        const t = now();
        const wait: u64 = if (t >= end) 0 else (end - t) / std.time.ns_per_ms + 1;
        var pf = [_]linux.pollfd{.{ .fd = fd, .events = linux.POLL.IN, .revents = 0 }};
        const rc = linux.poll(&pf, 1, @intCast(@min(wait, std.math.maxInt(i32))));
        if (linux.errno(rc) == .INTR) continue;
        if (linux.errno(rc) != .SUCCESS) return false;
        if (rc == 0) {
            if (t >= end) return false;
            continue;
        }
        const n = linux.read(fd, buf[at..].ptr, buf.len - at);
        if (linux.errno(n) == .INTR) continue;
        if (linux.errno(n) != .SUCCESS or n == 0) return false;
        at += n;
    }
    return true;
}

const libc = struct {
    extern "c" fn posix_spawn(pid: *linux.pid_t, path: [*:0]const u8, actions: *const Opaque, attr: ?*const anyopaque, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int;
    extern "c" fn posix_spawn_file_actions_init(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_destroy(actions: *Opaque) c_int;
    extern "c" fn posix_spawn_file_actions_adddup2(actions: *Opaque, fd: c_int, to: c_int) c_int;
    extern "c" fn posix_spawn_file_actions_addopen(actions: *Opaque, fd: c_int, path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
    extern "c" var environ: [*:null]?[*:0]u8;
};

const Opaque = extern struct { bytes: [512]u8 align(16) = undefined };

/// The tool with the shim preloaded and the control socket on fd 3; 0 if it could not start.
fn launch(a: Allocator, tool: []const u8, so: []const u8, control: linux.fd_t) Allocator.Error!linux.pid_t {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var env: std.ArrayList(?[*:0]const u8) = .empty;
    var i: usize = 0;
    while (libc.environ[i]) |e| : (i += 1) {
        const text = std.mem.span(e);
        if (std.mem.startsWith(u8, text, "LD_PRELOAD=") or std.mem.startsWith(u8, text, "TF_FORKSERVER_FD=")) continue;
        try env.append(arena, e);
    }
    try env.append(arena, try std.fmt.allocPrintSentinel(arena, "LD_PRELOAD={s}", .{so}, 0));
    try env.append(arena, "TF_FORKSERVER_FD=3");
    try env.append(arena, null);
    const path = try arena.dupeSentinel(u8, tool, 0);
    const argv = [_:null]?[*:0]const u8{path};
    var actions: Opaque = .{};
    if (libc.posix_spawn_file_actions_init(&actions) != 0) return 0;
    defer _ = libc.posix_spawn_file_actions_destroy(&actions);
    var rc = libc.posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", 0, 0);
    if (rc == 0) rc = libc.posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", 1, 0);
    if (rc == 0) rc = libc.posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", 1, 0);
    // fds above 3 close in the shim's serve(): posix_spawn_file_actions_addclosefrom_np needs glibc 2.34
    if (rc == 0) rc = libc.posix_spawn_file_actions_adddup2(&actions, control, 3);
    var pid: linux.pid_t = 0;
    if (rc == 0) rc = libc.posix_spawn(&pid, path, &actions, null, &argv, @ptrCast(env.items.ptr));
    return if (rc == 0) pid else 0;
}

fn reap(pid: linux.pid_t) void {
    var st: i32 = 0;
    while (linux.errno(linux.wait4(pid, &st, 0, null)) == .INTR) {}
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
