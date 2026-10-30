//! The compute trampoline: a freshly spawned process drops privileges, limits and confines itself, then execs Python.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const landlock = @import("compute_landlock.zig");
const seccomp = @import("compute_seccomp.zig");
const runner = @import("compute_runner.zig").runner;

/// The argument format's version, the first argument after the launcher.
pub const version = "v1";
/// What the trampoline needs of one job.
pub const Job = struct { max_ms: u32, mem_mb: u32, cpus: []const u8, python: []const u8 };
/// The user nobody runs jobs when the trampoline starts as root.
pub const nobody: u32 = 65534;
const env = [_:null]?[*:0]const u8{
    "PYTHONHASHSEED=0", "PYTHONDONTWRITEBYTECODE=1", "PYTHONIOENCODING=utf-8", "LANG=C.UTF-8", "PATH=/usr/bin",
};

/// The spawn argv: `launcher` then version, max_ms, mem_mb, cpus, python.
pub fn encode(a: Allocator, launcher: []const []const u8, j: Job) Allocator.Error![:null]?[*:0]const u8 {
    const argv = try a.allocSentinel(?[*:0]const u8, launcher.len + 5, null);
    for (launcher, 0..) |s, i| argv[i] = try a.dupeSentinel(u8, s, 0);
    const tail = argv[launcher.len..];
    tail[0] = version;
    tail[1] = try std.fmt.allocPrintSentinel(a, "{d}", .{j.max_ms}, 0);
    tail[2] = try std.fmt.allocPrintSentinel(a, "{d}", .{j.mem_mb}, 0);
    tail[3] = try a.dupeSentinel(u8, j.cpus, 0);
    tail[4] = try a.dupeSentinel(u8, j.python, 0);
    return argv;
}

/// The job in the trampoline's arguments (after the launcher), or null when they are not this version's.
pub fn decode(argv: []const [:0]const u8) ?Job {
    if (argv.len != 5 or !std.mem.eql(u8, argv[0], version)) return null;
    const max_ms = std.fmt.parseInt(u32, argv[1], 10) catch return null;
    const mem_mb = std.fmt.parseInt(u32, argv[2], 10) catch return null;
    if (max_ms == 0 or mem_mb == 0 or argv[4].len == 0) return null;
    return .{ .max_ms = max_ms, .mem_mb = mem_mb, .cpus = argv[3], .python = argv[4] };
}

/// Parses an affinity list such as "16-19" or "1,3,5-7" into `set`; false if malformed or past CPU 1023.
pub fn cpuList(text: []const u8, set: *[16]u64) bool {
    set.* = @splat(0);
    if (text.len == 0) return false;
    var parts = std.mem.splitScalar(u8, text, ',');
    while (parts.next()) |part| {
        const dash = std.mem.findScalar(u8, part, '-');
        const lo = std.fmt.parseInt(u16, part[0 .. dash orelse part.len], 10) catch return false;
        const hi = if (dash) |d| std.fmt.parseInt(u16, part[d + 1 ..], 10) catch return false else lo;
        if (lo > hi or hi >= 1024) return false;
        for (lo..@as(usize, hi) + 1) |cpu| set[cpu / 64] |= @as(u64, 1) << @intCast(cpu % 64);
    }
    return true;
}

/// Runs in the spawned process: confines it and execs the interpreter, or exits 125 with one line on stderr.
pub fn main(argv: []const [:0]const u8) noreturn {
    if (builtin.os.tag != .linux) std.process.exit(125) else confine(argv);
}

fn confine(argv: []const [:0]const u8) noreturn {
    const job = decode(argv) orelse die("arguments", code(.INVAL));
    check("close_range", linux.close_range(3, -1, .{ .UNSHARE = false, .CLOEXEC = false }));
    check("setsid", linux.setsid());
    const root = linux.geteuid() == 0;
    if (root) {
        const none = [_]linux.gid_t{0};
        check("setgroups", linux.setgroups(0, &none));
        check("setresgid", linux.setresgid(nobody, nobody, nobody));
        check("setresuid", linux.setresuid(nobody, nobody, nobody));
        if (linux.getuid() == 0 or linux.geteuid() == 0) die("uid drop", code(.PERM));
    }
    check("pdeathsig", linux.prctl(@backingInt(linux.PR.SET_PDEATHSIG), @backingInt(linux.SIG.KILL), 0, 0, 0));
    limit("RLIMIT_AS", .AS, @as(u64, job.mem_mb) << 20);
    limit("RLIMIT_CPU", .CPU, (job.max_ms + 999) / 1000 + 1);
    limit("RLIMIT_FSIZE", .FSIZE, 0);
    limit("RLIMIT_CORE", .CORE, 0);
    limit("RLIMIT_NOFILE", .NOFILE, 64);
    if (root) limit("RLIMIT_NPROC", .NPROC, 512);
    // nice 10 at least, never less than the engine's own: lowering it needs a privilege the job no longer has
    const inherited = 20 - @as(isize, @bitCast(linux.syscall2(.getpriority, 0, 0)));
    if (inherited < 10) check("setpriority", linux.syscall3(.setpriority, 0, 0, 10));
    if (job.cpus.len > 0) {
        var set: [16]u64 = undefined;
        if (!cpuList(job.cpus, &set)) die("cpu list", code(.INVAL));
        check("sched_setaffinity", linux.syscall3(.sched_setaffinity, 0, @sizeOf(@TypeOf(set)), @intFromPtr(&set)));
    }
    check("no_new_privs", linux.prctl(@backingInt(linux.PR.SET_NO_NEW_PRIVS), 1, 0, 0, 0));
    const python = argv[4];
    if (landlock.apply(python)) |f| die(f.step, f.errno);
    const e = seccomp.install();
    if (e != .SUCCESS) die("seccomp", code(e));
    check("chdir", linux.chdir("/"));
    const args = [_:null]?[*:0]const u8{ python, "-P", "-s", "-B", "-u", "-X", "utf8", "-c", runner };
    die("exec", code(linux.errno(linux.execve(python, &args, &env))));
}

fn code(e: linux.E) u16 {
    return @intCast(@backingInt(e));
}

fn check(step: []const u8, rc: usize) void {
    const e = linux.errno(rc);
    if (e != .SUCCESS) die(step, code(e));
}

fn limit(step: []const u8, res: linux.rlimit_resource, value: u64) void {
    const l: linux.rlimit = .{ .cur = value, .max = value };
    check(step, linux.setrlimit(res, &l));
}

/// Fails closed: one line on stderr, then exit 125 before anything runs unconfined.
fn die(step: []const u8, errno: u16) noreturn {
    var buf: [160]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "compute sandbox: {s} failed (errno {d})\n", .{ step, errno }) catch "compute sandbox: failed\n";
    _ = linux.write(2, line.ptr, line.len);
    linux.exit_group(125);
}
