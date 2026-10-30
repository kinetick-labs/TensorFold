//! The compute sandbox's Landlock ruleset: read and run beneath the interpreter's roots; no network, no signals out.
const std = @import("std");
const linux = std.os.linux;

const RulesetAttr = extern struct { handled_access_fs: u64, handled_access_net: u64, scoped: u64 };
const PathBeneath = extern struct { allowed_access: u64 align(1), parent_fd: i32 align(1) };

pub const fs = struct {
    pub const execute: u64 = 1 << 0;
    pub const write_file: u64 = 1 << 1;
    pub const read_file: u64 = 1 << 2;
    pub const read_dir: u64 = 1 << 3;
    /// Every filesystem right of ABI 6 (EXECUTE through IOCTL_DEV).
    pub const all: u64 = (1 << 16) - 1;
};
const net_bind_tcp: u64 = 1 << 0;
const net_connect_tcp: u64 = 1 << 1;
const scope_abstract_unix: u64 = 1 << 0;
const scope_signal: u64 = 1 << 1;
const create_ruleset_version: u32 = 1 << 0;
const rule_path_beneath: u32 = 1;
pub const min_abi = 6;

/// A failed step and its errno.
pub const Fail = struct { step: []const u8, errno: u16 };

extern "c" fn realpath(noalias path: [*:0]const u8, noalias resolved: [*]u8) ?[*:0]u8;

fn fail(step: []const u8, rc: usize) ?Fail {
    const e = linux.errno(rc);
    return .{ .step = step, .errno = code(if (e == .SUCCESS) .INVAL else e) };
}

fn code(e: linux.E) u16 {
    return @intCast(@backingInt(e));
}

/// The interpreter's install root when it lives outside /usr: dirname(dirname(realpath)); null under /usr.
pub fn installRoot(real: []const u8) error{Root}!?[]const u8 {
    if (std.mem.startsWith(u8, real, "/usr/")) return null;
    const bin = std.fs.path.dirnamePosix(real) orelse return error.Root;
    const root = std.fs.path.dirnamePosix(bin) orelse return error.Root;
    if (std.mem.eql(u8, root, "/")) return error.Root; // a root of "/" would open everything
    return root;
}

/// Restricts the calling process to the compute ruleset for `python`; null on success, else the failed step.
pub fn apply(python: [*:0]const u8) ?Fail {
    const abi = linux.syscall3(.landlock_create_ruleset, 0, 0, create_ruleset_version);
    if (linux.errno(abi) != .SUCCESS) return fail("landlock abi", abi);
    if (abi < min_abi) return .{ .step = "landlock abi 6", .errno = code(.NOSYS) };
    const attr: RulesetAttr = .{
        .handled_access_fs = fs.all,
        .handled_access_net = net_bind_tcp | net_connect_tcp,
        .scoped = scope_abstract_unix | scope_signal,
    };
    const made = linux.syscall3(.landlock_create_ruleset, @intFromPtr(&attr), @sizeOf(RulesetAttr), 0);
    if (linux.errno(made) != .SUCCESS) return fail("landlock ruleset", made);
    const ruleset: i32 = @intCast(made);
    defer _ = linux.close(ruleset);
    const run = fs.read_file | fs.read_dir | fs.execute;
    if (rule(ruleset, "/usr", run, true)) |f| return f;
    var buf: [4096]u8 = undefined;
    var path: [4096]u8 = undefined;
    const real = std.mem.span(realpath(python, &buf) orelse return .{ .step = "interpreter", .errno = code(.NOENT) });
    const root = installRoot(real) catch return .{ .step = "interpreter root", .errno = code(.INVAL) };
    if (root) |r| {
        const z = std.mem.printSentinel(&path, "{s}", .{r}, 0) catch return .{ .step = "interpreter root", .errno = code(.NAMETOOLONG) };
        if (rule(ruleset, z, run, true)) |f| return f;
    }
    // distributions link the stdlib's sitecustomize.py into /etc: that one file, not its directory
    const site = std.mem.printSentinel(&path, "{s}/lib/{s}/sitecustomize.py", .{ root orelse "/usr", std.fs.path.basenamePosix(real) }, 0) catch "";
    if (site.len > 0) if (realpath(site, &buf)) |target| {
        if (rule(ruleset, std.mem.span(target), fs.read_file, false)) |f| return f;
    };
    if (rule(ruleset, "/etc/ld.so.cache", fs.read_file, false)) |f| return f;
    if (rule(ruleset, "/dev/urandom", fs.read_file, false)) |f| return f;
    if (rule(ruleset, "/dev/null", fs.read_file | fs.write_file, false)) |f| return f;
    const done = linux.syscall2(.landlock_restrict_self, @intCast(ruleset), 0);
    if (linux.errno(done) != .SUCCESS) return fail("landlock restrict", done);
    return null;
}

/// Grants `access` beneath `path`; a missing optional path is skipped.
fn rule(ruleset: i32, path: [:0]const u8, access: u64, required: bool) ?Fail {
    const opened = linux.open(path.ptr, .{ .PATH = true, .CLOEXEC = true }, 0);
    if (linux.errno(opened) == .NOENT and !required) return null;
    if (linux.errno(opened) != .SUCCESS) return fail(path, opened);
    const fd: i32 = @intCast(opened);
    defer _ = linux.close(fd);
    const beneath: PathBeneath = .{ .allowed_access = access, .parent_fd = fd };
    const added = linux.syscall4(.landlock_add_rule, @intCast(ruleset), rule_path_beneath, @intFromPtr(&beneath), 0);
    if (linux.errno(added) != .SUCCESS) return fail(path, added);
    return null;
}
