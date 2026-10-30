//! The compute sandbox's seccomp filter: native arch only; no sockets, tracing, namespaces or child processes.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const BPF = linux.BPF;
const RET = linux.SECCOMP.RET;

/// One classic-BPF instruction (struct sock_filter).
pub const Insn = extern struct { code: u16, jt: u8, jf: u8, k: u32 };
const Fprog = extern struct { len: u16, filter: [*]const Insn };

/// Architectures the filter is built for; anywhere else `install` refuses.
pub const supported = builtin.os.tag == .linux and switch (builtin.cpu.arch) {
    .aarch64 => true,
    .x86_64 => builtin.abi != .gnux32 and builtin.abi != .muslx32,
    else => false,
};
pub const audit_arch: u32 = switch (builtin.cpu.arch) {
    .aarch64 => 0xC00000B7,
    .x86_64 => 0xC000003E,
    else => 0,
};
/// seccomp_data offsets: syscall number, audit arch and the low 32 bits of the first argument (little-endian).
pub const nr_at = 0;
pub const arch_at = 4;
pub const arg0_at = 16;
pub const x32_bit: u32 = 0x40000000;
pub const clone_thread: u32 = 0x00010000;
pub const eperm: u32 = 1;
pub const enosys: u32 = 38;

/// Calls that fail with EPERM; names missing on the native arch (fork, vfork off x86_64) are skipped.
pub const denied = [_][]const u8{
    "socket",            "socketpair",        "ptrace",          "process_vm_readv", "process_vm_writev",
    "mount",             "umount2",           "pivot_root",      "chroot",           "unshare",
    "setns",             "bpf",               "perf_event_open", "userfaultfd",      "io_uring_setup",
    "io_uring_enter",    "io_uring_register", "keyctl",          "add_key",          "request_key",
    "kexec_load",        "kexec_file_load",   "init_module",     "finit_module",     "delete_module",
    "reboot",            "swapon",            "swapoff",         "acct",             "name_to_handle_at",
    "open_by_handle_at", "fork",              "vfork",
};

fn stmt(code: u16, k: u32) Insn {
    return .{ .code = code, .jt = 0, .jf = 0, .k = k };
}

fn jump(code: u16, k: u32, jt: u8, jf: u8) Insn {
    return .{ .code = code, .jt = jt, .jf = jf, .k = k };
}

fn nr(comptime name: []const u8) u32 {
    return @intCast(@backingInt(@field(linux.SYS, name)));
}

const ld = BPF.LD | BPF.W | BPF.ABS;
const jeq = BPF.JMP | BPF.JEQ | BPF.K;
const ret = BPF.RET | BPF.K;

/// The filter for the native arch, built at compile time (empty where unsupported).
pub const program: []const Insn = if (!supported) &.{} else blk: {
    @setEvalBranchQuota(20_000);
    var p: [6 + 2 * denied.len + 7]Insn = undefined;
    var n: usize = 0;
    p[n] = stmt(ld, arch_at);
    p[n + 1] = jump(jeq, audit_arch, 1, 0);
    p[n + 2] = stmt(ret, RET.KILL_PROCESS);
    p[n + 3] = stmt(ld, nr_at);
    n += 4;
    if (builtin.cpu.arch == .x86_64) {
        p[n] = jump(BPF.JMP | BPF.JGE | BPF.K, x32_bit, 0, 1);
        p[n + 1] = stmt(ret, RET.KILL_PROCESS);
        n += 2;
    }
    for (denied) |name| if (@hasField(linux.SYS, name)) {
        p[n] = jump(jeq, nr(name), 0, 1);
        p[n + 1] = stmt(ret, RET.ERRNO | eperm);
        n += 2;
    };
    p[n] = jump(jeq, nr("clone3"), 0, 1);
    p[n + 1] = stmt(ret, RET.ERRNO | enosys); // glibc falls back to clone
    p[n + 2] = jump(jeq, nr("clone"), 0, 3);
    p[n + 3] = stmt(ld, arg0_at);
    p[n + 4] = jump(BPF.JMP | BPF.JSET | BPF.K, clone_thread, 1, 0);
    p[n + 5] = stmt(ret, RET.ERRNO | eperm);
    p[n + 6] = stmt(ret, RET.ALLOW);
    n += 7;
    const out = p[0..n].*;
    break :blk &out;
};

/// Installs the filter on the calling thread (no_new_privs already set); the errno, or SUCCESS.
pub fn install() linux.E {
    if (!supported) return .NOSYS;
    const prog: Fprog = .{ .len = program.len, .filter = program.ptr };
    return linux.errno(linux.seccomp(linux.SECCOMP.SET_MODE_FILTER, 0, &prog));
}
