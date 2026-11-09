//! Runtime compiled once per clang and level; a body using none of its statics compiles alone, as in the whole file.
const std = @import("std");
const Allocator = std.mem.Allocator;
const compile = @import("runner_frontend_program_compile.zig");
const forkserver = @import("runner_frontend_forkserver.zig");
const rvlink = @import("runner_frontend_rvlink.zig");
const log = std.log.scoped(.runner_frontend);

/// The whole file's runtime compiled alone: its static printf and helpers drop out, memset and memcpy stay.
pub const runtime = compile.runtime;

/// What a body sees of the runtime: its types, macros and Python's print, and the two functions it may call.
pub const declarations = compile.runtime[0..std.mem.indexOf(u8, compile.runtime, "void *memset").?] ++
    \\void *memset(void *destination, int value, size_t count);
    \\void *memcpy(void *destination, const void *source, size_t count);
    \\
;

/// A body may compile alone when it names no runtime static or inlinable call; tf_py_print is assembly in both builds.
pub fn eligible(body: []const u8) bool {
    var i: usize = 0;
    while (i < body.len) {
        const c = body[i];
        if (!(std.ascii.isAlphabetic(c) or c == '_')) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < body.len and (std.ascii.isAlphanumeric(body[i]) or body[i] == '_')) i += 1;
        const word = body[start..i];
        for ([_][]const u8{ "printf", "putchar", "puts", "memset", "memcpy" }) |helper| if (std.mem.eql(u8, word, helper)) return false;
        if (std.mem.startsWith(u8, word, "tf_") and !std.mem.eql(u8, word, "tf_py_print")) return false;
    }
    return true;
}

const Prebuilt = struct { clang: [256]u8 = undefined, clang_len: usize = 0, optimize: u8 = 0, object: []u8 = &.{}, su: []u8 = &.{} };

var lock: std.Io.Mutex = .init;
var prebuilt: [4]Prebuilt = @splat(.{});
var published: std.atomic.Value(usize) = .init(0); // entries below this never change, so calls read them unlocked

fn level(optimize: anytype) []const u8 {
    return switch (optimize) {
        .O0 => "-O0",
        .O1 => "-O1",
        .O2 => "-O2",
    };
}

/// The compile step's flags for an object: the whole-file build's, without the link.
fn objectArgv(config: compile.Config, src: []const u8, out: []const u8) [22][]const u8 {
    return .{
        config.clang,           "--target=riscv32-unknown-elf", "-march=rv32im_zba",  "-mabi=ilp32",
        level(config.optimize), "-ffreestanding",               "-fno-builtin",       "-nostdinc",
        "-fno-stack-protector", "-fno-pic",                     "-mno-relax",         "-Werror=format",
        "-Wno-format-security", "-fstack-usage",                "-fno-unwind-tables", "-fno-asynchronous-unwind-tables",
        "-c",                   "-x",                           "c",                  src,
        "-o",                   out,
    };
}

fn find(config: compile.Config) ?Prebuilt {
    for (prebuilt[0..published.load(.acquire)]) |e| if (e.optimize == @backingInt(config.optimize) and std.mem.eql(u8, e.clang[0..e.clang_len], config.clang)) return e;
    return null;
}

/// The runtime object and stack report for `config`, else compiled in empty `path`; a call (`wait` false) never queues.
fn runtimeFor(a: Allocator, io: std.Io, path: []const u8, config: compile.Config, timeout: std.Io.Duration, wait: bool) Allocator.Error!?Prebuilt {
    if (config.clang.len > 256) return null;
    if (find(config)) |e| return e;
    if (wait) lock.lockUncancelable(io) else if (!lock.tryLock()) return null;
    defer lock.unlock(io);
    if (find(config)) |e| return e;
    const n = published.load(.monotonic);
    if (n == prebuilt.len) return null;
    const src = try std.fs.path.join(a, &.{ path, "runtime.c" });
    defer a.free(src);
    const obj = try std.fs.path.join(a, &.{ path, "runtime.o" });
    defer a.free(obj);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = src, .data = runtime }) catch return null;
    var argv = objectArgv(config, src, obj);
    const ran = try forkserver.run(a, io, path, &argv, .{ .stdout = 4096, .stderr = 16384, .timeout = timeout }) orelse return null;
    defer ran.deinit(a);
    if (!ran.ok) return null;
    const object = readFile(io, path, "runtime.o", 1 << 20) orelse return null;
    const su = readFile(io, path, "runtime.su", 65536) orelse {
        std.heap.page_allocator.free(object);
        return null;
    };
    var entry: Prebuilt = .{ .optimize = @backingInt(config.optimize), .object = object, .su = su, .clang_len = config.clang.len };
    @memcpy(entry.clang[0..config.clang.len], config.clang);
    prebuilt[n] = entry;
    published.store(n + 1, .release);
    return entry;
}

fn readFile(io: std.Io, dir: []const u8, name: []const u8, limit: usize) ?[]u8 {
    var buf: [4096]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(limit)) catch null;
}

/// A linked image and the summed stack report of runtime and body; the caller frees both.
pub const Built = struct {
    elf: []u8,
    su: []u8,

    pub fn deinit(b: Built, a: Allocator) void {
        a.free(b.elf);
        a.free(b.su);
    }
};

/// The body compiled alone in `path`, linked with the runtime; null on any failure (the whole file builds instead).
pub fn build(a: Allocator, io: std.Io, path: []const u8, body: []const u8, config: compile.Config, timeout: std.Io.Duration) Allocator.Error!?Built {
    // the identity tests cover O1 and O2, the levels calls use
    if (config.optimize == .O0) return null;
    const rt = try runtimeFor(a, io, path, config, timeout, false) orelse return null;
    const src = try std.fs.path.join(a, &.{ path, "body.c" });
    defer a.free(src);
    const obj = try std.fs.path.join(a, &.{ path, "body.o" });
    defer a.free(obj);
    const text = try std.mem.concat(a, u8, &.{ declarations, body });
    defer a.free(text);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = src, .data = text }) catch return null;
    var argv = objectArgv(config, src, obj);
    const ran = try forkserver.run(a, io, path, &argv, .{ .stdout = 4096, .stderr = 16384, .timeout = timeout }) orelse return null;
    defer ran.deinit(a);
    if (!ran.ok) return null;
    const object = readFile(io, path, "body.o", 1 << 20) orelse return null;
    defer std.heap.page_allocator.free(object);
    // an aggregate copy or fill the compiler lowered to a call: the whole file may inline that function
    if (rvlink.references(object, "memset") or rvlink.references(object, "memcpy")) return null;
    const body_su = readFile(io, path, "body.su", 65536) orelse return null;
    defer std.heap.page_allocator.free(body_su);
    const elf = try rvlink.link(a, &.{ rt.object, object }) orelse {
        log.debug("in-process link declined a body; ld.lld links it", .{});
        return null;
    };
    errdefer a.free(elf);
    return .{ .elf = elf, .su = try std.mem.concat(a, u8, &.{ rt.su, body_su }) };
}

/// Starts the fork server and compiles the runtime at O1 and O2 (numeric programs); best effort, off the call path.
pub fn warm(a: Allocator, io: std.Io, config: compile.Config) Allocator.Error!void {
    if (!std.fs.path.isAbsolute(config.scratch_root)) return;
    const parent = std.Io.Dir.cwd().openDir(io, config.scratch_root, .{}) catch return;
    defer parent.close(io);
    var name_buf: [64]u8 = undefined;
    const name = compile.scratch(io, parent, &name_buf) orelse return;
    defer parent.deleteTree(io, name) catch {};
    const path = try std.fs.path.join(a, &.{ config.scratch_root, name });
    defer a.free(path);
    _ = try forkserver.start(a, io, config.clang, config.linker, path);
    for ([_]@TypeOf(config.optimize){ .O1, .O2 }) |optimize| {
        var at = config;
        at.optimize = optimize;
        _ = try runtimeFor(a, io, path, at, .fromMilliseconds(10_000), true);
    }
}
