//! Freestanding RV32 compilation; the host compiles source but never executes the resulting program.
const std = @import("std");
const Allocator = std.mem.Allocator;
const inputs = @import("runner_frontend_c_inputs.zig");
const cost = @import("runner_frontend_c_cost.zig");
const elf_cache = @import("runner_frontend_elf_cache.zig");
const forkserver = @import("runner_frontend_forkserver.zig");
const split = @import("runner_frontend_split.zig");
const log = std.log.scoped(.runner_frontend);

/// The build's own cap, a hang guard: under a translate deadline the build may use all that is left of it.
pub const build_cap_ms = 1900;

pub const Config = struct {
    clang: []const u8 = "/usr/bin/clang",
    linker: ?[]const u8 = null,
    scratch_root: []const u8 = "/tmp",
    optimize: enum { O0, O1, O2 } = .O1,
    stack_limit: usize = 0,
    collection_stack: bool = false,
    opaque_inputs: bool = false,
    estimated_steps: ?usize = null,
    step_budget: usize = 10522,
    deadline: ?std.Io.Timestamp = null,
};

pub const linker_script =
    \\ENTRY(_start)
    \\MEMORY {
    \\ RAM (rw) : ORIGIN = 0x100, LENGTH = 0x7f00
    \\ DROM (r) : ORIGIN = 0x20000, LENGTH = 128K
    \\ ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
    \\}
    \\PHDRS { text PT_LOAD FLAGS(5); rodata PT_LOAD FLAGS(4); data PT_LOAD FLAGS(6); }
    \\SECTIONS {
    \\ .text : { *(.text.init) *(.text .text.*) } > ROM :text
    \\ .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM :rodata
    \\ .data : { *(.data .data.* .sdata .sdata.*) } > RAM :data
    \\ .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } > RAM :data
    \\ ASSERT(ADDR(.bss) + SIZEOF(.bss) <= 0x7000, "RAM leaves 4K stack")
    \\ /DISCARD/ : { *(.comment) *(.eh_frame*) *(.riscv.attributes) }
    \\}
;

pub const runtime =
    \\typedef unsigned int size_t;
    \\typedef signed char int8_t;
    \\typedef unsigned char uint8_t;
    \\typedef short int16_t;
    \\typedef unsigned short uint16_t;
    \\typedef int int32_t;
    \\typedef unsigned int uint32_t;
    \\typedef long long int64_t;
    \\typedef unsigned long long uint64_t;
    \\typedef _Bool bool;
    \\#define true 1
    \\#define false 0
    \\#define NULL ((void*)0)
    \\void tf_py_print(int n, int b, char *digits); // Python print of an int or bool: assembly, so every build runs the same steps
    \\void *memset(void *destination, int value, size_t count) {
    \\ typedef unsigned int __attribute__((may_alias)) word;
    \\ unsigned char *d = destination; size_t i = 0;
    \\ if(!((size_t)d & 3)) {
    \\  word *w = destination, *end = w + count / 4; word v = (unsigned char)value * 0x01010101u;
    \\  while(w != end) *w++ = v;
    \\  i = count & ~(size_t)3;
    \\ }
    \\ for(; i<count; ++i) d[i]=(unsigned char)value;
    \\ return destination;
    \\}
    \\void *memcpy(void *destination, const void *source, size_t count) {
    \\ typedef unsigned int __attribute__((may_alias)) word;
    \\ unsigned char *d = destination; const unsigned char *s = source; size_t i = 0;
    \\ if(!(((size_t)d | (size_t)s) & 3)) {
    \\  word *w = destination, *end = w + count / 4; const word *v = source;
    \\  while(w != end) *w++ = *v++;
    \\  i = count & ~(size_t)3;
    \\ }
    \\ for(; i<count; ++i) d[i]=s[i];
    \\ return destination;
    \\}
    \\__asm__(".text\n.globl tf_py_print\n.type tf_py_print,@function\n.p2align 2\ntf_py_print:\n"
    \\ "li a7, 1; bnez a1, 3f; bgez a0, 1f; neg t0, a0; li a0, 45; ecall; mv a0, t0\n"
    \\ "1: mv t1, a2; lui t2, 838861; addi t2, t2, -819\n"
    \\ "2: mulhu t3, a0, t2; srli t3, t3, 3; sh2add t4, t3, t3; slli t4, t4, 1; sub t4, a0, t4\n"
    \\ "addi t4, t4, 48; sb t4, 0(t1); addi t1, t1, 1; beqz t3, 4f\n"
    \\ "mulhu a0, t3, t2; srli a0, a0, 3; sh2add t4, a0, a0; slli t4, t4, 1; sub t4, t3, t4\n"
    \\ "addi t4, t4, 48; sb t4, 0(t1); addi t1, t1, 1; bnez a0, 2b\n"
    \\ "4: addi t1, t1, -1; lbu a0, 0(t1); ecall; bne t1, a2, 4b; ret\n"
    \\ "3: beqz a0, 5f; li a0, 84; ecall; li a0, 114; ecall; li a0, 117; ecall; li a0, 101; ecall; ret\n"
    \\ "5: li a0, 70; ecall; li a0, 97; ecall; li a0, 108; ecall; li a0, 115; ecall; li a0, 101; ecall; ret\n"
    \\ ".size tf_py_print, .-tf_py_print\n");
    \\static unsigned int tf_output_count;
    \\static int putchar(int c) {
    \\ ++tf_output_count;
    \\ register int a0 __asm__("a0") = (unsigned char)c;
    \\ register int a7 __asm__("a7") = 1;
    \\ __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
    \\ return (unsigned char)c;
    \\}
    \\static void tf_unsigned(unsigned int n) {
    \\ char digits[10]; int len = 0;
    \\ do { digits[len++] = (char)('0' + n % 10); n /= 10; } while(n);
    \\ while(len) putchar(digits[--len]);
    \\}
    \\static void tf_signed(int n) {
    \\ if(n < 0) { putchar('-'); tf_unsigned(0u - (unsigned int)n); }
    \\ else tf_unsigned((unsigned int)n);
    \\}
    \\static unsigned long long tf_div10(unsigned long long n, unsigned int *remainder) {
    \\ unsigned int high = (unsigned int)(n >> 32), low = (unsigned int)n, r = high % 10;
    \\ unsigned int t = (r << 16) | (low >> 16), q1 = t / 10;
    \\ t = ((t % 10) << 16) | (low & 65535u);
    \\ *remainder = t % 10;
    \\ return ((unsigned long long)(high / 10) << 32) | (q1 << 16) | (t / 10);
    \\}
    \\static void tf_unsigned_wide(unsigned long long n) {
    \\ if(n <= 0xffffffffULL) { tf_unsigned((unsigned int)n); return; }
    \\ char digits[20]; int len = 0; unsigned int remainder;
    \\ do { n = tf_div10(n, &remainder); digits[len++] = (char)('0' + remainder); } while(n);
    \\ while(len) putchar(digits[--len]);
    \\}
    \\static void tf_signed_wide(long long n) {
    \\ unsigned long long magnitude = (unsigned long long)n;
    \\ if(n < 0) { putchar('-'); magnitude = 0ULL - magnitude; }
    \\ tf_unsigned_wide(magnitude);
    \\}
    \\static void tf_print_int(int n) { tf_signed(n); putchar('\n'); }
    \\static int puts(const char *s) { while(*s) putchar(*s++); putchar('\n'); return 0; }
    \\static int printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
    \\static int printf(const char *fmt, ...) {
    \\ unsigned int before = tf_output_count;
    \\ __builtin_va_list args; __builtin_va_start(args, fmt);
    \\ while(*fmt) {
    \\  if(*fmt != '%') { putchar(*fmt++); continue; }
    \\  ++fmt;
    \\  if(fmt[0] == 'l' && fmt[1] == 'l') {
    \\   fmt += 2;
    \\   if(*fmt == 'd' || *fmt == 'i') tf_signed_wide(__builtin_va_arg(args, long long));
    \\   else if(*fmt == 'u') tf_unsigned_wide(__builtin_va_arg(args, unsigned long long));
    \\   else { __builtin_va_end(args); return -1; }
    \\  }
    \\  else if(*fmt == 'd' || *fmt == 'i') tf_signed(__builtin_va_arg(args, int));
    \\  else if(*fmt == 'u') tf_unsigned(__builtin_va_arg(args, unsigned int));
    \\  else if(*fmt == 'c') putchar(__builtin_va_arg(args, int));
    \\  else if(*fmt == 's') { const char *s = __builtin_va_arg(args, const char*); while(*s) putchar(*s++); }
    \\  else if(*fmt == '%') putchar('%');
    \\  else { __builtin_va_end(args); return -1; }
    \\  ++fmt;
    \\ }
    \\ __builtin_va_end(args); return (int)(tf_output_count - before);
    \\}
    \\int main(void);
    \\__asm__(".section .text.init,\"ax\"\n.globl _start\n_start:\nli sp, 32768\ncall main\nli a7, 93\necall\n1: j 1b\n");
;

/// Wrap the checked Python body for compile(); callers own the returned source.
pub fn wrapBody(a: Allocator, body: []const u8) Allocator.Error![]u8 {
    return std.fmt.allocPrint(a, "int main(void) {{\n{s}}}\n", .{body});
}

/// Returns owned ELF32 bytes, or abstains on invalid source, unavailable toolchain or compilation failure.
pub fn compile(a: Allocator, io: std.Io, source: []const u8, config: Config) Allocator.Error!?[]u8 {
    if (config.deadline) |end| if (std.Io.Clock.awake.now(io).toNanoseconds() >= end.toNanoseconds()) return null;
    const normalized = try normalizeWithTrust(a, source, config.opaque_inputs) orelse return null;
    defer a.free(normalized);
    const cached = imageKey(normalized, config);
    if (cached) |k| if (try elf_cache.get(a, io, k)) |bytes| return bytes;
    if (!std.fs.path.isAbsolute(config.scratch_root)) return null;
    const parent = std.Io.Dir.cwd().openDir(io, config.scratch_root, .{}) catch return null;
    defer parent.close(io);
    var name_buf: [64]u8 = undefined;
    const name = scratch(io, parent, &name_buf) orelse return null;
    defer parent.deleteTree(io, name) catch {};
    const dir = parent.openDir(io, name, .{ .follow_symlinks = false }) catch return null;
    defer dir.close(io);
    const complete = try std.fmt.allocPrint(a, "{s}\n{s}\n", .{ runtime, normalized });
    defer a.free(complete);
    if (!write(io, dir, "program.c", complete) or !write(io, dir, "rom.ld", linker_script)) return null;
    const path = try std.fs.path.join(a, &.{ config.scratch_root, name });
    defer a.free(path);
    const src = try std.fs.path.join(a, &.{ path, "program.c" });
    defer a.free(src);
    const elf = try std.fs.path.join(a, &.{ path, "program.elf" });
    defer a.free(elf);
    const link_path = try std.fs.path.join(a, &.{ path, "rom.ld" });
    defer a.free(link_path);
    const linker_arg = if (config.linker) |value|
        try std.fmt.allocPrint(a, "--ld-path={s}", .{value})
    else
        try a.dupe(u8, "-fuse-ld=lld");
    defer a.free(linker_arg);
    if (config.step_budget != 0) {
        if (config.estimated_steps) |steps| if (steps > config.step_budget) return null;
    }
    if (!config.opaque_inputs or (config.step_budget != 0 and config.estimated_steps == null)) {
        const ast = try forkserver.run(a, io, path, &.{ config.clang, "--target=riscv32-unknown-elf", "-march=rv32im_zba", "-mabi=ilp32", "-ffreestanding", "-fno-builtin", "-nostdinc", "-Werror=format", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-x", "c", src }, .{
            .stdout = 4 * 1024 * 1024,
            .stderr = 16384,
            .timeout = remaining(io, config, 500) orelse return null,
        }) orelse return null;
        defer ast.deinit(a);
        if (!ast.ok) return null;
        var instrumented: ?[]u8 = null;
        var restoration_steps: usize = 0;
        defer if (instrumented) |text| a.free(text);
        if (!config.opaque_inputs) {
            const lowered = try inputs.lowerWithCost(a, normalized, ast.stdout, runtime.len + 1) orelse return null;
            instrumented = lowered.source;
            restoration_steps = lowered.restoration_steps;
        }
        if (config.step_budget != 0 and config.estimated_steps == null) {
            const startup = restoration_steps;
            if (startup >= config.step_budget) return null;
            _ = try cost.estimate(a, ast.stdout, runtime.len + 1, config.step_budget - startup) orelse return null;
        }
        if (instrumented) |text| {
            const rewritten = try std.fmt.allocPrint(a, "{s}\n{s}\n", .{ runtime, text });
            defer a.free(rewritten);
            const file = dir.createFile(io, "program.c", .{ .truncate = true, .permissions = .fromMode(0o600) }) catch return null;
            defer file.close(io);
            file.writeStreamingAll(io, rewritten) catch return null;
        }
    }
    const build_start = std.Io.Clock.awake.now(io);
    if (config.opaque_inputs and split.eligible(normalized)) {
        const split_timeout = remaining(io, config, build_cap_ms) orelse return null;
        if (try split.build(a, io, path, normalized, config, split_timeout)) |built| {
            defer a.free(built.su);
            if (config.stack_limit != 0 and !safeStackUsage(built.su, config.stack_limit)) {
                a.free(built.elf);
                return null;
            }
            if (validElf(built.elf)) {
                if (cached) |k| elf_cache.put(io, k, built.elf);
                return built.elf;
            }
            a.free(built.elf);
        }
    }
    const build_timeout = remaining(io, config, build_cap_ms) orelse {
        log.warn("clang stopped after {d} ms by the compile time limit; the call declines", .{@divTrunc(build_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds(), std.time.ns_per_ms)});
        return null;
    };
    const result = try forkserver.run(a, io, path, &.{
        config.clang,           "--target=riscv32-unknown-elf", "-march=rv32im_zba",   "-mabi=ilp32",
        switch (config.optimize) {
            .O0 => "-O0",
            .O1 => "-O1",
            .O2 => "-O2",
        },
        "-ffreestanding",       "-fno-builtin",                 "-nostdinc",           "-nostdlib",
        "-fno-stack-protector", "-fno-pic",                     "-mno-relax",          "-Werror=format",
        "-Wno-format-security", "-fstack-usage",                "-fno-unwind-tables",  "-fno-asynchronous-unwind-tables",
        linker_arg,             "-Wl,--no-relax",               "-Wl,--build-id=none", "-Wl,-T",
        link_path,              "-x",                           "c",                   src,
        "-o",                   elf,
    }, .{ .stdout = 4096, .stderr = 16384, .timeout = build_timeout }) orelse {
        const spent = build_start.durationTo(std.Io.Clock.awake.now(io));
        if (spent.toNanoseconds() >= build_timeout.toNanoseconds())
            log.warn("clang stopped after {d} ms by the compile time limit; the call declines", .{@divTrunc(spent.toNanoseconds(), std.time.ns_per_ms)});
        return null;
    };
    defer result.deinit(a);
    if (!result.ok) return null;
    if (config.stack_limit != 0) {
        const usage = dir.readFileAlloc(io, "program.su", a, .limited(65536)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        defer a.free(usage);
        if (!stackUsage(usage, config.stack_limit, config.collection_stack)) return null;
    }
    const bytes = dir.readFileAlloc(io, "program.elf", a, .limited(1 << 20)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (!validElf(bytes)) {
        a.free(bytes);
        return null;
    }
    if (cached) |k| elf_cache.put(io, k, bytes);
    return bytes;
}

/// A fresh private directory under `parent` (tf-rv32-<random>); its name lives in `buf`.
pub fn scratch(io: std.Io, parent: std.Io.Dir, buf: *[64]u8) ?[]const u8 {
    var nonce: [16]u8 = undefined;
    for (0..8) |_| {
        io.random(&nonce);
        const name = std.fmt.bufPrint(buf, "tf-rv32-{x}", .{std.mem.readInt(u128, &nonce, .little)}) catch unreachable;
        parent.createDir(io, name, .fromMode(0o700)) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return null,
        };
        return name;
    }
    return null;
}

fn imageKey(normalized: []const u8, config: Config) ?elf_cache.Key {
    if (!elf_cache.wanted(config.opaque_inputs, config.step_budget, config.estimated_steps)) return null;
    return elf_cache.key(normalized, config.clang, config.linker, @backingInt(config.optimize), config.stack_limit, config.collection_stack);
}

/// The image-cache key compile() uses for `source` under `config`, or null when it would not cache (tests).
pub fn cacheKey(a: Allocator, source: []const u8, config: Config) Allocator.Error!?elf_cache.Key {
    const normalized = try normalizeWithTrust(a, source, config.opaque_inputs) orelse return null;
    defer a.free(normalized);
    return imageKey(normalized, config);
}

fn remaining(io: std.Io, config: Config, milliseconds: i64) ?std.Io.Duration {
    const cap = std.Io.Duration.fromMilliseconds(milliseconds);
    const end = config.deadline orelse return cap;
    const left = std.Io.Clock.awake.now(io).durationTo(end);
    if (left.toNanoseconds() <= 0) return null;
    return .fromNanoseconds(@min(left.toNanoseconds(), cap.toNanoseconds()));
}

// Summed static frames bound every call chain in the trusted, recursion-free Python lowering.
pub fn safeStackUsage(report: []const u8, limit: usize) bool {
    return stackUsage(report, limit, false);
}

fn stackUsage(report: []const u8, limit: usize, collections: bool) bool {
    var total: usize = 0;
    var functions: usize = 0;
    var lines = std.mem.splitScalar(u8, report, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r");
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const label = fields.next() orelse return false;
        const name_start = if (std.mem.lastIndexOfScalar(u8, label, ':')) |at| at + 1 else 0;
        const name = label[name_start..];
        const recursive = std.mem.startsWith(u8, name, "equal") or std.mem.startsWith(u8, name, "compare") or
            std.mem.startsWith(u8, name, "repr");
        const factor: usize = if (collections and recursive) 9 else 1;
        const size = std.fmt.parseInt(usize, fields.next() orelse return false, 10) catch return false;
        const kind = fields.next() orelse return false;
        if (fields.next() != null or !std.mem.eql(u8, kind, "static") or size > (limit - total) / factor) return false;
        total += size * factor;
        functions += 1;
    }
    return functions != 0;
}

test "trusted Python stack admission sums static frames and rejects unbounded reports" {
    try std.testing.expect(safeStackUsage("p.c:1:main\t128\tstatic\np.c:2:f\t256\tstatic\n", 384));
    try std.testing.expect(!safeStackUsage("p.c:1:main\t128\tstatic\np.c:2:f\t256\tstatic\n", 383));
    try std.testing.expect(!safeStackUsage("p.c:1:main\t32\tdynamic\n", 4096));
    try std.testing.expect(!safeStackUsage("p.c:1:main\tbad\tstatic\n", 4096));
    try std.testing.expect(!safeStackUsage("", 4096));
}

fn write(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []const u8) bool {
    const f = dir.createFile(io, name, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return false;
    defer f.close(io);
    f.writeStreamingAll(io, bytes) catch return false;
    return true;
}

fn normalize(a: Allocator, source: []const u8) Allocator.Error!?[]u8 {
    return normalizeWithTrust(a, source, false);
}

pub fn normalizeWithTrust(a: Allocator, source: []const u8, trusted: bool) Allocator.Error!?[]u8 {
    // Fixed integer support plus bounded DAG data may exceed the raw-cell source limit.
    const max_source: usize = if (trusted) 32768 else 16384;
    if (source.len == 0 or source.len > max_source or std.mem.indexOfScalar(u8, source, 0) != null) return null;
    if (std.mem.indexOf(u8, source, "\\\n") != null or std.mem.indexOf(u8, source, "\\\r\n") != null) return null;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "#include")) {
            const header = std.mem.trim(u8, trimmed[8..], " \t");
            const allowed = [_][]const u8{ "<stdio.h>", "<stdint.h>", "<stddef.h>", "<stdbool.h>" };
            var ok = false;
            for (allowed) |include| {
                if (std.mem.eql(u8, header, include)) ok = true;
            }
            if (!ok) return null;
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "#define ")) {
            var fields = std.mem.tokenizeAny(u8, trimmed[8..], " \t");
            const name = fields.next() orelse return null;
            const value = fields.next() orelse return null;
            if (fields.next() != null or name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return null;
            for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return null;
            _ = std.fmt.parseInt(u32, value, 0) catch return null;
        } else if (std.mem.indexOfScalar(u8, line, '#') != null) return null;
        try out.appendSlice(a, line);
        try out.append(a, '\n');
    }
    const forbidden = [_][]const u8{ "__", "asm", "_Pragma", "??", "%:", "\\\n", "\\\r" };
    for (forbidden) |token| {
        if (trusted and (std.mem.eql(u8, token, "__") or std.mem.eql(u8, token, "asm"))) continue;
        if (std.mem.indexOf(u8, out.items, token) != null) return null;
    }
    if (!safeFormats(out.items)) return null;
    return try out.toOwnedSlice(a);
}

pub fn validElf(b: []const u8) bool {
    if (b.len < 52 or !std.mem.eql(u8, b[0..4], "\x7fELF") or b[4] != 1 or b[5] != 1 or b[6] != 1)
        return false;
    if (read(u16, b, 16) != 2 or read(u16, b, 18) != 243 or read(u32, b, 20) != 1 or
        read(u16, b, 40) != 52 or read(u16, b, 42) != 32 or read(u32, b, 36) != 0) return false;
    const entry = read(u32, b, 24);
    if (entry < 0x40000 or entry >= 0x80000 or entry % 4 != 0) return false;
    const off: usize = read(u32, b, 28);
    const count: usize = read(u16, b, 44);
    if (count == 0 or count > 16 or off > b.len or count * 32 > b.len - off) return false;
    var entry_loaded = false;
    for (0..count) |i| {
        const ph = off + 32 * i;
        const kind = read(u32, b, ph);
        if (kind == 2 or kind == 3 or kind == 7) return false;
        if (kind != 1) continue;
        const start: u64 = read(u32, b, ph + 8);
        const file_off: usize = read(u32, b, ph + 4);
        const file_size: usize = read(u32, b, ph + 16);
        const size: u64 = read(u32, b, ph + 20);
        const flags = read(u32, b, ph + 24);
        if (size == 0) continue;
        if (file_size > size or file_off > b.len or file_size > b.len - file_off) return false;
        const end = start + size;
        const ram = start >= 0x100 and end <= 0x7000 and flags & 1 == 0;
        const drom = start >= 0x20000 and end <= 0x40000 and flags & 3 == 0;
        const rom = start >= 0x40000 and end <= 0x80000 and flags & 2 == 0;
        if (!ram and !drom and !rom) return false;
        if (rom and flags & 1 != 0 and entry >= start and entry < start + file_size) entry_loaded = true;
    }
    return entry_loaded;
}

fn read(comptime T: type, b: []const u8, off: usize) T {
    return std.mem.readInt(T, b[off..][0..@sizeOf(T)], .little);
}

test "missing toolchain abstains" {
    try std.testing.expectEqual(@as(?[]u8, null), try compile(std.testing.allocator, std.testing.io, "int main(void) { return 0; }", .{ .clang = "/nonexistent-tensorfold-clang" }));
}

test "source cannot access host includes or inject assembly" {
    const bad = [_][]const u8{ "#include </etc/passwd>", "#define X(x) x", "asm(\".incbin x\");", "_Pragma(\"x\")", "int x = __FILE__;", "%:include </etc/passwd>", "pri\\\nntf(\"%f\", 1.0);" };
    for (bad) |s| try std.testing.expectEqual(@as(?[]u8, null), try normalize(std.testing.allocator, s));
    try std.testing.expect(!validElf("not an ELF"));
}

/// Checks literal printf conversion syntax; Clang checks argument types during compilation.
pub fn safeFormats(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], "//")) {
            while (i < s.len and s[i] != '\n') i += 1;
            continue;
        }
        if (s[i] == '/' and i + 1 < s.len and s[i + 1] == '*') {
            const end = std.mem.indexOf(u8, s[i + 2 ..], "*/") orelse return false;
            i += end + 4;
            continue;
        }
        if (s[i] == '"' or s[i] == '\'') {
            const quote = s[i];
            i += 1;
            while (i < s.len and s[i] != quote) : (i += 1) {
                if (s[i] == '\\') i += 1;
            }
            if (i >= s.len) return false;
            i += 1;
            continue;
        }
        if (std.ascii.isAlphabetic(s[i]) or s[i] == '_') {
            const start = i;
            while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
            if (!std.mem.eql(u8, s[start..i], "printf")) continue;
            skipSpace(s, &i);
            if (i == s.len or s[i] != '(') return false;
            i += 1;
            skipSpace(s, &i);
            if (i == s.len or s[i] != '"') return false;
            i += 1;
            while (i < s.len and s[i] != '"') : (i += 1) {
                if (s[i] == '\\') {
                    i += 1;
                    if (i == s.len or std.mem.indexOfScalar(u8, "nrtvabf\\\"?", s[i]) == null) return false;
                } else if (s[i] == '%') {
                    i += 1;
                    if (std.mem.startsWith(u8, s[i..], "ll")) {
                        i += 2;
                        if (i == s.len or std.mem.indexOfScalar(u8, "diu", s[i]) == null) return false;
                    } else if (i == s.len or std.mem.indexOfScalar(u8, "diucs%", s[i]) == null) return false;
                }
            }
            if (i == s.len) return false;
            i += 1;
            skipSpace(s, &i);
            if (i == s.len or (s[i] != ',' and s[i] != ')')) return false;
            continue;
        }
        i += 1;
    }
    return true;
}

fn skipSpace(s: []const u8, i: *usize) void {
    while (i.* < s.len and std.ascii.isWhitespace(s[i.*])) i.* += 1;
}

test "printf admits literal integer formats and refuses formats the runtime cannot reproduce" {
    try std.testing.expect(safeFormats("printf(\"%d %u %c %s %%\\n\", 1, 2u, 'c', \"text\");"));
    const bad = [_][]const u8{
        "printf(\"%f\", 1.0);", "printf(\"%ld\", 1L);",     "printf(\"%04d\", 1);",
        "printf(fmt, 1);",      "printf(\"\\x25f\", 1.0);", "printf(\"%\" \"f\", 1.0);",
        "void *p = printf;",
    };
    for (bad) |s| try std.testing.expect(!safeFormats(s));
}

pub fn hasMain(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], "//")) {
            while (i < s.len and s[i] != '\n') i += 1;
        } else if (s[i] == '/' and i + 1 < s.len and s[i + 1] == '*') {
            const end = std.mem.indexOf(u8, s[i + 2 ..], "*/") orelse return false;
            i += end + 4;
        } else if (s[i] == '"' or s[i] == '\'') {
            const quote = s[i];
            i += 1;
            while (i < s.len and s[i] != quote) : (i += 1) {
                if (s[i] == '\\') i += 1;
            }
            if (i < s.len) i += 1;
        } else if (std.ascii.isAlphabetic(s[i]) or s[i] == '_') {
            const start = i;
            while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
            if (std.mem.eql(u8, s[start..i], "main")) {
                skipSpace(s, &i);
                if (i < s.len and s[i] == '(') return true;
            }
        } else i += 1;
    }
    return false;
}

test "generated support has a bounded size allowance while raw cells retain their limit" {
    const a = std.testing.allocator;
    const padding = try a.alloc(u8, 17000);
    defer a.free(padding);
    @memset(padding, ' ');
    const support = try std.fmt.allocPrint(a, "{s}int main(void) {{ return 0; }}", .{padding});
    defer a.free(support);
    try std.testing.expectEqual(@as(?[]u8, null), try normalizeWithTrust(a, support, false));
    const result = (try normalizeWithTrust(a, support, true)).?;
    defer a.free(result);
    const oversized = try a.alloc(u8, 32769);
    defer a.free(oversized);
    @memset(oversized, ' ');
    try std.testing.expectEqual(@as(?[]u8, null), try normalizeWithTrust(a, oversized, true));
}

// The closed collections runtime allows at most nine frames through its recursive value walkers.
test "collections stack admission charges every bounded recursive frame" {
    const report = "p.c:1:main\t128\tstatic\np.c:2:equal\t32\tstatic\n" ++
        "p.c:3:compare.specialized\t48\tstatic\np.c:4:repr\t64\tstatic\n";
    try std.testing.expect(safeStackUsage(report, 272));
    try std.testing.expect(stackUsage(report, 1424, true));
    try std.testing.expect(!stackUsage(report, 1423, true));
    try std.testing.expect(!stackUsage("p.c:2:equal\t32\tdynamic\n", 4096, true));
}
