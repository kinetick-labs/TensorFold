//! Split compile vs the whole file: same functions, ld.lld's loaded image, served clang answering as spawned.
const std = @import("std");
const split = @import("runner_frontend_split.zig");
const rvlink = @import("runner_frontend_rvlink.zig");
const forkserver = @import("runner_frontend_forkserver.zig");
const compile = @import("runner_frontend_program_compile.zig");
const program = @import("runner_frontend_program.zig");
const blocks = @import("runner_frontend_python_blocks.zig");
const spawn = @import("runner_frontend_spawn.zig");
const t = std.testing;

/// The 10 slow ELF cells' functions (S3 window, run-check and earlier dr cells).
const cells = [_][]const u8{
    "def square_perimeter(side):\n    return 4 * side\n\nassert square_perimeter(10) == 40\nprint(square_perimeter(10))",
    "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nfor n in [0, 1, 5, 12]:\n    print(factorial(n))\n",
    "def centered_hexagonal_number(n):\n    return 3 * n * (n - 1) + 1\n\nassert centered_hexagonal_number(10) == 271\nprint(centered_hexagonal_number(10))",
    "def check(n):\n    rev = int(str(n)[::-1])\n    return n == 2 * rev - 1\n\nprint(check(70))\nprint(check(13))\nprint(check(0))\n",
    "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nassert pos_count([1, -2, 3, -4]) == 2\nprint(pos_count([1, -2, 3, -4]))",
    "def tetrahedral_number(n):\n    return n * (n + 1) * (n + 2) // 6\n\nassert tetrahedral_number(5) == 35\nprint(tetrahedral_number(5))",
};

test "split: a body compiles alone only when it names none of the runtime's statics or inlinable calls" {
    try t.expect(split.eligible("static void py_out(int c) {} int main(void) { py_out(10); return 0; }"));
    try t.expect(!split.eligible("int main(void) { printf(\"%d\", 1); return 0; }"));
    try t.expect(!split.eligible("int main(void) { char b[8]; memset(b, 0, 8); return b[1]; }"));
    try t.expect(!split.eligible("int main(void) { puts(\"y\"); }"));
    try t.expect(!split.eligible("int main(void) { tf_print_int(3); }"));
    try t.expect(split.eligible("int main(void) { char d[10]; tf_py_print(-3, 0, d); }"));
    try t.expect(!split.eligible("int main(void) { putchar(10); }"));
    try t.expect(std.mem.indexOf(u8, split.declarations, "printf") == null);
}

test "in-process link: the relocations the compile step emits patch as ld.lld does; others decline" {
    var seg: [8]u8 = undefined;
    std.mem.writeInt(u32, seg[0..4], 0x00000537, .little);
    try t.expect(rvlink.patch(&seg, 0, 26, 0x20abc, 0));
    try t.expectEqual(@as(u32, 0x00021537), std.mem.readInt(u32, seg[0..4], .little));
    std.mem.writeInt(u32, seg[0..4], 0x00050513, .little);
    try t.expect(rvlink.patch(&seg, 0, 27, 0x20abc, 0));
    try t.expectEqual(@as(u32, 0xabc50513), std.mem.readInt(u32, seg[0..4], .little));
    std.mem.writeInt(u32, seg[0..4], 0x00a52023, .little);
    try t.expect(rvlink.patch(&seg, 0, 28, 0x123, 0));
    try t.expectEqual(@as(u32, 0x12a521a3), std.mem.readInt(u32, seg[0..4], .little));
    std.mem.writeInt(u32, seg[0..4], 0x00000097, .little);
    std.mem.writeInt(u32, seg[4..8], 0x000080e7, .little);
    try t.expect(rvlink.patch(&seg, 0, 19, 0x40000 - 0x10, 0x40000));
    try t.expectEqual(@as(u32, 0x00000097), std.mem.readInt(u32, seg[0..4], .little));
    try t.expectEqual(@as(u32, 0xff0080e7), std.mem.readInt(u32, seg[4..8], .little));
    try t.expect(!rvlink.patch(&seg, 0, 51, 0, 0));
    try t.expect(!rvlink.patch(&seg, 6, 1, 0, 0));
    try t.expectEqual(@as(?[]u8, null), try rvlink.link(t.allocator, &.{"not an object"}));
}

fn clangOr() ?[]const u8 {
    const c = std.c.getenv("TF_RUNNER_CLANG") orelse return null;
    return std.mem.span(c);
}

/// A cell's C source as the frontend builds it (the plain lowering, else the structured one).
fn cellSource(a: std.mem.Allocator, code: []const u8) ![]u8 {
    if (try program.lowerWithCost(a, code)) |body| {
        defer a.free(body.source);
        return program.wrapBody(a, body.source);
    }
    return (try blocks.lowerWithCost(a, code)).?.source;
}

fn run(a: std.mem.Allocator, argv: []const []const u8) !void {
    const r = (try spawn.run(a, t.io, argv, .{ .stdout = 1 << 20, .stderr = 1 << 16, .timeout = .fromMilliseconds(20_000) })).?;
    if (!r.ok) std.debug.print("{s}\n", .{r.stderr});
    try t.expect(r.ok);
}

/// Function bodies of clang's assembly by name, compiler-local labels renamed and comments dropped.
fn functions(a: std.mem.Allocator, asm_text: []const u8) !std.StringHashMapUnmanaged([]const u8) {
    var out: std.StringHashMapUnmanaged([]const u8) = .empty;
    var want: ?[]const u8 = null;
    var name: ?[]const u8 = null;
    var body: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, asm_text, '\n');
    while (lines.next()) |raw| {
        const hash = std.mem.indexOf(u8, raw, " #") orelse std.mem.indexOf(u8, raw, "\t#") orelse raw.len;
        const line = std.mem.trimEnd(u8, raw[0..hash], " \t");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "\t.type\t") and std.mem.endsWith(u8, line, ",@function")) {
            want = line[7 .. line.len - 10];
            continue;
        }
        if (want != null and line.len == want.?.len + 1 and std.mem.startsWith(u8, line, want.?) and line[line.len - 1] == ':') {
            name = want;
            want = null;
            body = .empty;
            continue;
        }
        if (name == null) continue;
        if (std.mem.startsWith(u8, line, ".Lfunc_end")) {
            try out.put(a, name.?, try body.toOwnedSlice(a));
            name = null;
            continue;
        }
        try body.appendSlice(a, try local(a, line));
        try body.append(a, '\n');
    }
    return out;
}

/// .LBB3_2 -> .LBB_2 and the like: label numbers follow function order, which differs between the two files.
fn local(a: std.mem.Allocator, line: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < line.len) {
        inline for (.{ ".LBB", ".LJTI", ".Ltmp", ".Lpcrel_hi", ".L.str", ".Lfunc_end" }) |p| {
            if (std.mem.startsWith(u8, line[i..], p)) {
                try out.appendSlice(a, p);
                i += p.len;
                while (i < line.len and (std.ascii.isDigit(line[i]) or (std.mem.eql(u8, p, ".L.str") and line[i] == '.'))) i += 1;
                break;
            }
        } else {
            try out.append(a, line[i]);
            i += 1;
        }
    }
    return out.items;
}

const flags = [_][]const u8{ "--target=riscv32-unknown-elf", "-march=rv32im_zba", "-mabi=ilp32", "-ffreestanding", "-fno-builtin", "-nostdinc", "-fno-stack-protector", "-fno-pic", "-mno-relax", "-Werror=format", "-Wno-format-security", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables" };

fn assembly(a: std.mem.Allocator, clang: []const u8, dir: []const u8, name: []const u8, text: []const u8, level: []const u8) ![]const u8 {
    const src = try std.fmt.allocPrint(a, "{s}/{s}.c", .{ dir, name });
    const out = try std.fmt.allocPrint(a, "{s}/{s}.s", .{ dir, name });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = src, .data = text });
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ clang, level });
    try argv.appendSlice(a, &flags);
    try argv.appendSlice(a, &.{ "-S", "-x", "c", src, "-o", out });
    try run(a, argv.items);
    return std.Io.Dir.cwd().readFileAlloc(t.io, out, a, .limited(1 << 22));
}

test "configured split compile: every function of the slow ELF cells equals the whole-file compile's (O1, O2)" {
    const clang = clangOr() orelse return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    for (cells) |code| {
        const source = try cellSource(a, code);
        const body = (try compile.normalizeWithTrust(a, source, true)).?;
        try t.expect(split.eligible(body));
        for ([_][]const u8{ "-O1", "-O2" }) |level| {
            const whole = try functions(a, try assembly(a, clang, dir, "whole", try std.mem.concat(a, u8, &.{ compile.runtime, "\n", body, "\n" }), level));
            const alone = try functions(a, try assembly(a, clang, dir, "body", try std.mem.concat(a, u8, &.{ split.declarations, body }), level));
            const rt = try functions(a, try assembly(a, clang, dir, "rt", split.runtime, level));
            var it = whole.iterator();
            var count: usize = 0;
            while (it.next()) |e| : (count += 1) {
                const other = alone.get(e.key_ptr.*) orelse rt.get(e.key_ptr.*) orelse {
                    std.debug.print("{s}: {s} missing from the split compile\n", .{ level, e.key_ptr.* });
                    return error.TestExpectedEqual;
                };
                if (!std.mem.eql(u8, other, e.value_ptr.*)) std.debug.print("{s}: {s} differs\n{s}\n", .{ level, e.key_ptr.*, code });
                try t.expectEqualStrings(e.value_ptr.*, other);
            }
            try t.expect(count >= 3);
        }
    }
}

/// (function, frame size) of each stack report line, sorted: the multiset the admission check sums.
fn frames(a: std.mem.Allocator, report: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, report, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t').?;
        const name = line[std.mem.lastIndexOfScalar(u8, line[0..tab], ':').? + 1 .. tab];
        try out.append(a, try std.fmt.allocPrint(a, "{s} {s}", .{ name, line[tab + 1 ..] }));
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    return out.items;
}

test "configured split build: the stack report sums the frames of the whole file's (O1, O2)" {
    const clang = clangOr() orelse return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    for (cells) |code| {
        const body = (try compile.normalizeWithTrust(a, try cellSource(a, code), true)).?;
        for ([_]@TypeOf(@as(compile.Config, .{}).optimize){ .O1, .O2 }) |optimize| {
            const src = try std.fmt.allocPrint(a, "{s}/whole.c", .{dir});
            try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = src, .data = try std.mem.concat(a, u8, &.{ compile.runtime, "\n", body, "\n" }) });
            var argv: std.ArrayList([]const u8) = .empty;
            try argv.appendSlice(a, &.{ clang, if (optimize == .O1) "-O1" else "-O2" });
            try argv.appendSlice(a, &flags);
            try argv.appendSlice(a, &.{ "-fstack-usage", "-c", "-x", "c", src, "-o", try std.fmt.allocPrint(a, "{s}/whole.o", .{dir}) });
            try run(a, argv.items);
            const whole = try std.Io.Dir.cwd().readFileAlloc(t.io, try std.fmt.allocPrint(a, "{s}/whole.su", .{dir}), a, .limited(1 << 16));
            var config: compile.Config = .{ .clang = clang, .optimize = optimize };
            config.scratch_root = dir;
            const sub = try std.fmt.allocPrint(a, "{s}/split-{t}", .{ dir, optimize });
            try std.Io.Dir.cwd().createDirPath(t.io, sub);
            const built = (try split.build(a, t.io, sub, body, config, .fromMilliseconds(20_000))).?;
            const want = try frames(a, whole);
            const got = try frames(a, built.su);
            if (want.len != got.len) std.debug.print("{t}: whole {s} | split {s}\n", .{ optimize, whole, built.su });
            try t.expectEqual(want.len, got.len);
            for (want, got) |w, g| try t.expectEqualStrings(w, g);
        }
    }
}

/// vaddr, file bytes and memory size of each loaded segment.
fn segments(a: std.mem.Allocator, b: []const u8) ![]const [3]u64 {
    var out: std.ArrayList([3]u64) = .empty;
    const phoff = std.mem.readInt(u32, b[28..32], .little);
    for (0..std.mem.readInt(u16, b[44..46], .little)) |i| {
        const p = b[phoff + 32 * i ..][0..32];
        if (std.mem.readInt(u32, p[0..4], .little) != 1 or std.mem.readInt(u32, p[20..24], .little) == 0) continue;
        const off = std.mem.readInt(u32, p[4..8], .little);
        const size = std.mem.readInt(u32, p[16..20], .little);
        const h = std.hash.Wyhash.hash(0, b[off..][0..size]);
        try out.append(a, .{ std.mem.readInt(u32, p[8..12], .little), h, std.mem.readInt(u32, p[20..24], .little) });
    }
    return out.items;
}

test "configured in-process link: loads the same segments as ld.lld for a body calling into the runtime" {
    const clang = clangOr() orelse return error.SkipZigTest;
    const lld = std.c.getenv("TF_RUNNER_LLD") orelse return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const body =
        \\static const char name[] = {'s', 'u', 'm', 0};
        \\static volatile int seen[4] = {3, 1, 4, 1};
        \\static int total;
        \\static void out(int c) {
        \\ register int a0 __asm__("a0") = c; register int a7 __asm__("a7") = 1;
        \\ __asm__ volatile("ecall" : "+r"(a0) : "r"(a7));
        \\}
        \\int main(void) {
        \\ char copy[24];
        \\ for (int i = 0; i < 4; ++i) { total += seen[i]; out('0' + total % 10); }
        \\ memset(copy, seen[0], sizeof copy);
        \\ memcpy(copy + 4, name, sizeof name);
        \\ for (const char *p = copy + 4; *p; ++p) out(*p);
        \\ return 0;
        \\}
        \\
    ;
    const objects = [_][2][]const u8{ .{ "rt", split.runtime }, .{ "body", split.declarations ++ body } };
    var paths: [2][]const u8 = undefined;
    var bytes: [2][]const u8 = undefined;
    for (objects, &paths, &bytes) |o, *path, *obj| {
        const src = try std.fmt.allocPrint(a, "{s}/{s}.c", .{ dir, o[0] });
        path.* = try std.fmt.allocPrint(a, "{s}/{s}.o", .{ dir, o[0] });
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = src, .data = o[1] });
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(a, &.{ clang, "-O1" });
        try argv.appendSlice(a, &flags);
        try argv.appendSlice(a, &.{ "-c", "-x", "c", src, "-o", path.* });
        try run(a, argv.items);
        obj.* = try std.Io.Dir.cwd().readFileAlloc(t.io, path.*, a, .limited(1 << 20));
    }
    const script = try std.fmt.allocPrint(a, "{s}/rom.ld", .{dir});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = script, .data = compile.linker_script });
    const elf = try std.fmt.allocPrint(a, "{s}/lld.elf", .{dir});
    try run(a, &.{ std.mem.span(lld), "--no-relax", "--build-id=none", "-T", script, paths[0], paths[1], "-o", elf });
    const ours = (try rvlink.link(a, &bytes)).?;
    try t.expect(compile.validElf(ours));
    const theirs = try std.Io.Dir.cwd().readFileAlloc(t.io, elf, a, .limited(1 << 20));
    try t.expectEqual(std.mem.readInt(u32, theirs[24..28], .little), std.mem.readInt(u32, ours[24..28], .little));
    try t.expectEqualSlices([3]u64, try segments(a, theirs), try segments(a, ours));
    // a relocation kind the compile step never emits makes the link decline (ld.lld then links the program)
    const broken = try a.dupe(u8, bytes[1]);
    try t.expect(corrupt(broken));
    try t.expectEqual(@as(?[]u8, null), try rvlink.link(a, &.{ bytes[0], broken }));
}

/// Rewrites the first relocation's type to R_RISCV_RELAX; false when the object has none.
fn corrupt(b: []u8) bool {
    const shoff = std.mem.readInt(u32, b[32..36], .little);
    for (0..std.mem.readInt(u16, b[48..50], .little)) |i| {
        const sh = b[shoff + 40 * i ..][0..40];
        if (std.mem.readInt(u32, sh[4..8], .little) != 4 or std.mem.readInt(u32, sh[20..24], .little) < 12) continue;
        b[std.mem.readInt(u32, sh[16..20], .little) + 4] = 51;
        return true;
    }
    return false;
}

test "configured fork server: answers as a spawned clang, streams included; a timeout leaves it serving" {
    const clang = clangOr() orelse return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const linker = if (std.c.getenv("TF_RUNNER_LLD")) |l| std.mem.span(l) else null;
    // an inheritable pipe the runner holds: the server must not keep it (it outlives every call)
    var leak: [2]i32 = undefined;
    try t.expectEqual(@as(usize, 0), std.os.linux.pipe2(&leak, .{}));
    defer for (leak) |fd| {
        _ = std.os.linux.close(fd);
    };
    try t.expect(try forkserver.start(a, t.io, clang, linker, dir));
    try t.expect(forkserver.ready(clang));
    const fds = try std.fmt.allocPrint(a, "/proc/{d}/fd", .{forkserver.serverPid(clang).?});
    var held = try std.Io.Dir.cwd().openDir(t.io, fds, .{ .iterate = true });
    defer held.close(t.io);
    var it = held.iterate();
    while (try it.next(t.io)) |e| try t.expect(try std.fmt.parseInt(u32, e.name, 10) <= 3);
    const src = try std.fmt.allocPrint(a, "{s}/x.c", .{dir});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = src, .data = split.declarations ++ "int main(void) { char b[4]; memset(b, 7, sizeof b); return b[1]; }\n" });
    const limits: spawn.Limits = .{ .stdout = 4 << 20, .stderr = 1 << 16, .timeout = .fromMilliseconds(20_000) };
    // -ast-print writes through clang's buffered stdout, which a served child flushes before it answers
    const ast = [_][]const u8{ clang, "--target=riscv32-unknown-elf", "-ffreestanding", "-nostdinc", "-fsyntax-only", "-Xclang", "-ast-print", "-x", "c", src };
    const served = (try forkserver.run(a, t.io, dir, &ast, limits)).?;
    const spawned = (try spawn.run(a, t.io, &ast, limits)).?;
    try t.expect(served.ok and spawned.ok and std.mem.indexOf(u8, served.stdout, "int main(void)") != null);
    try t.expectEqualStrings(spawned.stdout, served.stdout);
    const bad = (try forkserver.run(a, t.io, dir, &.{ clang, "-fno-such-flag-tf", "-c", src }, limits)).?;
    try t.expect(!bad.ok and bad.stderr.len > 0);
    const object = [_][]const u8{ clang, "--target=riscv32-unknown-elf", "-O2", "-ffreestanding", "-nostdinc", "-c", src, "-o", "x.o" };
    // past its deadline a call runs nothing; 2 ms in, the child is killed mid-compile and the server keeps its stream
    for ([_]u64{ 1, 2 * std.time.ns_per_ms }) |ns| {
        const late = try forkserver.run(a, t.io, dir, &object, .{ .stdout = 4096, .stderr = 4096, .timeout = .fromNanoseconds(ns) });
        try t.expectEqual(@as(?spawn.Ran, null), late);
        try t.expect(forkserver.ready(clang));
    }
    const again = (try forkserver.run(a, t.io, dir, &.{ clang, "--version" }, limits)).?;
    try t.expect(again.ok and std.mem.indexOf(u8, again.stdout, "clang version") != null);
}
