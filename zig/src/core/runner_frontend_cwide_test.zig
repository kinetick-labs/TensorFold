const std = @import("std");
const compiler = @import("runner_frontend_program_compile.zig");
const frontend = @import("runner_frontend.zig");
const reference = @import("runner_frontend_reference_test.zig");
const t = std.testing;

pub const observed_fibonacci =
    \\#include <stdio.h>
    \\
    \\unsigned long long fib_iter(unsigned int n) {
    \\    if (n == 0) return 0ULL;
    \\    if (n == 1) return 1ULL;
    \\
    \\    unsigned long long a = 0ULL; // F(0)
    \\    unsigned long long b = 1ULL; // F(1)
    \\    unsigned long long next;
    \\
    \\    for (unsigned int i = 2; i <= n; ++i) {
    \\        next = a + b;
    \\        a = b;
    \\        b = next;
    \\    }
    \\
    \\    return b;
    \\}
    \\
    \\int main(void) {
    \\    unsigned int indices[] = {0, 1, 10, 40};
    \\    size_t count = sizeof(indices) / sizeof(indices[0]);
    \\
    \\    for (size_t i = 0; i < count; ++i) {
    \\        unsigned int n = indices[i];
    \\        printf("F(%u) = %llu\n", n, fib_iter(n));
    \\    }
    \\
    \\    return 0;
    \\}
;

test "wide printf lexical validation accepts only supported lengths and conversions" {
    const good = [_][]const u8{
        "printf(\"%lld %lli %llu\\n\", -1LL, 2LL, 3ULL);",
        "printf(\"%%llu %d %i %u %c %s\", -1, 2, 3u, 'x', \"ok\");",
        "// printf(\"%lln\", 0);\nprintf(\"%llu\", 0ULL);",
        "const char *s = \"%lln\"; printf(\"%s\", s);",
        observed_fibonacci,
    };
    for (good) |source| try t.expect(compiler.safeFormats(source));
    const bad = [_][]const u8{
        "printf(\"%ll\", 1ULL);",
        "printf(\"%l\", 1L);",
        "printf(\"%\");",
        "printf(\"%llu",
        "printf(\"%llllu\", 1ULL);",
        "printf(\"%ld\", 1L);",
        "printf(\"%lu\", 1UL);",
        "printf(\"%llx\", 1ULL);",
        "printf(\"%llf\", 1.0);",
        "printf(\"%lls\", \"text\");",
        "printf(\"%llc\", 'x');",
        "printf(\"%lln\", 0);",
        "printf(\"%ll%\");",
        "printf(\"%020llu\", 1ULL);",
        "printf(\"%+lld\", 1LL);",
        "printf(\"%.*llu\", 2, 1ULL);",
        "printf(\"%1$llu\", 1ULL);",
        "printf(\"%ll\" \"u\", 1ULL);",
        "printf(\"\\x25llu\", 1ULL);",
        "printf(format, 1ULL);",
    };
    for (bad) |source| try t.expect(!compiler.safeFormats(source));
}

test "observed unsigned long long C Fibonacci fence executes exactly" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var context: frontend.Context = .{
        .io = t.io,
        .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") },
    };
    const front = context.frontend();
    const result = try front.translate(front.ctx, t.allocator, .{
        .form = .fence,
        .lang = "c",
        .code = observed_fibonacci,
    });
    try t.expect(result == .program);
    defer t.allocator.free(result.program.bytes);
    try t.expectEqual(.elf, result.program.kind);
    try reference.check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", result.program, "F(0) = 0\nF(1) = 1\nF(10) = 55\nF(40) = 102334155\n");
}

test "wide C decimals preserve extrema signs aliases varargs alignment and printf counts" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    const cases = [_][2][]const u8{
        .{ "printf(\"%llu\", 18446744073709551615ULL);", "18446744073709551615" },
        .{ "printf(\"%lld\", (-9223372036854775807LL - 1LL));", "-9223372036854775808" },
        .{ "printf(\"%lli\", 9223372036854775807LL);", "9223372036854775807" },
        .{ "printf(\"%lld %lli %llu\", -1LL, 0LL, 0ULL);", "-1 0 0" },
        .{ "uint64_t n = 4294967296ULL; int64_t s = -4294967296LL; printf(\"%llu %lli\", n, s);", "4294967296 -4294967296" },
        .{ "printf(\"%d %llu %u %lld %s %c %%\", -7, 18446744073709551615ULL, 9u, -8LL, \"ok\", 'x');", "-7 18446744073709551615 9 -8 ok x %" },
        .{ "int n = printf(\"%llu\", 18446744073709551615ULL); printf(\" %d\", n);", "18446744073709551615 20" },
        .{ "int n = printf(\"%lld\", (-9223372036854775807LL - 1LL)); printf(\" %d\", n);", "-9223372036854775808 20" },
        .{ "unsigned long long a = 0, b = 1; for(int i = 2; i <= 93; ++i) {" ++ " unsigned long long next = a + b; a = b; b = next; } printf(\"%llu\", b);", "12200160415121876738" },
    };
    for ([_]compiler.Config{ .{ .optimize = .O0 }, .{ .optimize = .O1 } }) |profile| {
        for (cases) |case| {
            const source = try std.fmt.allocPrint(t.allocator, "int main(void) {{ {s} return 0; }}", .{case[0]});
            defer t.allocator.free(source);
            const elf = (try compiler.compile(t.allocator, t.io, source, .{
                .clang = clang,
                .linker = env.get("TF_RUNNER_LLD"),
                .optimize = profile.optimize,
                .step_budget = 0,
            })) orelse return error.UnexpectedAbstention;
            defer t.allocator.free(elf);
            try checkWide(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", elf, case[1]);
        }
    }
}

test "C compiler refuses wide printf argument width and count mismatches" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    _ = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    const control = (try compiler.compile(t.allocator, t.io, "int main(void) { printf(\"%llu\", 1ULL); return 0; }", .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
    })) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(control);
    const bad = [_][]const u8{
        "printf(\"%llu\", 1u);",
        "printf(\"%lld\", 1);",
        "printf(\"%lli\", 1L);",
        "printf(\"%llu\", 1.0);",
        "printf(\"%llu\");",
        "printf(\"%d\", 1LL);",
        "printf(\"%lu\", 1ULL);",
        "printf(\"%lln\", 0);",
    };
    for (bad) |body| {
        const source = try std.fmt.allocPrint(t.allocator, "int main(void) {{ {s} return 0; }}", .{body});
        defer t.allocator.free(source);
        const result = try compiler.compile(t.allocator, t.io, source, .{
            .clang = clang,
            .linker = env.get("TF_RUNNER_LLD"),
        });
        defer if (result) |elf| t.allocator.free(elf);
        try t.expect(result == null);
    }
}

fn checkWide(root: []const u8, python: []const u8, elf: []const u8, expected: []const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "wide.elf", .{});
    defer file.close(t.io);
    try file.writeStreamingAll(t.io, elf);
    const path = try tmp.dir.realPathFileAlloc(t.io, "wide.elf", t.allocator);
    defer t.allocator.free(path);
    const script =
        \\import sys
        \\from pathlib import Path
        \\sys.path.insert(0, str(Path(sys.argv[1]) / 'experimental/riscv'))
        \\from chipformer.rv import elf, ref
        \\m = ref.Machine.from_program(elf.program(sys.argv[2]))
        \\m.run(5000000)
        \\assert m.halted and m.fault is None and m.exit_code == 0, (m.halted, m.fault, m.exit_code)
        \\sys.stdout.buffer.write(bytes(m.out))
    ;
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ python, "-c", script, root, path },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(10), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("wide C RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}
