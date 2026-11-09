//! The compiled-image cache through compile() and the frontend: opaque Python cells repeat from it, nothing else does.
const std = @import("std");
const compiler = @import("runner_frontend_program_compile.zig");
const elf_cache = @import("runner_frontend_elf_cache.zig");
const f = @import("runner_frontend.zig");
const t = std.testing;

test {
    _ = @import("runner_frontend_spawn.zig");
    _ = @import("runner_frontend_compile_time_test.zig");
    _ = @import("runner_frontend_split_test.zig");
    _ = @import("runner_frontend_rvlink.zig");
    _ = elf_cache;
}

const body = "int main(void) {\n return 0;\n}\n";

test "compile answers a cached opaque-input source with no compiler run; other settings and trusted C miss" {
    elf_cache.reset(t.io);
    defer elf_cache.reset(t.io);
    const config: compiler.Config = .{ .clang = "/nonexistent-elf-cache-clang", .opaque_inputs = true, .estimated_steps = 40 };
    elf_cache.put(t.io, (try compiler.cacheKey(t.allocator, body, config)).?, "cached image");
    const hit = (try compiler.compile(t.allocator, t.io, body, config)).?;
    defer t.allocator.free(hit);
    try t.expectEqualStrings("cached image", hit);
    try t.expectEqual(@as(u64, 1), elf_cache.hitCount(t.io));
    // each of these reaches the missing compiler and abstains: none may take the cached image
    var plain = config;
    plain.opaque_inputs = false;
    var over = config;
    over.estimated_steps = config.step_budget + 1;
    var unknown = config;
    unknown.estimated_steps = null;
    var o2 = config;
    o2.optimize = .O2;
    var stack = config;
    stack.stack_limit = 4096;
    var other_clang = config;
    other_clang.clang = "/nonexistent-elf-cache-clang-2";
    for ([_]compiler.Config{ plain, over, unknown, o2, stack, other_clang }) |c| {
        try t.expectEqual(@as(?[]u8, null), try compiler.compile(t.allocator, t.io, body, c));
    }
    try t.expectEqual(@as(?[]u8, null), try compiler.compile(t.allocator, t.io, "int main(void) {\n return 1;\n}\n", config));
    try t.expectEqual(@as(u64, 1), elf_cache.hitCount(t.io));
}

test "configured RV32: a repeated Python cell is compiled once and returns the same image; a C cell compiles each time" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    elf_cache.reset(t.io);
    defer elf_cache.reset(t.io);
    var context: f.Context = .{ .io = t.io, .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") } };
    const front = context.frontend();
    const code = "def f(n):\n    s = 0\n    for i in range(n):\n        s += i * i\n    return s\nprint(f(12))\n";
    var images: [2][]const u8 = undefined;
    for (&images, 0..) |*image, i| {
        const r = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = code });
        try t.expect(r == .program);
        try t.expectEqual(.elf, r.program.kind);
        image.* = r.program.bytes;
        try t.expectEqual(@as(u64, i), elf_cache.hitCount(t.io));
    }
    defer for (images) |image| t.allocator.free(image);
    try t.expectEqualSlices(u8, images[0], images[1]);
    for (0..2) |_| {
        const r = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "c", .code = "int main(void) { return 0; }\n" });
        try t.expect(r == .program);
        t.allocator.free(r.program.bytes);
    }
    try t.expectEqual(@as(u64, 1), elf_cache.hitCount(t.io));
}
