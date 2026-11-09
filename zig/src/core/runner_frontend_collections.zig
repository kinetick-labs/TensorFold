const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const parser = @import("runner_frontend_collections_parse.zig");
const proof = @import("runner_frontend_collections_proof.zig");
const emitter = @import("runner_frontend_collections_emit.zig");
pub const Lowered = struct { source: []u8, estimated_steps: usize };

pub fn lowerWithCost(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!?Lowered {
    if (source.len == 0 or source.len > 4096 or std.mem.indexOfScalar(u8, source, 0) != null) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var p: ast.Program = .{ .a = arena.allocator() };
    parser.parse(&p, source) catch |err| return switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
    proof.prove(&p) catch |err| return switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
    return .{ .source = try emitter.emit(a, &p), .estimated_steps = p.stats.steps };
}
