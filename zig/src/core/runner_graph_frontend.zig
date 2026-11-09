//! A graph-only frontend that binds exact integer cells without invoking the Python/RV32 frontend.
const std = @import("std");
const api = @import("runner_api.zig");
const Bank = @import("runner_graph_bank.zig").Bank;
const bind = @import("runner_graph_bind.zig");

pub fn frontend(bank: *const Bank) api.Frontend {
    return .{ .ctx = @constCast(bank), .translate = translate };
}

fn translate(ctx: ?*anyopaque, a: std.mem.Allocator, cell: api.Cell) std.mem.Allocator.Error!api.Translation {
    const bank: *const Bank = @ptrCast(@alignCast(ctx.?));
    const bytes = try bind.cell(a, bank, cell) orelse return .{ .abstain = .unsupported };
    return .{ .program = .{ .kind = .graph, .bytes = bytes } };
}
