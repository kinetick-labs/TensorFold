//! Offline CLI: runner-graph-pack INPUT.tfimr OUTPUT.tfgp; runtime never uses this compiler path.
const std = @import("std");
const model = @import("steps_v1/runner_circuit_model.zig");
const binary = @import("runner_graph_binary.zig");
const loader = @import("steps_v1/runner_circuit_load.zig");
const Identity = struct { experimental: struct { kind: []const u8, contract: struct { op: []const u8, width: u16 } } };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedInputAndOutputPaths;
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.gpa, .limited(1 << 30));
    defer init.gpa.free(source);
    const payload, _ = try loader.payload(source);
    const identity = try std.json.parseFromSlice(Identity, init.gpa, payload, .{ .ignore_unknown_fields = true });
    defer identity.deinit();
    const contract = identity.value.experimental.contract;
    if (!std.mem.eql(u8, identity.value.experimental.kind, "graph_template") or
        !@import("runner_graph_packet.zig").widthValid(contract.width) or contract.width == 255) return error.ExpectedGraphTemplate;
    var circuit = try loader.fromBytes(init.gpa, source);
    defer circuit.deinit(init.gpa);
    try requireGraph(&circuit, contract.width);
    circuit.program.image.rom = &.{};
    circuit.program.image.drom = &.{};
    circuit.program.image.ram = &.{};
    circuit.program.ram_tables = @splat(std.math.maxInt(u32));
    circuit.program.input_address = 0;
    circuit.program.input_capacity = 0;
    circuit.program.kind = .elf;
    const bytes = try binary.encode(init.gpa, circuit, contract.op, contract.width);
    defer init.gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = bytes });
}

fn requireGraph(c: *const model.Circuit, width: u16) !void {
    var input = false;
    var output = false;
    for (c.table_names) |name| {
        if (std.mem.startsWith(u8, name, "output.cursor")) continue;
        if (std.mem.eql(u8, name, "in0.0")) input = true;
        if (std.mem.eql(u8, name, "out0.0")) output = true;
        const prefix: usize = if (std.mem.startsWith(u8, name, "in")) 2 else if (std.mem.startsWith(u8, name, "out")) 3 else continue;
        const dot = std.mem.indexOfScalarPos(u8, name, prefix, '.') orelse return error.ExpectedGraphTemplate;
        const logical = std.fmt.parseInt(u32, name[prefix..dot], 10) catch return error.ExpectedGraphTemplate;
        const offset = std.fmt.parseInt(u32, name[dot + 1 ..], 10) catch return error.ExpectedGraphTemplate;
        if (logical >= 16 or offset % 31 != 0 or offset >= width) return error.ExpectedGraphTemplate;
    }
    if (!input or !output) return error.ExpectedGraphTemplate;
}
