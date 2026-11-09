//! CPU steps/s of the circuit evaluator: boots a calc expression (or an .elf file) and runs it to HALT, repeated.
const std = @import("std");
const rc = @import("runner_circuit.zig");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: runner-circuit-bench <circuit> <expression | image.elf> [repeats]\n", .{});
        return 2;
    }
    const repeats = if (argv.len > 3) try std.fmt.parseInt(u32, argv[3], 10) else 3;
    const t_load = rc.nowNs();
    var c = try rc.Circuit.load(gpa, argv[1]);
    defer c.deinit(gpa);
    std.debug.print("loaded {d} dims, {d} nodes, {d} batches, {d} tables, {d} nonzeros in {d} ms\n", .{ c.dims, c.nodes, c.batches.len, c.tables.len, c.nonzeros(), (rc.nowNs() - t_load) / std.time.ns_per_ms });
    var cancel: std.atomic.Value(bool) = .init(false);
    const is_elf = std.mem.endsWith(u8, argv[2], ".elf");
    const bytes = if (is_elf) try std.Io.Dir.cwd().readFileAlloc(init.io, argv[2], init.arena.allocator(), .limited(1 << 26)) else argv[2];
    for (0..repeats) |_| {
        const t0 = rc.nowNs();
        var m = try rc.Machine.boot(gpa, &c, .{ .kind = if (is_elf) .elf else .calc, .bytes = bytes });
        defer m.deinit(gpa);
        const t1 = rc.nowNs();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        const end = try m.run(10_000_000, &out, null, &cancel);
        const ns = rc.nowNs() - t1;
        std.debug.print("{s}: {d} steps, boot {d} us, run {d} ms, {d} steps/s, out {s}", .{ @tagName(end.end), end.steps, (t1 - t0) / 1000, ns / std.time.ns_per_ms, end.steps * std.time.ns_per_s / @max(ns, 1), out.items });
    }
    return 0;
}
