//! Request-owned states retain allocations between template invocations over immutable circuit weights.
const std = @import("std");
const State = @import("steps_v1/runner_circuit_state.zig").State;
const Template = @import("runner_graph_bank.zig").Template;

pub const Cache = struct {
    entries: std.ArrayList(Entry) = .empty,

    const Entry = struct { template: *const Template, state: State };

    pub fn deinit(c: *Cache, a: std.mem.Allocator) void {
        for (c.entries.items) |*e| e.state.deinit();
        c.entries.deinit(a);
        c.* = .{};
    }

    /// The returned pointer stays valid until the next get; callers release it with reset on every exit.
    pub fn get(c: *Cache, a: std.mem.Allocator, t: *const Template) !*State {
        try t.prepare();
        for (c.entries.items) |*e| if (e.template == t) return &e.state;
        try c.entries.ensureUnusedCapacity(a, 1);
        const state = try State.init(a, &t.circuit);
        c.entries.appendAssumeCapacity(.{ .template = t, .state = state });
        return &c.entries.items[c.entries.items.len - 1].state;
    }
};

/// Restores all persistent steps-v1 state, including host-owned copies of originally shared tables.
pub fn reset(s: *State) void {
    for (s.circuit.tables, s.tables, s.owned, s.present) |table, rows, owned, *present| {
        if (owned) @memcpy(rows, table.rows);
        present.* = table.present;
    }
    @memset(s.history, 0);
    @memset(s.x, 0);
    @memset(s.seen, false);
    s.active.clearRetainingCapacity();
    s.steps = 0;
    @memset(s.work, 0);
}
