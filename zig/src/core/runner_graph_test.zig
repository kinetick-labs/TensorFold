//! The graph bank's host tests: packets, programs, tables, dispatch and the frozen v1 step model the bank hashes.
test {
    _ = @import("runner_graph.zig");
    _ = @import("runner_graph_binary.zig");
    _ = @import("runner_graph_dispatch.zig");
    _ = @import("runner_graph_edit.zig");
    _ = @import("runner_graph_packet.zig");
    _ = @import("runner_graph_program.zig");
    _ = @import("runner_graph_route.zig");
    _ = @import("runner_graph_table_test.zig");
    _ = @import("steps_v1/runner_circuit_model.zig");
}
