//! The in-model runner's test files (core/runner.zig refers to this one).
test {
    _ = @import("runner_scan.zig");
    _ = @import("runner_api.zig");
    _ = @import("runner_trace_test.zig");
    _ = @import("runner_settle_test.zig");
    _ = @import("runner_sync_test.zig");
    _ = @import("runner_probe.zig");
    _ = @import("runner_graph_table_test.zig");
    _ = @import("runner_inline_mod_test.zig");
    _ = @import("runner_inline_arith_test.zig");
    _ = @import("runner_inline_chain_test.zig");
    _ = @import("runner_receipt_test.zig");
    _ = @import("runner_inline_context_test.zig");
    _ = @import("runner_inline_hex_test.zig");
    _ = @import("runner_trace_stated_test.zig");
    _ = @import("runner_trace_divmod_test.zig");
    _ = @import("runner_trace_named_test.zig");
    _ = @import("runner_trace_check_test.zig");
    _ = @import("runner_graph_edit.zig");
    _ = @import("runner_commit.zig");
    _ = @import("runner_digit_stop_test.zig");
    _ = @import("runner_inline_quoted_test.zig");
}
