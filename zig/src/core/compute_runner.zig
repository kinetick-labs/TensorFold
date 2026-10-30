//! The compute lane's Python prelude and the canonical, deterministic result text of one job.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Outcome = @import("compute_lane.zig").Outcome;

/// Passed with -c: runs stdin's source as `<code>`, printing the repr of a final bare expression (REPL convention).
pub const runner: [:0]const u8 =
    \\import sys, ast, io, linecache, random, traceback
    \\random.seed(0)
    \\src = sys.stdin.read()
    \\sys.stdin = io.StringIO()
    \\linecache.cache["<code>"] = (len(src), None, src.splitlines(True), "<code>")
    \\try:
    \\    tree = ast.parse(src, "<code>")
    \\    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    \\    ns = {"__name__": "__main__"}
    \\    exec(compile(tree, "<code>", "exec"), ns)
    \\    if last is not None:
    \\        value = eval(compile(ast.Expression(last.value), "<code>", "eval"), ns)
    \\        if value is not None:
    \\            print(repr(value))
    \\except SystemExit:
    \\    raise
    \\except BaseException as e:
    \\    tb = None if isinstance(e, SyntaxError) else e.__traceback__.tb_next
    \\    traceback.print_exception(type(e), e, tb)
    \\    sys.stdout.flush()
    \\    sys.stderr.flush()
    \\    sys.exit(1)
    \\sys.stdout.flush()
    \\sys.stderr.flush()
;

/// The result text: stdout, then stderr, then a status line for a non-ok job; never a timing.
pub fn body(a: Allocator, o: Outcome, max_ms: u32, max_output_bytes: u32) Allocator.Error![]u8 {
    var t: std.ArrayList(u8) = .empty;
    try t.appendSlice(a, o.stdout);
    if (o.stderr.len > 0) {
        try t.appendSlice(a, if (t.items.len == 0) "[stderr]\n" else "\n[stderr]\n");
        try t.appendSlice(a, o.stderr);
    }
    if (o.status != .ok) {
        if (t.items.len > 0 and t.items[t.items.len - 1] != '\n') try t.append(a, '\n');
        switch (o.status) {
            .ok => unreachable,
            .exit_error => if (o.signal) |s|
                try t.print(a, "[killed by signal {d}]", .{s})
            else
                try t.print(a, "[exit status {d}]", .{o.exit_code orelse 0}),
            .timeout => try t.print(a, "[timed out after {d} ms]", .{max_ms}),
            .output_limit => try t.print(a, "[output truncated at {d} bytes]", .{max_output_bytes}),
            .cancelled => try t.appendSlice(a, "[cancelled]"),
            .unavailable => try t.appendSlice(a, "[sandbox unavailable]"),
        }
    }
    if (t.items.len > 0 and t.items[t.items.len - 1] == '\n') t.items.len -= 1;
    if (t.items.len == 0) try t.appendSlice(a, "[no output]");
    return t.toOwnedSlice(a);
}
