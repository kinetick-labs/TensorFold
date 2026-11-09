//! Integer `def`s inlined at their calls, `assert <int cmp>` proved at lowering: S3's mbpp-80/86 on the graph route.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const table = @import("runner_graph_table.zig");
const integer_python = @import("runner_frontend_integer_python.zig");

const gpa = std.testing.allocator;

/// The two recorded arm-d lane calls (imr-measure-20261101T155322Z), then shapes around them; CPython 3's stdout.
const cases = [_][2][]const u8{
    .{ "def tetrahedral_number(n):\n    return n * (n + 1) * (n + 2) // 6\n\nassert tetrahedral_number(5) == 35\nprint(tetrahedral_number(5))", "35\n" },
    .{ "def centered_hexagonal_number(n):\n    return 3 * n * (n - 1) + 1\n\nassert centered_hexagonal_number(10) == 271\nprint(centered_hexagonal_number(10))", "271\n" },
    .{ "def f(n): return n*n - 1\nprint(f(12345678901234567890))\n", "152415787532388367501905199875019052099\n" },
    .{ "def area(w: int, h: int) -> int:\n    \"\"\"Rectangle area.\"\"\"\n    return w * h\n\nprint(area(7, 6), area(-3, 4))\n", "42 -12\n" },
    .{ "def sq(x):\n    return x*x\ndef hyp2(a, b):\n    return sq(a) + sq(b)\nassert hyp2(3, 4) == 25\nprint(hyp2(3, 4))\n", "25\n" },
    // globals resolve at the call, as Python reads them: k is 10 when f runs
    .{ "def f(n):\n    return n + k\nk = 10\nprint(f(5))\nk = 20\nprint(f(5))\n", "15\n25\n" },
    // a parameter shadows the module's name of the same spelling
    .{ "n = 100\ndef f(n):\n    return n // 7\nprint(f(50), n)\n", "7 100\n" },
    .{ "def even(n):\n    return n % 2 == 0\nprint(even(10), even(7))\n", "True False\n" },
    .{ "def f(a, b):\n    return a ** b % 1000000007\nassert f(3, 200) != 0\nprint(f(3, 200))\n", "136318165\n" },
    .{ "def c(n):\n    return 3*n*(n-1)+1 # hexagon\n\n# blank and comment lines in the suite pass\nprint(c(10))\n", "271\n" },
};

/// Scripts Python raises on, or runs beyond this lowering: each refuses, and the sandbox runs it as before.
const refused = [_][]const u8{
    // recursion, direct and mutual
    "def f(n):\n    return f(n - 1)\nprint(f(3))\n",
    "def f(n):\n    return g(n)\ndef g(n):\n    return f(n)\nprint(f(3))\n",
    // side effects in the body (multi-statement suites: runner_frontend_integer_loop_test.zig)
    "def f(n):\n    print(n)\n    return n\nprint(f(3))\n",
    "def f(n):\n    global k\n    return n\nprint(f(3))\n",
    "def f(n): return n; k = 1\nprint(f(3))\n",
    "def f(n=2):\n    return n\nprint(f())\n",
    "def f(*n):\n    return 1\nprint(f(3))\n",
    "def f(n): return n\n    return 2\nprint(f(3))\n",
    // non-int returns
    "def f(n):\n    return n / 2\nprint(f(3))\n",
    "def f(n):\n    return \"x\"\nprint(f(3))\n",
    "def f(n):\n    return [n]\nprint(f(3))\n",
    "def f(n):\n    return n, n\nprint(f(3))\n",
    "def f(n):\n    return\nprint(f(3))\n",
    "def f(n: str):\n    return 1\nprint(f(3))\n",
    "def f(n) -> float:\n    return 1\nprint(f(3))\n",
    // a false assert raises AssertionError in Python; a message or a non-comparison is not taken
    "def f(n):\n    return n * 2\nassert f(3) == 7\nprint(f(3))\n",
    "print(1)\nassert 2 + 2 == 5\n",
    "assert 1 == 1, 'msg'\nprint(1)\n",
    "assert 5\nprint(1)\n",
    // names Python would not find: a caller's parameter, a global bound after the call, a call before the def
    "def f():\n    return x\ndef g(x):\n    return f()\nprint(g(3))\n",
    "def f(n):\n    return n + k\nprint(f(5))\nk = 1\n",
    "print(f(3))\ndef f(n):\n    return n\n",
    // arity, a name reused as a value or redefined, a builtin's name, a call with a keyword
    "def f(n):\n    return n\nprint(f(1, 2))\n",
    "def f(n):\n    return n\nf = 3\nprint(f)\n",
    "def f(n):\n    return n\nprint(f)\n",
    "def f(n):\n    return n\ndef f(n):\n    return n + 1\nprint(f(1))\n",
    "def print(n):\n    return n\nprint(1)\n",
    "def f(n):\n    return n\nprint(f(n=1))\n",
    // a bare expression after a def is REPL-shaped: the script prints nothing in Python
    "def f(n):\n    return n\nf(3)\n",
    // division by zero inside the body raises in Python
    "def f(n):\n    return 1 // n\nprint(f(0))\n",
};

/// What the table's proved values print (ints and bools; the cases have no tuples or formats).
fn proved(a: std.mem.Allocator, code: []const u8) !?[]u8 {
    const lowered = try integer_python.lowerWithCost(a, code, 0) orelse return null;
    defer a.free(lowered.source);
    const t = lowered.table.?;
    var out: std.ArrayList(u8) = .empty;
    for (t.roots[0..t.outputs], t.endings[0..t.outputs]) |root, end| {
        const n = t.nodes[root];
        switch (n.op) {
            .eq, .ne, .lt, .le, .gt, .ge => try out.appendSlice(a, if (n.value != 0) "True" else "False"),
            else => try out.print(a, "{d}", .{n.value}),
        }
        try out.append(a, end);
    }
    return try out.toOwnedSlice(a);
}

test "runner frontend integer def: inlined defs and proved asserts give CPython's values; other shapes refuse" {
    for (cases) |c| {
        const got = try proved(gpa, c[0]) orelse {
            std.debug.print("refused: {s}\n", .{c[0]});
            return error.TestExpectedLowering;
        };
        defer gpa.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    for (refused) |code| {
        const got = try integer_python.lowerWithCost(gpa, code, 0);
        if (got) |g| {
            gpa.free(g.source);
            std.debug.print("lowered, must refuse: {s}\n", .{code});
            return error.TestExpectedRefusal;
        }
    }
}

test "runner frontend integer def: an assert's nodes never print; the def adds no output of its own" {
    const lowered = (try integer_python.lowerWithCost(gpa, cases[0][0], 0)).?;
    defer gpa.free(lowered.source);
    const t = lowered.table.?;
    try std.testing.expectEqual(@as(usize, 1), t.outputs);
    try std.testing.expectEqual(@as(i512, 35), t.nodes[t.roots[0]].value);
}

/// A backend no test here may start (every program is a graph one).
const Never = struct {
    fn backend() api.Backend {
        return .{ .ctx = undefined, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(_: *anyopaque, _: u64, _: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
        return error.Busy;
    }
    fn poll(_: *anyopaque, _: api.JobId) api.Poll {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
};

test "runner frontend integer def: the scripts run on the real bank and print CPython's stdout (TF_RUNNER_GRAPH_BANK)" {
    const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(gpa, std.mem.span(g));
    defer bank.deinit();
    const backend = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(backend);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = backend };
    defer d.deinit();
    const b = d.backend();
    for (cases, 0..) |c, i| {
        const bytes = (try table.translate(gpa, &bank, c[0])) orelse {
            std.debug.print("not on the bank: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        const job = try b.start(i + 1, .{ .kind = .graph, .bytes = bytes }, 50_000_000);
        const r = while (true) switch (b.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |x| break x,
        };
        defer b.release(i + 1);
        if (r.status != .ok or !std.mem.eql(u8, r.out, c[1])) std.debug.print("{s}: {t} '{s}'\n", .{ c[0], r.status, r.out });
        try std.testing.expectEqual(api.Status.ok, r.status);
        try std.testing.expectEqualStrings(c[1], r.out);
    }
    for (refused) |code| try std.testing.expectEqual(null, try table.translate(gpa, &bank, code));
}

/// T5's merge gate on the list tip (T5-S3-MERGE.md); the mutated-packet check is runner_frontend_integer_list_test.zig.
const lists = [_][2][]const u8{
    .{ "print(min([2, 3]))\n", "2\n" },
    .{ "print(min([4, 3]))\n", "3\n" },
    .{ "print(sorted([4, 3]))\n", "[3, 4]\n" },
    .{ "print(len([x for x in [7] if x > 0]))\n", "1\n" },
    .{ "print(sum(range(3, 1)))\n", "0\n" },
};

test "runner frontend integer def: the served graph route refuses each list program or prints CPython's stdout" {
    const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(gpa, std.mem.span(g));
    defer bank.deinit();
    var context: @import("runner_frontend.zig").Context = .{ .io = std.testing.io, .compiler = .{ .clang = "/nonexistent-list-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    const backend = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(backend);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = backend };
    defer d.deinit();
    const b = d.backend();
    for (lists, 0..) |c, i| {
        const result = try f.translate(f.ctx, gpa, .{ .form = .fence, .lang = "python", .code = c[0] });
        defer if (result == .program) gpa.free(result.program.bytes);
        if (result != .program or result.program.kind != .graph) continue;
        const job = try b.start(i + 1, result.program, 50_000_000);
        const r = while (true) switch (b.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |x| break x,
        };
        defer b.release(i + 1);
        if (r.status == .ok) try std.testing.expectEqualStrings(c[1], r.out);
    }
}
