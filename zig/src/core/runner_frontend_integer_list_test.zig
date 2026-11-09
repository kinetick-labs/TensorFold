//! Static int lists in the integer lowering (S3): mbpp-66 and shapes around it on the graph route, CPython's stdout.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const table = @import("runner_graph_table.zig");
const integer_python = @import("runner_frontend_integer_python.zig");
const integer = @import("runner_frontend_integer.zig");

const gpa = std.testing.allocator;

/// mbpp-66 as recorded at arm-d (imr-measure-20261101T155322Z), then variations; expected stdout from CPython 3.
const cases = [_][2][]const u8{
    .{ "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nassert pos_count([1, -2, 3, -4]) == 2\nprint(pos_count([1, -2, 3, -4]))", "2\n" },
    .{ "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nprint(pos_count([0, 5, -7, 8, 9]))\n", "3\n" },
    .{ "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nprint(pos_count(list(range(-3, 4))))\n", "3\n" },
    .{ "xs = [3, 1, 2]\nprint(len(xs), sum(xs), min(xs), max(xs))\n", "3 6 1 3\n" },
    .{ "print(sorted([3, 1, 2]))\n", "[1, 2, 3]\n" },
    .{ "print(sorted([3, 1, 2], reverse=True))\n", "[3, 2, 1]\n" },
    .{ "print(sorted([2, -1, 2, 0]))\n", "[-1, 0, 2, 2]\n" },
    .{ "print(sorted([4, 1, 3, 2], reverse=True))\n", "[4, 3, 2, 1]\n" },
    .{ "print(list(range(5)))\nprint(list(range(2, 11, 3)))\nprint(list(range(10, 0, -3)))\n", "[0, 1, 2, 3, 4]\n[2, 5, 8]\n[10, 7, 4, 1]\n" },
    .{ "n = 6\nprint(sum(i * i for i in range(1, n) if i % 2 == 1))\n", "35\n" },
    .{ "n = 6\nprint(len([i for i in range(1, n) if i % 2 == 1]))\n", "3\n" },
    .{ "a = [1, 2, 3]\nb = [10, 20, 30]\nprint([x + y for x, y in zip(a, b)])\n", "[11, 22, 33]\n" },
    .{ "print(list(x * y for x, y in zip([1, 2, 3], [4, 5])))\n", "[4, 10]\n" },
    .{ "xs = [5, 7, 9]\nprint([i * x for i, x in enumerate(xs)], sum(i for i, x in enumerate(xs, 1) if x > 6))\n", "[0, 7, 18] 5\n" },
    .{ "print(sum(x * x for x in range(10)))\n", "285\n" },
    .{ "xs = [4, -1, 7]\nprint(xs, xs[0], xs[-1])\n", "[4, -1, 7] 4 7\n" },
    .{ "print([x > 2 for x in [1, 2, 3, 4]])\n", "[False, False, True, True]\n" },
    .{ "print(max(3, 9, 4), min(3, -9, 4), max([1, 5, 5, 2]))\n", "9 -9 5\n" },
    .{ "print(sum([1 > 0]), sum([2 > 1, 3 > 1]))\n", "1 2\n" },
    .{ "def avg_floor(xs):\n    return sum(xs) // len(xs)\nprint(avg_floor([10, 20, 31]))\n", "20\n" },
    .{ "x = 100\nys = [x + 1 for x in range(3)]\nprint(ys, x)\n", "[1, 2, 3] 100\n" },
    .{ "a = [1, 2]\nb = a\nprint(b, a)\n", "[1, 2] [1, 2]\n" },
    .{ "print([2 ** 100, -(3 ** 40)])\n", "[1267650600228229401496703205376, -12157665459056928801]\n" },
    .{ "print(sum(x for x in range(5) if x > 10), len([]), sum([]))\n", "0 0 0\n" },
    .{ "xs = [1, 2]\nprint(xs)\nprint(len(xs))\n", "[1, 2]\n2\n" },
    // T5's ownership programs (S3 merge gate), then filters whose masks are int truths
    .{ "print(min([2, 3]), len([x for x in [7] if x > 0]))\n", "2 1\n" },
    .{ "print(sorted([2, 3]))\n", "[2, 3]\n" },
    .{ "print(sum(x for x in [2, 5] if x > 3), max(3, 2))\n", "5 3\n" },
    .{ "print(min(x * x for x in [2, 3]))\n", "4\n" },
    .{ "print(len([x for x in [7, -2, 9] if x % 3]))\n", "2\n" },
    .{ "print(list(range(12)))\n", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]\n" },
    // an empty range with a start (T5: `sum(range(3,1))` panicked in ReleaseSafe)
    .{ "print(sum(range(3, 1)), len(range(4, 2)), sum(range(2, 2, 3)))\n", "0 0 0\n" },
    .{ "k = 3\nxs = [k, k * 2]\nk = 10\nprint(xs, k)\n", "[3, 6] 10\n" },
    .{ "print([1, 2, 3,][1], len(range(7, 1, -2)), range(10)[3])\n", "2 3 3\n" },
    .{ "xs = sorted([5, 3, 8])\nprint(xs[0] + xs[-1])\n", "11\n" },
};

/// Scripts Python raises on, or runs beyond this lowering or its budget: each refuses, and the sandbox runs it.
const refused = [_][]const u8{
    // mutation through an alias, or of a list at all
    "a = [1, 2]\nb = a\nb.append(3)\nprint(a)\n",
    "a = [1, 2]\nb = a\nb[0] = 9\nprint(a)\n",
    "a = [1, 2]\na += [3]\nprint(a)\n",
    "a = [3, 1]\na.sort()\nprint(a)\n",
    "a = [1, 2]\ndel a[0]\nprint(a)\n",
    // nested lists
    "print([[1, 2], [3]])\n",
    "xs = [[1], [2]]\nprint(len(xs))\n",
    "print([[1, 2]][0])\n",
    "print(sum([x] for x in [1]))\n",
    // non-int elements
    "print([1, 'a'])\n",
    "print([1.5, 2])\n",
    "print([None])\n",
    "print([divmod(7, 2)])\n",
    "xs = [1, 2]\nprint(xs + [3])\n",
    "xs = [1]\nprint(xs * 2)\n",
    // unbounded or oversized range, a zero step
    "print(sum(range(10 ** 9)))\n",
    "print(len(range(100)))\n",
    "print(list(range(1, 5, 0)))\n",
    // sorted with key=, min/max with key= or default=
    "print(sorted([3, 1], key=abs))\n",
    "print(sorted([3, 1], key=lambda x: -x))\n",
    "print(max([3, 1], key=abs))\n",
    "print(min([], default=0))\n",
    // Python raises
    "print(min([]))\n",
    "print(max(x for x in [1] if x > 5))\n",
    "print([1, 2][2])\n",
    "xs = [1, 1 // 0]\nprint(2)\n",
    "print(len(x for x in [1]))\n",
    "ys = [x for x in [1]]\nprint(x)\n",
    // print of what has no list value here
    "print([])\n",
    "print(range(3))\n",
    "xs = range(4)\nprint(xs)\n",
    // beyond this lowering
    "print([x * y for x in [1, 2] for y in [3]])\n",
    "print(sum([1], 10))\n",
    "print(*[1, 2])\n",
    "print([x if x > 0 else 0 for x in [1, -1]])\n",
    // a filter's kept elements or a circuit choice as a shape: which elements, how many, which one
    "print([x for x in [1, 2] if x > 1])\n",
    "print([x for x in range(20) if x % 2 == 0 if x % 3 == 0])\n",
    "xs = [x for x in [1, 2] if x > 1]\nprint(xs[0])\n",
    "print(max(x for x in [1, 2] if x > 0))\n",
    "print(sorted(x for x in [3, 1] if x))\n",
    "print([y for y in [x for x in [1, 2] if x > 1]])\n",
    "print(list(range(min(2, 3))))\n",
    "print([1, 2, 3][min(0, 1)])\n",
    "print(sum(range(len([x for x in [1] if x]))))\n",
    // bools min/max/sorted would return as bools
    "print(min(True, 2))\n",
    "print(sorted([1 > 0, 0]))\n",
    // a sorting network past the node budget
    "print(sorted(x % 5 for x in range(12, 3, -2)))\n",
    "sorted = 3\nprint(sorted)\n",
};

/// What the table's proved values print, lists in Python's repr (ints and bools; no tuples or formats here).
fn proved(a: std.mem.Allocator, code: []const u8) !?[]u8 {
    const lowered = try integer_python.lowerWithCost(a, code, 0) orelse return null;
    defer a.free(lowered.source);
    const t = lowered.table.?;
    var out: std.ArrayList(u8) = .empty;
    for (t.roots[0..t.outputs], t.endings[0..t.outputs], t.brackets[0..t.outputs]) |root, end, bracket| {
        if (bracket & 1 != 0) try out.append(a, '[');
        const n = t.nodes[root];
        switch (n.op) {
            .eq, .ne, .lt, .le, .gt, .ge => try out.appendSlice(a, if (n.value != 0) "True" else "False"),
            else => try out.print(a, "{d}", .{n.value}),
        }
        if (bracket & 2 != 0) try out.append(a, ']');
        try out.appendSlice(a, if (end == ',') ", " else &.{end});
    }
    return try out.toOwnedSlice(a);
}

test "runner frontend integer list: static int lists give CPython's values; other shapes refuse" {
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

test "runner frontend integer list: lists are the graph route's only; calc and C lowerings refuse them as before" {
    const printed = "print(sorted([3, 1, 2]))\n";
    const lowered = (try integer_python.lowerWithCost(gpa, printed, 0)).?;
    defer gpa.free(lowered.source);
    try std.testing.expect(lowered.graph_only);
    try std.testing.expectEqual(@as(?[]u8, null), try integer_python.lower(gpa, printed));
    try std.testing.expect(try integer_python.lowerWithCost(gpa, printed, 1 << 30) == null);
    for ([_][]const u8{ "xs = [3, 1, 2]\nprint(sum(xs), max(xs))\n", "print(sum(i*i for i in range(10)))\n", "print(max(3, 4))\n" }) |code| {
        try std.testing.expectEqual(@as(?[]u8, null), try integer_python.lower(gpa, code));
        try std.testing.expectEqual(@as(?[]u8, null), try integer_python.lowerCalc(gpa, code));
    }
    // the inline expression lowering (and Decimal, Fraction over the same parser) takes no list grammar
    for ([_][]const u8{ "sum([1, 2])", "max(3, 4)", "len([1])", "[5, 6][0]" }) |code|
        try std.testing.expectEqual(@as(?[]u8, null), try integer.lower(gpa, code));
    // the integer lowering without lists is unchanged
    const source = (try integer_python.lower(gpa, "x = 3\nprint(x * 4, len(str(x)))\n")).?;
    defer gpa.free(source);
    try std.testing.expect(std.mem.indexOf(u8, source, "tf_run_prints(") != null);
}

test "runner frontend integer list: a filter's masks stay circuit nodes; an assert's nodes are dropped after its proof" {
    const lowered = (try integer_python.lowerWithCost(gpa, cases[0][0], 0)).?;
    defer gpa.free(lowered.source);
    const t = lowered.table.?;
    // four `number > 0` masks, each times its 1, summed from 0; the assert's `== 2` is gone
    var masks: usize = 0;
    for (t.nodes[0..t.count]) |n| {
        try std.testing.expect(n.op != .eq);
        masks += @intFromBool(n.op == .gt);
    }
    try std.testing.expectEqual(@as(usize, 4), masks);
    try std.testing.expectEqual(@as(i512, 2), t.nodes[t.roots[0]].value);
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

test "runner frontend integer list: the scripts run on the real bank and print CPython's stdout (TF_RUNNER_GRAPH_BANK)" {
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

/// T5's ownership check: input `from` read as `to` in the same packets prints CPython's stdout for the mutated script.
const Mutation = struct { code: []const u8, from: []const u8, to: []const u8, after: []const u8 };
const D = "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\n";
const mutations = [_]Mutation{
    // the control, then T5's three failures (min, sorted, a filter's len), then the rest of the circuit choices
    .{ .code = "print(2*3)\n", .from = "2", .to = "4", .after = "12\n" },
    .{ .code = "print(min([2, 3]))\n", .from = "2", .to = "4", .after = "3\n" },
    .{ .code = "print(sorted([2, 3]))\n", .from = "2", .to = "4", .after = "[3, 4]\n" },
    .{ .code = "print(len([x for x in [7] if x > 0]))\n", .from = "7", .to = "0", .after = "0\n" },
    .{ .code = "print(max(3, 2))\n", .from = "2", .to = "4", .after = "4\n" },
    .{ .code = "print(sorted([2, -1, 2, 0]))\n", .from = "2", .to = "4", .after = "[-1, 0, 4, 4]\n" },
    .{ .code = "print(sorted([3, 1, 2], reverse=True))\n", .from = "2", .to = "4", .after = "[4, 3, 1]\n" },
    .{ .code = "print(sum(x for x in [2, 5] if x > 3))\n", .from = "2", .to = "4", .after = "9\n" },
    .{ .code = "xs = [5, 3, 8]\nys = sorted(xs)\nprint(ys[0] + ys[-1])\n", .from = "3", .to = "9", .after = "14\n" },
    .{ .code = D ++ "print(pos_count([1, -2, 3, -4]))\n", .from = "3", .to = "0", .after = "1\n" },
    .{ .code = "print(min(x * x for x in [2, 3]))\n", .from = "2", .to = "4", .after = "9\n" },
    .{ .code = "print(len([x for x in [7, -2, 9] if x % 3]))\n", .from = "9", .to = "8", .after = "3\n" },
};

fn runOn(b: api.Backend, id: u64, bytes: []const u8) !struct { api.Status, []u8 } {
    const job = try b.start(id, .{ .kind = .graph, .bytes = bytes }, 50_000_000);
    const r = while (true) switch (b.poll(job)) {
        .running => std.Thread.yield() catch {},
        .done => |x| break x,
    };
    defer b.release(id);
    return .{ r.status, try gpa.dupe(u8, r.out) };
}

test "runner frontend integer list: the circuit owns min, sorted and filters: a mutated input prints CPython's mutated stdout (TF_RUNNER_GRAPH_BANK)" {
    const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(gpa, std.mem.span(g));
    defer bank.deinit();
    const backend = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(backend);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = backend };
    defer d.deinit();
    const b = d.backend();
    for (mutations, 0..) |m, i| {
        const bytes = (try table.translate(gpa, &bank, m.code)) orelse {
            std.debug.print("not on the bank: {s}\n", .{m.code});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        // the input as a packet value: `"2"`, in every packet of the sequence
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m.from});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m.to});
        defer gpa.free(to);
        if (std.mem.count(u8, bytes, from) == 0) {
            std.debug.print("input {s} not a packet value: {s}\n{s}\n", .{ m.from, m.code, bytes });
            return error.TestExpectedInput;
        }
        const mutated = try std.mem.replaceOwned(u8, gpa, bytes, from, to);
        defer gpa.free(mutated);
        const status, const out = try runOn(b, 1000 + i, mutated);
        defer gpa.free(out);
        if (status != .ok or !std.mem.eql(u8, out, m.after))
            std.debug.print("host-owned: {s} with {s} -> {s}: {t} '{s}', CPython '{s}'\n", .{ m.code, m.from, m.to, status, out, m.after });
        try std.testing.expectEqual(api.Status.ok, status);
        try std.testing.expectEqualStrings(m.after, out);
    }
}
