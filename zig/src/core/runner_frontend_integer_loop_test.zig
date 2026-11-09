//! Loops, multi-statement defs, digit reversal on the graph route: the ELF-only cells, CPython stdout, packet mutation.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const table = @import("runner_graph_table.zig");
const integer_python = @import("runner_frontend_integer_python.zig");

const gpa = std.testing.allocator;

/// The ten ELF-only cells (t2-evidence/elf-vs-sandbox/elf10.jsonl, deduplicated), then loops; CPython 3's stdout.
const cases = [_][2][]const u8{
    .{ "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nfor n in [0, 1, 5, 12]:\n    print(factorial(n))\n", "1\n1\n120\n479001600\n" },
    .{ "def check(n):\n    rev = int(str(n)[::-1])\n    return n == 2 * rev - 1\n\nprint(check(70))\nprint(check(13))\nprint(check(0))\n", "False\nFalse\nFalse\n" },
    .{ "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nassert pos_count([1, -2, 3, -4]) == 2\nprint(pos_count([1, -2, 3, -4]))", "2\n" },
    .{ "def tetrahedral_number(n):\n    return n * (n + 1) * (n + 2) // 6\n\nassert tetrahedral_number(5) == 35\nprint(tetrahedral_number(5))", "35\n" },
    .{ "def centered_hexagonal_number(n):\n    return 3 * n * (n - 1) + 1\n\nassert centered_hexagonal_number(10) == 271\nprint(centered_hexagonal_number(10))", "271\n" },
    .{ "def square_perimeter(side):\n    return 4 * side\n\nassert square_perimeter(10) == 40\nprint(square_perimeter(10))", "40\n" },
    .{ "s = 0\nfor i in range(1, 11):\n    s += i * i\nprint(s)\n", "385\n" },
    .{ "t = 1\nfor i in range(10, 0, -3):\n    t *= i\nprint(t)\n", "280\n" },
    .{ "def digits(n):\n    c = 0\n    while n > 0:\n        n //= 10\n        c += 1\n    return c\nprint(digits(98765), digits(7))\n", "5 1\n" },
    .{ "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(6))\n", "720\n" },
    .{ "def rev(n):\n    return int(str(n)[::-1])\nprint(rev(1200), rev(7), rev(123456789))\n", "21 7 987654321\n" },
    .{ "def f(n):\n    m = n * 2\n    return m\nprint(f(3))\n", "6\n" },
    .{ "def f(n):\n    for i in range(n):\n        n += i\n    return n\nprint(f(3))\n", "6\n" },
    .{ "def tri(n):\n    \"\"\"Triangular number.\"\"\"\n    total = 0\n    for k in range(n + 1): total += k\n    return total\nprint(tri(10), tri(0))\n", "55 0\n" },
    .{ "def g(n):\n    s = 0\n    for i in range(n):\n        for j in range(i):\n            s += j\n    return s\nprint(g(5))\n", "10\n" },
    .{ "x = 7\nfor k in [3, 4]:\n    x -= k\n    print(x)\nprint(x)\n", "4\n0\n0\n" },
    .{ "def p(b, e):\n    r = 1\n    for _ in range(e):\n        r *= b\n    return r\nprint(p(3, 7), p(-2, 5))\n", "2187 -32\n" },
    .{ "def s(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total\nprint(s(4321))\n", "10\n" },
    .{ "def f(n):\n    return 1 if n < 2 else n * f(n - 1)\nprint(f(5))\n", "120\n" },
    // both branches recurse: every level but the base leaves is chosen on the graph
    .{ "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nprint(fib(6))\n", "8\n" },
    .{ "def avg(a, b):\n    s = a + b\n    s //= 2\n    return s\nprint(avg(7, 10))\n", "8\n" },
};

/// Shapes left to the sandbox: unbounded or over-budget loops, break/continue, floats, value-chosen prints.
const refused = [_][]const u8{
    // unbounded, or past the 32-trip / 24-call budget
    "n = 5\nwhile n > 0:\n    n += 1\nprint(n)\n",
    "def f(n):\n    while True:\n        n += 1\n    return n\nprint(f(1))\n",
    "s = 0\nfor i in range(100):\n    s += i\nprint(s)\n",
    "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(30))\n",
    // break and continue, and an `if` inside a loop
    "s = 0\nfor i in range(5):\n    if i == 3:\n        break\n    s += i\nprint(s)\n",
    "s = 0\nfor i in range(5):\n    continue\nprint(s)\n",
    "def f(n):\n    for i in range(n):\n        break\n    return n\nprint(f(3))\n",
    // float accumulators
    "s = 0.0\nfor i in range(3):\n    s += i\nprint(s)\n",
    "s = 0\nfor i in range(3):\n    s += i / 2\nprint(s)\n",
    // `%=` under a trip's test; truthiness tests; a value-chosen list to loop over
    "x = 100\nfor i in range(3):\n    x %= 7\nprint(x)\n",
    "def s(n):\n    t = 0\n    while n:\n        t += n % 10\n        n //= 10\n    return t\nprint(s(42))\n",
    "for x in [y for y in [1, 2] if y > 1]:\n    print(x)\n",
    "for x in range(3):\n    print(x)\n",
    // recursion past the node budget; a guard whose other branch raises
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nprint(fib(20))\n",
    "def f(n):\n    if n == 0:\n        return 0\n    return 10 // n\nprint(f(0))\n",
    // Python raises: a local read before assignment, a negative int's reversed text, a zero step
    "def f(n):\n    s += n\n    s = 0\n    return s\nprint(f(3))\n",
    "print(int(str(-12)[::-1]))\n",
    "s = 0\nfor i in range(1, 5, 0):\n    s += i\nprint(s)\n",
    // a loop variable after a range loop; a return inside a loop; a print inside a def; a missing return
    "for i in range(3):\n    pass\nprint(i)\n",
    "def f(n):\n    for i in range(n):\n        return i\n    return 0\nprint(f(3))\n",
    "def f(n):\n    print(n)\n    return n\nprint(f(3))\n",
    "def f(n):\n    m = n\nprint(f(3))\n",
    "def f(n):\n    if n > 0:\n        return 1\nprint(f(3))\n",
    // math's factorial imported bare: a def of that name is not taken
    "from math import factorial\ndef factorial(n):\n    return n\nprint(factorial(3))\n",
};

test "runner frontend integer loop: suites lower to CPython's proved values; other shapes refuse" {
    var bad: usize = 0;
    for (cases) |c| {
        const lowered = try integer_python.lowerWithCost(gpa, c[0], 0) orelse {
            std.debug.print("refused: {s}\n", .{c[0]});
            bad += 1;
            continue;
        };
        defer gpa.free(lowered.source);
        const t = lowered.table.?;
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        for (t.roots[0..t.outputs], t.endings[0..t.outputs]) |root, end| {
            const n = t.nodes[root];
            switch (n.op) {
                .eq, .ne, .lt, .le, .gt, .ge => try out.appendSlice(gpa, if (n.value != 0) "True" else "False"),
                else => try out.print(gpa, "{d}", .{n.value}),
            }
            try out.append(gpa, end);
        }
        try std.testing.expectEqualStrings(c[1], out.items);
        try std.testing.expect(lowered.graph_only); // calc and ELF never take a loop
    }
    if (bad != 0) return error.TestExpectedLowering;
    for (refused) |code| {
        if (try integer_python.lowerWithCost(gpa, code, 0)) |g| {
            gpa.free(g.source);
            std.debug.print("lowered, must refuse: {s}\n", .{code});
            return error.TestExpectedRefusal;
        }
    }
}

/// A script, a literal to change in its packet, and CPython's stdout for the script so changed (asserts dropped).
const mutations = [_][4][]const u8{
    .{ "def fact(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nprint(fact(37 - 30))\n", "37", "35", "120\n" },
    // within the schedule's headroom: 6 trips run as 8, so a larger input still prints CPython's value
    .{ "def fact(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nprint(fact(37 - 30))\n", "37", "38", "40320\n" },
    .{ "def check(n):\n    rev = int(str(n)[::-1])\n    return n == 2 * rev - 1\n\nprint(check(70))\nprint(check(13))\nprint(check(0))\n", "70", "73", "True\nFalse\nFalse\n" },
    .{ "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\n\nassert pos_count([1, -2, 3, -4]) == 2\nprint(pos_count([1, -2, 3, -4]))", "3", "0", "1\n" },
    .{ "def digits(n):\n    c = 0\n    while n > 0:\n        n //= 10\n        c += 1\n    return c\nprint(digits(98765))\n", "98765", "432", "3\n" },
    .{ "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(37 - 30))\n", "37", "35", "120\n" },
    .{ "def rev(n):\n    return int(str(n)[::-1])\nprint(rev(1200))\n", "1200", "3400", "43\n" },
    .{ "s = 0\nfor i in range(100, 140, 5):\n    s += i\nprint(s)\n", "140", "125", "550\n" },
};

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

fn run(b: api.Backend, id: u64, bytes: []const u8, want: []const u8, code: []const u8) !void {
    const job = try b.start(id, .{ .kind = .graph, .bytes = bytes }, 50_000_000);
    const r = while (true) switch (b.poll(job)) {
        .running => std.Thread.yield() catch {},
        .done => |x| break x,
    };
    defer b.release(id);
    if (r.status != .ok or !std.mem.eql(u8, r.out, want)) std.debug.print("{s}: {t} '{s}'\n", .{ code, r.status, r.out });
    try std.testing.expectEqual(api.Status.ok, r.status);
    try std.testing.expectEqualStrings(want, r.out);
}

test "runner frontend integer loop: the cells run on the real bank, and a changed packet value follows CPython (TF_RUNNER_GRAPH_BANK)" {
    const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(gpa, std.mem.span(g));
    defer bank.deinit();
    const backend = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(backend);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = backend };
    defer d.deinit();
    const b = d.backend();
    var missing: usize = 0;
    for (cases, 0..) |c, i| {
        const bytes = (try table.translate(gpa, &bank, c[0])) orelse {
            std.debug.print("not on the bank: {s}\n", .{c[0]});
            missing += 1;
            continue;
        };
        defer gpa.free(bytes);
        try run(b, i + 1, bytes, c[1], c[0]);
    }
    for (mutations, 0..) |m, i| {
        const bytes = (try table.translate(gpa, &bank, m[0])) orelse {
            std.debug.print("not on the bank: {s}\n", .{m[0]});
            missing += 1;
            continue;
        };
        defer gpa.free(bytes);
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[1]});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[2]});
        defer gpa.free(to);
        // the value must be in the packet: a folded decision would leave nothing to change
        try std.testing.expect(std.mem.indexOf(u8, bytes, from) != null);
        const changed = try std.mem.replaceOwned(u8, gpa, bytes, from, to);
        defer gpa.free(changed);
        try run(b, 100 + i, changed, m[3], m[0]);
    }
    for (refused) |code| try std.testing.expectEqual(null, try table.translate(gpa, &bank, code));
    if (missing != 0) return error.TestExpectedRoute;
}
