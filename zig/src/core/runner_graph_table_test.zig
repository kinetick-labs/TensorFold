//! A checked whole script on the graph bank (runner_graph_table.zig): Python's output, no compile, the same tokens.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const table = @import("runner_graph_table.zig");
const integer_python = @import("runner_frontend_integer_python.zig");

const gpa = std.testing.allocator;

/// A graph backend that prints each packet's own text as its value: `-4` prints -4.
const Echo = struct {
    jobs: std.ArrayList([]u8) = .empty,
    out: [64]u8 = undefined,

    fn backend(e: *Echo) api.Backend {
        return .{ .ctx = e, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        try e.jobs.append(gpa, try gpa.dupe(u8, p.bytes));
        return e.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: api.JobId) api.Poll {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        const v = std.fmt.bufPrint(&e.out, "{s}\n", .{e.jobs.items[job]}) catch unreachable;
        return .{ .done = .{ .status = .ok, .out = v, .steps = 1 } };
    }
    fn cancel(_: *anyopaque, _: api.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(e: *Echo) void {
        for (e.jobs.items) |j| gpa.free(j);
        e.jobs.deinit(gpa);
    }
};

/// A backend no test here may start (the circuit, when every program is a graph one).
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

/// The sequence's end through Dispatch, its text copied (owned by the caller).
fn run(d: *route.Dispatch, bytes: []const u8) !struct { status: api.Status, out: []u8 } {
    const b = d.backend();
    const job = try b.start(1, .{ .kind = .graph, .bytes = bytes }, 50_000_000);
    const r = while (true) switch (b.poll(job)) {
        .running => std.Thread.yield() catch {},
        .done => |x| break x,
    };
    defer b.release(1);
    return .{ .status = r.status, .out = try gpa.dupe(u8, r.out) };
}

test "runner graph table: text pieces, integral floats and fixed scales print as the runtime does; bad pieces fault" {
    var echo: Echo = .{};
    defer echo.deinit();
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = echo.backend() };
    defer d.deinit();
    const ok = try run(&d, "TFGS1\nLJ (\nIJ -4\nLJ , \nIJ 1\nLN )\nFS -21\n3S 5\n3S 123\n2S -1250\n1N -3\nBS 1\nBN 0\n");
    defer gpa.free(ok.out);
    try std.testing.expectEqual(api.Status.ok, ok.status);
    try std.testing.expectEqualStrings("(-4, 1)\n-21.0 0.005 0.123 -12.50 -0.3\nTrue False\n", ok.out);
    // negative controls: an unknown separator or kind, a bool that is not 0/1, a value that is not a number
    for ([_][]const u8{ "TFGS1\nIX 7\n", "TFGS1\nQN 7\n", "TFGS1\nBN 7\n", "TFGS1\nFN x\n", "TFGS1\nLQ (\n" }) |bad| {
        const r = try run(&d, bad);
        defer gpa.free(r.out);
        try std.testing.expectEqual(api.Status.fault, r.status);
    }
}

test "runner graph table: the frontend's checked table keeps its outputs, endings and styles" {
    const lowered = (try integer_python.lowerWithCost(gpa, "a=-17\nb=5\nprint(a//b, a%b)\nprint(divmod(a,b))\n", 0)).?;
    defer gpa.free(lowered.source);
    const t = lowered.table.?;
    try std.testing.expectEqual(@as(usize, 3), t.outputs);
    try std.testing.expectEqualSlices(u8, " \n\n", t.endings[0..3]);
    try std.testing.expectEqual(.divmod, t.nodes[t.roots[2]].op);
    for (t.nodes[0..t.count]) |n| try std.testing.expectEqualStrings("", n.text);
}

/// Whole problems T5 admits (div_big, S1 divmod, mul_big) and edges, with CPython's stdout under the lane prelude.
const cases = [_][2][]const u8{
    .{ "a=-23777868243178464\nb=-41832496\nprint(a//b, a%b)\nprint(a/b)\nprint(divmod(a,b))\n", "568406634 0\n568406634.0\n(568406634, 0)\n" },
    .{ "from fractions import Fraction\nn=44253970636272\nprint(n//48, n%48)\nprint(Fraction(n,-48))\nprint(n/-48)\n", "921957721589 0\n-921957721589\n-921957721589.0\n" },
    .{ "from fractions import Fraction\nn=-160371104\nd=932\nf=Fraction(n,d)\nprint(f, float(f))\nprint(n//d, n%d)\nimport math\nprint(math.gcd(abs(n),d))\n", "-172072 -172072.0\n-172072 0\n932\n" },
    .{ "from fractions import Fraction\nn=32685677580\nd=-4826\nf=Fraction(n,d)\nprint(f, f.numerator, f.denominator)\nprint(n/d)\nimport math\nprint(math.gcd(n,4826))\n", "-6772830 -6772830 1\n-6772830.0\n4826\n" },
    .{ "A = 314159265358979323846264338327950288419716939937510\nB = 271828182845904523536\nprint(divmod(A, B))\n", "(1155727349790921717910215407028, 23663814570294126502)\n" },
    .{ "from decimal import Decimal\nprint(Decimal('-27.201')*Decimal('4952356'))\n", "-134709035.556\n" },
    .{ "from decimal import Decimal\nprint(Decimal('562.1') * Decimal('7.691'))\n", "4323.1111\n" },
    .{ "from decimal import Decimal\nprint(Decimal('0.002')*Decimal('3'))\nprint(Decimal('-0.5')*Decimal('0.25'))\nprint(Decimal('1.50')*Decimal('2'))\n", "0.006\n-0.125\n3.00\n" },
    .{ "print(divmod(-7, 2))\n", "(-4, 1)\n" },
    .{ "x=10**20\ny=10**20+1\nprint(x<y, x<=y, x>y, x>=y, x==y, x!=y)\n", "True True False False False True\n" },
    .{ "x=10**20\ny=10**20+1\nprint(y==x, y!=x, x==x, x!=x, -y==-y, -y!=y)\n", "False True True False True True\n" },
    .{ "from fractions import Fraction\nprint(float(Fraction(-84, 4)))\n", "-21.0\n" },
    .{ "a=-17\nb=5\nprint(a//b, a%b, (a<b)+1)\n", "-4 3 2\n" },
    .{ "import math\nprint(math.gcd(-12, 18), math.gcd(12, -18))\n", "6 6\n" },
};

/// Scripts the table leaves to the route behind it: a fraction, a computed divisor (T1's binding owns it), a string.
const refused = [_][]const u8{ "print(7/2)\n", "a=3\nprint(100//(a+1))\n", "print(len(str(2**100)))\n" };

fn openBank() !graph.Bank {
    const g = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return graph.Bank.load(gpa, std.mem.span(g));
}

test "runner graph table: whole problems print CPython's output on the real bank (TF_RUNNER_GRAPH_BANK)" {
    var bank = try openBank();
    defer bank.deinit();
    const g = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(g);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = g };
    defer d.deinit();
    const has_gcd = bank.plan("gcd-16") != null;
    for (cases) |c| {
        // gcd needs T1's algorithm plan; a bank without it leaves the script to the frontend's route
        if (!has_gcd and std.mem.indexOf(u8, c[0], "gcd") != null) {
            try std.testing.expectEqual(null, try table.translate(gpa, &bank, c[0]));
            continue;
        }
        const bytes = (try table.translate(gpa, &bank, c[0])) orelse {
            std.debug.print("not on the bank: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer gpa.free(bytes);
        const r = try run(&d, bytes);
        defer gpa.free(r.out);
        if (r.status != .ok or !std.mem.eql(u8, r.out, c[1])) std.debug.print("{s}: {t} '{s}'\n", .{ c[0], r.status, r.out });
        try std.testing.expectEqual(api.Status.ok, r.status);
        try std.testing.expectEqualStrings(c[1], r.out);
    }
    for (refused) |code| try std.testing.expectEqual(null, try table.translate(gpa, &bank, code));
}

test "runner graph table: the bank computes from the literals; the frontend's proof values never reach the output" {
    var bank = try openBank();
    defer bank.deinit();
    const g = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(g);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = g };
    defer d.deinit();
    const lowered = (try @import("runner_frontend_fraction.zig").lowerWithCost(gpa, cases[1][0], 0)).?;
    gpa.free(lowered.source);
    var t = lowered.table.?;
    // a wrong proof on every computed node: the output must not move
    for (t.nodes[0..t.count]) |*n| if (n.op != .literal) {
        n.value = 12345;
    };
    const same = (try table.lower(gpa, &bank, &t)).?;
    defer gpa.free(same);
    const r = try run(&d, same);
    defer gpa.free(r.out);
    try std.testing.expectEqualStrings(cases[1][1], r.out);
    // a changed literal changes the output (n + 48): the run reads its inputs
    for (t.nodes[0..t.count]) |*n| if (n.op == .literal and n.value == 44253970636272) {
        n.value += 48;
    };
    const moved = (try table.lower(gpa, &bank, &t)).?;
    defer gpa.free(moved);
    const m = try run(&d, moved);
    defer gpa.free(m.out);
    try std.testing.expectEqualStrings("921957721590 0\n-921957721590\n-921957721590.0\n", m.out);
}

/// A frontend that counts its calls and abstains (the compile route a graph script must not reach).
const Inner = struct {
    calls: u32 = 0,
    fn frontend(i: *Inner) api.Frontend {
        return .{ .ctx = i, .translate = translate };
    }
    fn translate(ctx: ?*anyopaque, _: std.mem.Allocator, _: api.Cell) std.mem.Allocator.Error!api.Translation {
        const i: *Inner = @ptrCast(@alignCast(ctx.?));
        i.calls += 1;
        return .{ .abstain = .unsupported };
    }
};

test "runner graph table: Routed takes a checked fence script before the compile; earlier cells or refusals go on" {
    var bank = try openBank();
    defer bank.deinit();
    var inner: Inner = .{};
    var routed: route.Routed = .{ .inner = inner.frontend(), .bank = &bank };
    const fe = routed.frontend();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try fe.translate(fe.ctx, a, .{ .form = .fence, .lang = "python", .code = cases[0][0] });
    try std.testing.expect(t == .program and t.program.kind == .graph);
    try std.testing.expectEqual(@as(u32, 0), inner.calls);
    // a computed divisor: T1's direct binding owns it before the compile and prints CPython's 25
    const owned = try fe.translate(fe.ctx, a, .{ .form = .fence, .lang = "python", .code = refused[1] });
    try std.testing.expect(owned == .program and owned.program.kind == .graph);
    try std.testing.expectEqual(@as(u32, 0), inner.calls);
    const g = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(g);
    var d: route.Dispatch = .{ .gpa = gpa, .circuit = Never.backend(), .graph = g };
    defer d.deinit();
    const r = try run(&d, owned.program.bytes);
    defer gpa.free(r.out);
    try std.testing.expectEqual(api.Status.ok, r.status);
    try std.testing.expectEqualStrings("25\n", r.out);
    // a string the bank cannot hold: no route takes it, so the frontend is asked
    const declined = try fe.translate(fe.ctx, a, .{ .form = .fence, .lang = "python", .code = refused[2] });
    try std.testing.expect(declined == .abstain);
    try std.testing.expectEqual(@as(u32, 1), inner.calls);
    // a later cell could have rebound a name the script uses: the frontend's own checks decide
    _ = try fe.translate(fe.ctx, a, .{ .form = .fence, .lang = "python", .code = cases[0][0], .prior = &.{"divmod = max\n"} });
    try std.testing.expectEqual(@as(u32, 2), inner.calls);
}
