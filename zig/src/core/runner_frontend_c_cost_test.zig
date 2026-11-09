const std = @import("std");
const cost = @import("runner_frontend_c_cost.zig");
const t = std.testing;
const budget = 10_522;
const Node = struct {
    kind: ?[]const u8 = null,
    id: []const u8 = "",
    name: []const u8 = "",
    type: struct { qualType: []const u8 = "" } = .{},
    loc: struct { offset: usize = 100 } = .{},
    inner: []const Node = &.{},
    opcode: []const u8 = "",
    value: ?[]const u8 = null,
    referencedDecl: ?struct { id: []const u8, kind: []const u8, name: []const u8 } = null,
};

const Fixture = struct {
    a: std.mem.Allocator,

    fn node(f: Fixture, kind: []const u8, typ: []const u8, inner: []const Node) !Node {
        return .{ .kind = kind, .type = .{ .qualType = typ }, .inner = try f.a.dupe(Node, inner) };
    }
    fn number(f: Fixture, n: i128, typ: []const u8) !Node {
        var out = try f.node("IntegerLiteral", typ, &.{});
        out.value = try std.fmt.allocPrint(f.a, "{d}", .{n});
        return out;
    }
    fn reference(id: []const u8, typ: []const u8) Node {
        return .{
            .kind = "DeclRefExpr",
            .type = .{ .qualType = typ },
            .referencedDecl = .{ .id = id, .kind = "VarDecl", .name = id },
        };
    }
    fn binary(f: Fixture, op: []const u8, typ: []const u8, x: Node, y: Node) !Node {
        var out = try f.node("BinaryOperator", typ, &.{ x, y });
        out.opcode = op;
        return out;
    }
    fn variable(f: Fixture, id: []const u8, typ: []const u8, init: ?Node) !Node {
        var out = try f.node("VarDecl", typ, if (init) |n| &.{n} else &.{});
        out.id = id;
        out.name = id;
        return out;
    }
    fn declare(f: Fixture, id: []const u8, typ: []const u8, init: ?Node) !Node {
        return f.node("DeclStmt", "", &.{try f.variable(id, typ, init)});
    }
    fn unary(f: Fixture, op: []const u8, typ: []const u8, child: Node) !Node {
        var out = try f.node("UnaryOperator", typ, &.{child});
        out.opcode = op;
        return out;
    }
    fn block(f: Fixture, nodes: []const Node) !Node {
        return f.node("CompoundStmt", "", nodes);
    }
    fn ret(f: Fixture, n: Node) !Node {
        return f.node("ReturnStmt", "", &.{n});
    }
    fn function(f: Fixture, name: []const u8, params: []const Node, body: []const Node) !Node {
        var inner: std.ArrayList(Node) = .empty;
        try inner.appendSlice(f.a, params);
        try inner.append(f.a, try f.block(body));
        var out = try f.node("FunctionDecl", "int (void)", inner.items);
        out.id = name;
        out.name = name;
        return out;
    }
    fn call(f: Fixture, name: []const u8, typ: []const u8, args: []const Node) !Node {
        var inner: std.ArrayList(Node) = .empty;
        try inner.append(f.a, .{
            .kind = "DeclRefExpr",
            .referencedDecl = .{ .id = name, .kind = "FunctionDecl", .name = name },
        });
        try inner.appendSlice(f.a, args);
        return f.node("CallExpr", typ, inner.items);
    }
    fn string(f: Fixture, value: []const u8) !Node {
        var out = try f.node("StringLiteral", "char[128]", &.{});
        out.value = value;
        return out;
    }
    fn counted(f: Fixture, id: []const u8, typ: []const u8, start: i128, stop: Node, body: []const Node) !Node {
        return f.node("ForStmt", "", &.{
            try f.declare(id, typ, try f.number(start, typ)),
            .{},
            try f.binary("<", "int", reference(id, typ), stop),
            try f.unary("++", typ, reference(id, typ)),
            try f.block(body),
        });
    }
    fn estimate(f: Fixture, functions: []const Node, body: []const Node, limit: usize) !?usize {
        var top: std.ArrayList(Node) = .empty;
        for ([_][]const u8{ "printf", "puts", "putchar", "tf_print_int" }) |name| {
            try top.append(f.a, .{ .kind = "FunctionDecl", .id = name, .name = name, .loc = .{ .offset = 1 } });
        }
        try top.appendSlice(f.a, functions);
        try top.append(f.a, try f.function("main", &.{}, body));
        const root = try f.node("TranslationUnitDecl", "", top.items);
        const json = try std.json.Stringify.valueAlloc(f.a, root, .{});
        return cost.estimate(t.allocator, json, 100, limit);
    }
};

test "C cost admission includes startup and saturates at the caller limit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const body = [_]Node{try f.ret(try f.number(0, "int"))};
    const steps = (try f.estimate(&.{}, &body, budget)).?;
    try t.expect(steps >= 200);
    try t.expectEqual(@as(?usize, steps), try f.estimate(&.{}, &body, steps));
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &body, steps - 1));
    try t.expectEqual(@as(?usize, null), try cost.estimate(t.allocator, "{", 0, budget));
    try t.expectEqual(@as(?usize, null), try cost.estimate(t.allocator, "{}", 0, budget));
}

test "counted arithmetic admits a small loop and refuses huge counts before expansion" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const sum = Fixture.reference("sum", "int");
    const i = Fixture.reference("i", "int");
    const update = try f.binary("=", "int", sum, try f.binary("+", "int", sum, try f.binary("*", "int", i, i)));
    for ([_]i128{ 10, 1_000_000_000 }) |n| {
        const body = [_]Node{
            try f.declare("sum", "int", try f.number(0, "int")),
            try f.counted("i", "int", 0, try f.number(n, "int"), &.{update}),
            try f.call("tf_print_int", "void", &.{sum}),
        };
        const steps = try f.estimate(&.{}, &body, budget);
        try t.expect((steps != null) == (n == 10));
    }
}

test "body mutation of invariant bounds and unsigned induction wrap abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const bound = Fixture.reference("n", "int");
    const mutation = try f.unary("++", "int", bound);
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        try f.declare("n", "int", try f.number(3, "int")),
        try f.counted("i", "int", 0, bound, &.{mutation}),
    }, budget));
    const i = Fixture.reference("i", "unsigned int");
    const loop = try f.node("ForStmt", "", &.{
        try f.declare("i", "unsigned int", try f.number(std.math.maxInt(u32), "unsigned int")),
        .{},
        try f.binary("<=", "int", i, try f.number(std.math.maxInt(u32), "unsigned int")),
        try f.unary("++", "unsigned int", i),
        try f.block(&.{}),
    });
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{loop}, budget));
}

test "acyclic calls are charged and reachable recursion abstains" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const recursive = try f.function("recur", &.{}, &.{try f.ret(try f.call("recur", "int", &.{}))});
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{recursive}, &.{try f.call("recur", "int", &.{})}, budget));
    const leaf = try f.function("leaf", &.{}, &.{try f.ret(try f.number(7, "int"))});
    try t.expect((try f.estimate(&.{leaf}, &.{try f.call("leaf", "int", &.{})}, budget)) != null);
}

test "uncertain branches charge the larger path and include expensive alternatives" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const light = try f.counted("i", "int", 0, try f.number(180, "int"), &.{});
    const heavy = try f.counted("i", "int", 0, try f.number(200, "int"), &.{});
    const base = (try f.estimate(&.{}, &.{heavy}, budget)).?;
    const flags = try f.declare("flags", "int[2]", try f.node("InitListExpr", "int[2]", &.{
        try f.number(0, "int"), try f.number(1, "int"),
    }));
    const condition = try f.node("ArraySubscriptExpr", "int", &.{
        Fixture.reference("flags", "int[2]"), try f.number(0, "int"),
    });
    try t.expect((try f.estimate(&.{}, &.{
        flags,
        try f.node("IfStmt", "", &.{ condition, light, heavy }),
    }, base + 256)) != null);
    const huge = try f.counted("i", "int", 0, try f.number(1_000_000, "int"), &.{});
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        flags,
        try f.node("IfStmt", "", &.{ condition, light, huge }),
    }, budget));
}

test "readonly string ranking is generic and nonconstant printf formats abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    var param = try f.node("ParmVarDecl", "const char *", &.{});
    param.id = "s";
    const n = Fixture.reference("n", "int");
    const index = try f.node("ArraySubscriptExpr", "char", &.{ Fixture.reference("s", "const char *"), n });
    const length = try f.function("measure_text", &.{param}, &.{
        try f.declare("n", "int", try f.number(0, "int")),
        try f.node("WhileStmt", "", &.{ index, try f.block(&.{try f.unary("++", "int", n)}) }),
        try f.ret(n),
    });
    try t.expect((try f.estimate(&.{length}, &.{
        try f.call("puts", "int", &.{try f.string("\"ordinary text\"")}),
        try f.call("tf_print_int", "void", &.{try f.call("measure_text", "int", &.{try f.string("\"hello\"")})}),
    }, budget)) != null);
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        try f.declare("fmt", "const char *", try f.string("\"%d\"")),
        try f.call("printf", "int", &.{ Fixture.reference("fmt", "const char *"), try f.number(3, "int") }),
    }, budget));
}

fn fibonacci(f: Fixture, largest: i128) !?usize {
    const u = "unsigned int";
    const wide = "unsigned long long";
    const n = Fixture.reference("n", u);
    const a = Fixture.reference("a", wide);
    const b = Fixture.reference("b", wide);
    const next = Fixture.reference("next", wide);
    var param = try f.node("ParmVarDecl", u, &.{});
    param.id = "n";
    const loop = try f.node("ForStmt", "", &.{
        try f.declare("j", u, try f.number(2, u)),
        .{},
        try f.binary("<=", "int", Fixture.reference("j", u), n),
        try f.unary("++", u, Fixture.reference("j", u)),
        try f.block(&.{
            try f.binary("=", wide, next, try f.binary("+", wide, a, b)),
            try f.binary("=", wide, a, b),
            try f.binary("=", wide, b, next),
        }),
    });
    const fib = try f.function("fib_iter", &.{param}, &.{
        try f.node("IfStmt", "", &.{ try f.binary("==", "int", n, try f.number(0, u)), try f.ret(try f.number(0, wide)) }),
        try f.node("IfStmt", "", &.{ try f.binary("==", "int", n, try f.number(1, u)), try f.ret(try f.number(1, wide)) }),
        try f.declare("a", wide, try f.number(0, wide)),
        try f.declare("b", wide, try f.number(1, wide)),
        try f.declare("next", wide, null),
        loop,
        try f.ret(b),
    });
    const values = try f.node("InitListExpr", "unsigned int[4]", &.{
        try f.number(0, u), try f.number(1, u), try f.number(10, u), try f.number(largest, u),
    });
    const indices = Fixture.reference("indices", "unsigned int[4]");
    var all_size = try f.node("UnaryExprOrTypeTraitExpr", u, &.{indices});
    all_size.name = "sizeof";
    var one_size = try f.node("UnaryExprOrTypeTraitExpr", u, &.{try f.node("ArraySubscriptExpr", u, &.{ indices, try f.number(0, "int") })});
    one_size.name = "sizeof";
    const element = try f.node("ArraySubscriptExpr", u, &.{ indices, Fixture.reference("i", u) });
    return f.estimate(&.{fib}, &.{
        try f.declare("indices", "unsigned int[4]", values),
        try f.declare("count", u, try f.binary("/", u, all_size, one_size)),
        try f.counted("i", u, 0, Fixture.reference("count", u), &.{
            try f.declare("value", u, element),
            try f.call("printf", "int", &.{
                try f.string("\"F(%u) = %llu\\n\""),
                Fixture.reference("value", u),
                try f.call("fib_iter", wide, &.{Fixture.reference("value", u)}),
            }),
        }),
    }, budget);
}

test "F40 interval calls retain the 32 bit wide formatting path and F93 exceeds budget" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    try t.expect((try fibonacci(f, 40)) != null);
    try t.expectEqual(@as(?usize, null), try fibonacci(f, 93));
}

test "Euclid structural summary admits nonnegative bounds without computing a GCD" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const a = Fixture.reference("a", "int");
    const b = Fixture.reference("b", "int");
    const body = try f.block(&.{
        try f.declare("tmp", "int", try f.binary("%", "int", a, b)),
        try f.binary("=", "int", a, b),
        try f.binary("=", "int", b, Fixture.reference("tmp", "int")),
    });
    try t.expect((try f.estimate(&.{}, &.{
        try f.declare("a", "int", try f.number(462, "int")),
        try f.declare("b", "int", try f.number(1071, "int")),
        try f.node("WhileStmt", "", &.{ try f.binary("!=", "int", b, try f.number(0, "int")), body }),
        try f.call("tf_print_int", "void", &.{a}),
    }, budget)) != null);
}

test "uninitialized values and unknown termination abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        try f.declare("x", "int", null),
        try f.call("tf_print_int", "void", &.{Fixture.reference("x", "int")}),
    }, budget));
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        try f.node("WhileStmt", "", &.{ try f.number(1, "int"), try f.block(&.{}) }),
    }, budget));
}

test "pointer aliases cannot modify a string behind a caller bound" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    var param = try f.node("ParmVarDecl", "char *", &.{});
    param.id = "s";
    const target = try f.node("ArraySubscriptExpr", "char", &.{
        Fixture.reference("s", "char *"), try f.number(0, "int"),
    });
    const mutator = try f.function("mutate", &.{param}, &.{
        try f.binary("=", "char", target, try f.number(1, "int")),
        try f.ret(try f.number(0, "int")),
    });
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{mutator}, &.{
        try f.call("mutate", "int", &.{try f.string("\"abc\"")}),
    }, budget));
}

test "narrowing casts cannot discard feasible loop bounds during branch refinement" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator() };
    const values = try f.node("InitListExpr", "int[2]", &.{
        try f.number(0, "int"), try f.number(256, "int"),
    });
    const n = Fixture.reference("n", "int");
    const narrowing = try f.node("CStyleCastExpr", "unsigned char", &.{n});
    try t.expectEqual(@as(?usize, null), try f.estimate(&.{}, &.{
        try f.declare("values", "int[2]", values),
        try f.declare("n", "int", try f.node("ArraySubscriptExpr", "int", &.{
            Fixture.reference("values", "int[2]"), try f.number(0, "int"),
        })),
        try f.node("IfStmt", "", &.{ narrowing, try f.ret(try f.number(0, "int")) }),
        try f.counted("i", "int", 0, n, &.{}),
    }, 1000));
}
