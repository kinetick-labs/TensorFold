const std = @import("std");
const inputs = @import("runner_frontend_c_inputs.zig");
const support = @import("runner_frontend_c_inputs_support.zig");
const t = std.testing;
const Loc = struct {
    offset: usize,
    tokLen: usize = 1,
    expansionLoc: ?*const Loc = null,
    spellingLoc: ?*const Loc = null,
    isMacroArgExpansion: bool = false,
};
const Range = struct { begin: Loc, end: Loc };
const Node = struct {
    kind: []const u8,
    type: struct { qualType: []const u8 = "", desugaredQualType: []const u8 = "" } = .{},
    name: []const u8 = "",
    range: ?Range = null,
    loc: ?Loc = null,
    inner: []const Node = &.{},
    array_filler: []const Node = &.{},
    value: std.json.Value = .null,
    storageClass: []const u8 = "",
    castKind: []const u8 = "",
    argType: struct { qualType: []const u8 = "" } = .{},
};

// These fixtures exercise the AST schema and edits; real Clang/ELF validation belongs to the parent host suite.
const Fixture = struct {
    a: std.mem.Allocator,
    source: []const u8,
    offset: usize = 0,

    fn at(f: Fixture, text: []const u8) usize {
        return std.mem.indexOf(u8, f.source, text).?;
    }
    fn range(f: Fixture, lo: usize, hi: usize) Range {
        return .{ .begin = .{ .offset = f.offset + lo }, .end = .{ .offset = f.offset + hi - 1 } };
    }
    fn node(f: Fixture, kind: []const u8, text: []const u8, typ: []const u8) Node {
        const lo = f.at(text);
        return .{ .kind = kind, .type = .{ .qualType = typ }, .range = f.range(lo, lo + text.len) };
    }
    fn lastNode(f: Fixture, kind: []const u8, text: []const u8, typ: []const u8) Node {
        const lo = std.mem.lastIndexOf(u8, f.source, text).?;
        return .{ .kind = kind, .type = .{ .qualType = typ }, .range = f.range(lo, lo + text.len) };
    }
    fn list(f: Fixture, nodes: []const Node) ![]const Node {
        return f.a.dupe(Node, nodes);
    }
    fn readNode(f: Fixture, child: Node, typ: []const u8) !Node {
        return .{ .kind = "ImplicitCastExpr", .type = .{ .qualType = typ }, .range = child.range, .castKind = "LValueToRValue", .inner = try f.list(&.{child}) };
    }
    fn variable(f: Fixture, text: []const u8, name: []const u8, typ: []const u8, init: ?Node) !Node {
        var n = f.node("VarDecl", text, typ);
        n.name = name;
        var name_index: usize = 0;
        while (name_index < text.len) : (name_index += 1) {
            if (!std.mem.startsWith(u8, text[name_index..], name)) continue;
            if (name_index != 0 and (std.ascii.isAlphanumeric(text[name_index - 1]) or text[name_index - 1] == '_')) continue;
            const end = name_index + name.len;
            if (end < text.len and (std.ascii.isAlphanumeric(text[end]) or text[end] == '_')) continue;
            break;
        }
        const name_at = f.at(text) + name_index;
        n.loc = .{ .offset = f.offset + name_at, .tokLen = name.len };
        if (init) |child| n.inner = try f.list(&.{child});
        if (std.mem.startsWith(u8, text, "static")) n.storageClass = "static";
        return n;
    }
    fn ast(f: Fixture, globals: []const Node, body: []const Node) ![]const u8 {
        var function = f.node("FunctionDecl", f.source[f.at("int main")..], "int (void)");
        function.name = "main";
        const open = std.mem.indexOfScalarPos(u8, f.source, f.at("int main"), '{').?;
        function.inner = try f.list(&.{.{ .kind = "CompoundStmt", .inner = body, .range = f.range(open, f.source.len) }});
        var top: std.ArrayList(Node) = .empty;
        try top.appendSlice(f.a, globals);
        try top.append(f.a, function);
        return std.json.Stringify.valueAlloc(f.a, Node{ .kind = "TranslationUnitDecl", .inner = top.items }, .{});
    }
    fn lower(f: Fixture, globals: []const Node, body: []const Node) !?[]u8 {
        return inputs.lower(t.allocator, f.source, try f.ast(globals, body), f.offset);
    }
    fn lowerCost(f: Fixture, globals: []const Node, body: []const Node) !?inputs.Lowered {
        return inputs.lowerWithCost(t.allocator, f.source, try f.ast(globals, body), f.offset);
    }
};

fn contains(s: []const u8, text: []const u8) !void {
    try t.expect(std.mem.indexOf(u8, s, text) != null);
}

// The parent supplies real normalized source and its matching Clang JSON without storing private fixture paths here.
pub fn expectClangFixture(source: []const u8, ast_json: []const u8, source_offset: usize) !void {
    const output = (try inputs.lower(t.allocator, source, ast_json, source_offset)) orelse
        return error.UnsupportedClangFixture;
    defer t.allocator.free(output);
    try contains(output, "#define main __tfimr_user_main\n");
    try contains(output, "#undef main\nint main(void)");
    try contains(output, "return __tfimr_user_main();");
    const repeated = (try inputs.lower(t.allocator, source, ast_json, source_offset)) orelse
        return error.UnsupportedClangFixture;
    defer t.allocator.free(repeated);
    try t.expectEqualStrings(output, repeated);
}

test "literal leaves retain signedness and width and source arithmetic" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ return 21U + 'a' + 9LL; }", .offset = 173 };
    var character = f.node("CharacterLiteral", "'a'", "int");
    character.value = .{ .integer = 97 };
    const body = [_]Node{
        f.node("IntegerLiteral", "21U", "unsigned int"), character,
        f.node("IntegerLiteral", "9LL", "long long"),
    };
    const output = (try f.lower(&.{}, &body)).?;
    defer t.allocator.free(output);
    try contains(output, "volatile unsigned int __tfimr_v_0");
    try contains(output, "= 21U;");
    try contains(output, "volatile int __tfimr_v_1");
    try contains(output, "= 'a';");
    try contains(output, "volatile long long __tfimr_v_2");
    try contains(output, "return (__tfimr_v_0) + (__tfimr_v_1) + (__tfimr_v_2);");
    const again = (try f.lower(&.{}, &body)).?;
    defer t.allocator.free(again);
    try t.expectEqualStrings(output, again);
}

test "automatic numeric array leaves remain inside the initializer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ int a[3]={4,5}; return a[1]; }" };
    var init = f.node("InitListExpr", "{4,5}", "int[3]");
    const four = f.node("IntegerLiteral", "4", "int");
    init.inner = try f.list(&.{four});
    init.array_filler = try f.list(&.{ .{ .kind = "ImplicitValueInitExpr" }, four, f.node("IntegerLiteral", "5", "int") });
    const variable = try f.variable("int a[3]={4,5}", "a", "int[3]", init);
    const output = (try f.lower(&.{}, &.{ variable, f.node("IntegerLiteral", "1", "int") })).?;
    defer t.allocator.free(output);
    try contains(output, "int a[3]={(__tfimr_v_0),(__tfimr_v_1)};");
    try contains(output, "a[(__tfimr_v_2)]");
}

test "global string and integer arrays preserve inferred lengths and runtime initialization" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static const char A[]=\"abc\"; static int a[]={10+2,3}; int main(void){ return a[0]; }" };
    const string_init = f.node("StringLiteral", "\"abc\"", "const char[4]");
    const letters = try f.variable("static const char A[]=\"abc\"", "A", "const char[4]", string_init);
    var numbers_init = f.node("InitListExpr", "{10+2,3}", "int[2]");
    numbers_init.inner = try f.list(&.{
        f.node("IntegerLiteral", "10", "int"), f.node("IntegerLiteral", "2", "int"), f.node("IntegerLiteral", "3", "int"),
    });
    const numbers = try f.variable("static int a[]={10+2,3}", "a", "int[2]", numbers_init);
    const output = (try f.lower(&.{ letters, numbers }, &.{f.lastNode("IntegerLiteral", "0", "int")})).?;
    defer t.allocator.free(output);
    try contains(output, "static  char A[]=\"abc\"; static int a[2]={0};");
    try contains(output, "__typeof__(a) __tfimr_t_0 = {(__tfimr_v_1)+(__tfimr_v_2),(__tfimr_v_3)};");
    try contains(output, "__asm__ volatile(\"\" : : \"r\"(&A) : \"memory\");");
    try contains(output, "__asm__ volatile(\"\" : : \"r\"(&a) : \"memory\");");
    try t.expect(std.mem.indexOf(u8, output, "__tfimr_b_") == null);
    try contains(output, "return __tfimr_user_main();");
}

test "grouped static locals initialize once after the complete declaration" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ static const int a=7,b=8; return a+b; }" };
    const a = try f.variable("static const int a=7", "a", "const int", f.node("IntegerLiteral", "7", "int"));
    var b = try f.variable("static const int a=7,b=8", "b", "const int", f.node("IntegerLiteral", "8", "int"));
    b.storageClass = "static";
    const output = (try f.lower(&.{}, &.{ a, b })).?;
    defer t.allocator.free(output);
    try contains(output, "static  int a=0,b=0; static int __tfimr_g_0;");
    try contains(output, "if (!__tfimr_g_0)");
    try contains(output, "__typeof__(a) __tfimr_t_0 = (__tfimr_v_1)");
    try contains(output, "if (!__tfimr_g_2)");
    try contains(output, "__typeof__(b) __tfimr_t_2 = (__tfimr_v_3)");
}

test "automatic string array keeps shape while decayed strings use opaque addresses" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ char s[6]=\"x\"; puts(\"ok\"); return s[0]; }" };
    const variable = try f.variable("char s[6]=\"x\"", "s", "char[6]", f.node("StringLiteral", "\"x\"", "char[2]"));
    var decay: Node = .{ .kind = "ImplicitCastExpr", .castKind = "ArrayToPointerDecay" };
    decay.inner = try f.list(&.{f.node("StringLiteral", "\"ok\"", "char[3]")});
    const output = (try f.lower(&.{}, &.{ variable, decay, f.node("IntegerLiteral", "0", "int") })).?;
    defer t.allocator.free(output);
    try contains(output, "volatile char __tfimr_b_0[6]");
    try contains(output, "char s[6]={0}; __tfimr_copy");
    try contains(output, "char * volatile __tfimr_p_1");
    try contains(output, "puts((__tfimr_p_1))");
}

test "sizeof local metadata becomes a typed opaque size slot" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ int a[4]; return sizeof(a)+2; }" };
    var size = f.node("UnaryExprOrTypeTraitExpr", "sizeof(a)", "unsigned int");
    size.name = "sizeof";
    size.inner = try f.list(&.{f.node("DeclRefExpr", "a)", "int[4]")});
    const output = (try f.lower(&.{}, &.{ size, f.node("IntegerLiteral", "2", "int") })).?;
    defer t.allocator.free(output);
    try contains(output, "volatile unsigned int __tfimr_v_0");
    try contains(output, "= sizeof(int[4]);");
    try contains(output, "return (__tfimr_v_0)+(__tfimr_v_1);");
}

test "case labels stay compile time and implicit scalar globals use opaque zero" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int x; int main(void){ switch(x){ case 4: return 8; } }" };
    const x = try f.variable("int x", "x", "int", null);
    var case: Node = .{ .kind = "CaseStmt" };
    case.inner = try f.list(&.{ f.node("ConstantExpr", "4", "int"), f.node("IntegerLiteral", "8", "int") });
    const output = (try f.lower(&.{x}, &.{case})).?;
    defer t.allocator.free(output);
    try contains(output, "case 4: return (__tfimr_v_1)");
    try contains(output, "__typeof__(x) __tfimr_t_0 = (__tfimr_zero)");
}

test "macro literals use expansion position and refuse ambiguous expansion trees" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "#define N 6\nint main(void){ return N; }", .offset = 32 };
    const expansion: Loc = .{ .offset = f.offset + f.at("N;"), .tokLen = 1 };
    var literal = f.node("IntegerLiteral", "6", "int");
    literal.range = .{
        .begin = .{ .offset = 1, .expansionLoc = &expansion },
        .end = .{ .offset = 1, .expansionLoc = &expansion },
    };
    literal.value = .{ .string = "6" };
    const output = (try f.lower(&.{}, &.{literal})).?;
    defer t.allocator.free(output);
    try t.expect(std.mem.startsWith(u8, output, "#define N 6\n"));
    try contains(output, "= N;");
    try contains(output, "#define N 6\n");
    var other = literal;
    other.value = .{ .string = "7" };
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{ literal, other }));
}

test "unsafe contexts malformed JSON offsets and reserved identifiers abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const source = "int main(void){ return 6; }";
    const f: Fixture = .{ .a = arena.allocator(), .source = source };
    for ([_][]const u8{ "EnumDecl", "RecordDecl", "GenericSelectionExpr", "ConstantExpr", "FloatingLiteral", "AlignedAttr" }) |kind| {
        try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{f.node(kind, "6", "int")}));
    }
    var null_pointer: Node = .{ .kind = "ImplicitCastExpr", .castKind = "NullToPointer" };
    null_pointer.inner = try f.list(&.{f.node("IntegerLiteral", "6", "int")});
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{null_pointer}));
    var invalid = f.node("IntegerLiteral", "6", "int");
    invalid.range.?.end.offset = source.len + 1;
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{invalid}));
    try t.expectEqual(@as(?[]u8, null), try inputs.lower(t.allocator, source, "{", 0));
    try t.expectEqual(@as(?[]u8, null), try inputs.lower(t.allocator, "int __tfimr_x;", "{}", 0));
    const vla: Fixture = .{ .a = f.a, .source = "int main(void){ int n=6; int a[n]; }" };
    const variable = try vla.variable("int a[n]", "a", "int[n]", null);
    try t.expectEqual(@as(?[]u8, null), try vla.lower(&.{}, &.{variable}));
}

test "bounded input and JSON nesting refuse before transformation" {
    const source: [16 * 1024 + 1]u8 = @splat(' ');
    try t.expectEqual(@as(?[]u8, null), try inputs.lower(t.allocator, &source, "{}", 0));
    const open: [65]u8 = @splat('[');
    const close: [65]u8 = @splat(']');
    const deep = open ++ close;
    try t.expectEqual(@as(?[]u8, null), try inputs.lower(t.allocator, "", &deep, 0));
    const json = try t.allocator.alloc(u8, 4 * 1024 * 1024 + 1);
    defer t.allocator.free(json);
    @memset(json, ' ');
    try t.expectEqual(@as(?[]u8, null), try inputs.lower(t.allocator, "", json, 0));
}

test "global arithmetic runs on volatile leaves and implicit zero follows declaration order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static int x=2*3,y; int main(void){ return x+y; }" };
    var math = f.node("BinaryOperator", "2*3", "int");
    math.inner = try f.list(&.{ f.node("IntegerLiteral", "2", "int"), f.node("IntegerLiteral", "3", "int") });
    const x = try f.variable("static int x=2*3", "x", "int", math);
    const y = try f.variable("static int x=2*3,y", "y", "int", null);
    const output = (try f.lower(&.{ x, y }, &.{})).?;
    defer t.allocator.free(output);
    try contains(output, "static int x=0,y;");
    try contains(output, "__typeof__(x) __tfimr_t_0 = (__tfimr_v_1)*(__tfimr_v_2)");
    try contains(output, "__typeof__(y) __tfimr_t_3 = (__tfimr_zero)");
}

test "prelude size_t aliases are validated through Clang desugared builtin types" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ size_t n=4; }", .offset = 3014 };
    var variable = try f.variable("size_t n=4", "n", "size_t", f.node("IntegerLiteral", "4", "int"));
    variable.type.desugaredQualType = "unsigned int";
    const prelude: Node = .{ .kind = "RecordDecl", .range = .{ .begin = .{ .offset = 0 }, .end = .{ .offset = 10 } } };
    const output = (try f.lower(&.{prelude}, &.{variable})).?;
    defer t.allocator.free(output);
    try contains(output, "size_t n=(__tfimr_v_0);");
    try contains(output, "return __tfimr_zero;");
}

test "automatic string restore cannot be delayed across another grouped initializer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ char a[]=\"x\",b=a[0]; }" };
    const variable = try f.variable("char a[]=\"x\"", "a", "char[2]", f.node("StringLiteral", "\"x\"", "char[2]"));
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{variable}));
}

test "function-like macros abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "#define N(x) x\nint main(void){ return 3; }" };
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{f.node("IntegerLiteral", "3", "int")}));
}

test "zero array barriers preserve pointer aliases and remove const without adding volatile" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static const int z[4]; int main(void){ const int *p=z; return *p; }" };
    const zeros = try f.variable("static const int z[4]", "z", "const int[4]", null);
    const pointer = try f.variable("const int *p=z", "p", "const int *", f.lastNode("DeclRefExpr", "z", "const int[4]"));
    const lowered = (try f.lowerCost(&.{zeros}, &.{pointer})).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32), lowered.restoration_steps);
    try contains(lowered.source, "static  int z[4];");
    try contains(lowered.source, "const int *p=z; return *p;");
    try contains(lowered.source, "int main(void) {  __asm__ volatile(\"\" : : \"r\"(&z) : \"memory\");");
    try t.expect(std.mem.indexOf(u8, lowered.source, "volatile static") == null);
    try t.expect(std.mem.indexOf(u8, lowered.source, "volatile int z") == null);
}

test "automatic partial arrays are mutable and fenced immediately after initialization" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ const int a[4]={7}; const int *p=a; return *p; }" };
    var init = f.node("InitListExpr", "{7}", "const int[4]");
    init.inner = try f.list(&.{f.node("IntegerLiteral", "7", "int")});
    const array = try f.variable("const int a[4]={7}", "a", "const int[4]", init);
    const lowered = (try f.lowerCost(&.{}, &.{array})).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32), lowered.restoration_steps);
    try contains(lowered.source, " int a[4]={(__tfimr_v_0)}; __asm__ volatile(\"\" : : \"r\"(&a) : \"memory\");");
    try contains(lowered.source, "const int *p=a;");
}

test "pointer array elements retain their original pointee qualifiers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ int n; int *p[2]={&n,&n}; return p[0]==p[1]; }" };
    var init = f.node("InitListExpr", "{&n,&n}", "int *[2]");
    var address = f.node("UnaryOperator", "&n", "int *");
    const ref_at = f.at("&n") + 1;
    address.inner = try f.list(&.{.{ .kind = "DeclRefExpr", .type = .{ .qualType = "int" }, .range = f.range(ref_at, ref_at + 1) }});
    var second = f.lastNode("UnaryOperator", "&n", "int *");
    const second_ref = std.mem.lastIndexOf(u8, f.source, "&n").? + 1;
    second.inner = try f.list(&.{.{ .kind = "DeclRefExpr", .type = .{ .qualType = "int" }, .range = f.range(second_ref, second_ref + 1) }});
    init.inner = try f.list(&.{ address, second });
    const array = try f.variable("int *p[2]={&n,&n}", "p", "int *[2]", init);
    const output = (try f.lower(&.{}, &.{array})).?;
    defer t.allocator.free(output);
    try contains(output, "int *p[2]={&n,&n}; __asm__ volatile(\"\" : : \"r\"(&p) : \"memory\");");
    try t.expect(std.mem.indexOf(u8, output, "volatile int *p") == null);
    try t.expect(std.mem.indexOf(u8, output, "volatile int * p") == null);
}

test "global zero pointer arrays and grouped integer arrays use barriers without restoration" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static int *p[2]; static const int a[2],b[3]; int main(void){ }" };
    const pointers = try f.variable("static int *p[2]", "p", "int *[2]", null);
    const first = try f.variable("static const int a[2]", "a", "const int[2]", null);
    const second = try f.variable("static const int a[2],b[3]", "b", "const int[3]", null);
    const lowered = (try f.lowerCost(&.{ pointers, first, second }, &.{})).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32), lowered.restoration_steps);
    try contains(lowered.source, "static int *p[2]; static  int a[2],b[3];");
    for ([_][]const u8{ "p", "a", "b" }) |name| {
        const barrier = try std.fmt.allocPrint(f.a, "\"r\"(&{s}) : \"memory\"", .{name});
        try contains(lowered.source, barrier);
    }
}

test "large global and static local strings retain input initializers with constant startup cost" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static const char g[4096]=\"x\"; int main(void){ static const char s[4096]=\"y\"; }" };
    const global = try f.variable("static const char g[4096]=\"x\"", "g", "const char[4096]", f.node("StringLiteral", "\"x\"", "const char[2]"));
    const local = try f.variable("static const char s[4096]=\"y\"", "s", "const char[4096]", f.node("StringLiteral", "\"y\"", "const char[2]"));
    const lowered = (try f.lowerCost(&.{global}, &.{local})).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32), lowered.restoration_steps);
    try contains(lowered.source, "static  char g[4096]=\"x\";");
    try contains(lowered.source, "static  char s[4096]=\"y\"; __asm__ volatile");
    try t.expect(std.mem.indexOf(u8, lowered.source, "__tfimr_b_") == null);
    try t.expect(std.mem.indexOf(u8, lowered.source, "__tfimr_g_") == null);
    try t.expect(std.mem.indexOf(u8, lowered.source, "__tfimr_copy((") == null);
}

test "restoration cost accounts for exact global multidimensional and scalar bytes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static int a[2][3]={{1},{2}}; static unsigned long long n=9ULL; int main(void){ }" };
    var init = f.node("InitListExpr", "{{1},{2}}", "int[2][3]");
    init.inner = try f.list(&.{ f.node("IntegerLiteral", "1", "int"), f.lastNode("IntegerLiteral", "2", "int") });
    const array = try f.variable("static int a[2][3]={{1},{2}}", "a", "int[2][3]", init);
    const scalar = try f.variable("static unsigned long long n=9ULL", "n", "unsigned long long", f.node("IntegerLiteral", "9ULL", "unsigned long long"));
    const lowered = (try f.lowerCost(&.{ array, scalar }, &.{})).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32 + (24 * 8 + 32) + (8 * 8 + 32)), lowered.restoration_steps);
    try contains(lowered.source, "__typeof__(a)");
    try contains(lowered.source, "__typeof__(n)");
    const plain = (try f.lower(&.{ array, scalar }, &.{})).?;
    defer t.allocator.free(plain);
    try t.expectEqualStrings(lowered.source, plain);
}

test "local copies are excluded from global restoration metadata" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ char s[6]=\"x\"; static int n=2*3; }" };
    const array = try f.variable("char s[6]=\"x\"", "s", "char[6]", f.node("StringLiteral", "\"x\"", "char[2]"));
    var math = f.node("BinaryOperator", "2*3", "int");
    math.inner = try f.list(&.{ f.node("IntegerLiteral", "2", "int"), f.node("IntegerLiteral", "3", "int") });
    const scalar = try f.variable("static int n=2*3", "n", "int", math);
    const lowered = (try f.lowerCost(&.{}, &.{ array, scalar })).?;
    defer t.allocator.free(lowered.source);
    try t.expectEqual(@as(usize, 32), lowered.restoration_steps);
    try contains(lowered.source, "__tfimr_copy((volatile unsigned char *)&s");
    try contains(lowered.source, "\"r\"(&s) : \"memory\"");
    try contains(lowered.source, "if (!__tfimr_g_");
}

test "grouped automatic arrays and for initializer arrays abstain before delayed barriers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ int n=1,a[2]={7}; }" };
    const scalar = try f.variable("int n=1", "n", "int", f.node("IntegerLiteral", "1", "int"));
    var init = f.node("InitListExpr", "{7}", "int[2]");
    init.inner = try f.list(&.{f.node("IntegerLiteral", "7", "int")});
    const array = try f.variable("int n=1,a[2]={7}", "a", "int[2]", init);
    const decl: Node = .{ .kind = "DeclStmt", .inner = try f.list(&.{ scalar, array }) };
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{}, &.{decl}));
    const g: Fixture = .{ .a = f.a, .source = "int main(void){ for(int a[2]={7};;) break; }" };
    var loop_init = g.node("InitListExpr", "{7}", "int[2]");
    loop_init.inner = try g.list(&.{g.node("IntegerLiteral", "7", "int")});
    const loop_array = try g.variable("int a[2]={7}", "a", "int[2]", loop_init);
    const loop_decl: Node = .{ .kind = "DeclStmt", .inner = try g.list(&.{loop_array}) };
    const loop: Node = .{ .kind = "ForStmt", .inner = try g.list(&.{loop_decl}) };
    try t.expectEqual(@as(?[]u8, null), try g.lower(&.{}, &.{loop}));
}

test "RV32 shape sizing is exact and bounds every fixed dimension" {
    const cases = [_]struct { type: []const u8, bytes: usize }{
        .{ .type = "_Bool", .bytes = 1 },              .{ .type = "const unsigned char", .bytes = 1 },
        .{ .type = "volatile short[7]", .bytes = 14 }, .{ .type = "long", .bytes = 4 },
        .{ .type = "unsigned long long", .bytes = 8 }, .{ .type = "const int **[3]", .bytes = 12 },
        .{ .type = "int[2][3][4]", .bytes = 96 },      .{ .type = "char[32768]", .bytes = 32768 },
    };
    for (cases) |case| try t.expectEqual(@as(?usize, case.bytes), support.objectBytes(case.type));
    const bad = [_][]const u8{ "int[]", "int[n]", "char[0]", "char[32769]", "int[32768]", "long long[99][99]", "int[18446744073709551616]", "struct S", "int (*)[2]" };
    for (bad) |typ| try t.expectEqual(@as(?usize, null), support.objectBytes(typ));
}

test "oversized objects and cumulative global restoration bytes abstain" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static char a[32769]; int main(void){ }" };
    const big = try f.variable("static char a[32769]", "a", "char[32769]", null);
    try t.expectEqual(@as(?[]u8, null), try f.lower(&.{big}, &.{}));
    const g: Fixture = .{ .a = f.a, .source = "static int a[5000]={1}; static int b[5000]={2}; int main(void){ }" };
    var first = g.node("InitListExpr", "{1}", "int[5000]");
    first.inner = try g.list(&.{g.node("IntegerLiteral", "1", "int")});
    var second = g.node("InitListExpr", "{2}", "int[5000]");
    second.inner = try g.list(&.{g.node("IntegerLiteral", "2", "int")});
    const a = try g.variable("static int a[5000]={1}", "a", "int[5000]", first);
    const b = try g.variable("static int b[5000]={2}", "b", "int[5000]", second);
    try t.expectEqual(@as(?[]u8, null), try g.lower(&.{ a, b }, &.{}));
}

test "fixed multidimensional array decay is permitted only in expression metadata" {
    try t.expect(support.expressionType("int (*)[3]"));
    try t.expect(support.expressionType("const char (*)[8]"));
    try t.expect(!support.expressionType("int (*)[n]"));
    try t.expect(!support.expressionType("float (*)[3]"));
    try t.expectEqual(@as(?usize, null), support.objectBytes("int (*)[3]"));
}

test "each scalar use has its own register barrier and duplicate AST ranges share edits" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_][]const u8{ "-", "==" }) |op| {
        const source = try std.fmt.allocPrint(arena.allocator(), "int main(void){{ int x=7; return x{s}x; }}", .{op});
        const f: Fixture = .{ .a = arena.allocator(), .source = source };
        const variable = try f.variable("int x=7", "x", "int", f.node("IntegerLiteral", "7", "int"));
        const pos = f.at("return x") + 7;
        const left = try f.readNode(.{ .kind = "DeclRefExpr", .type = .{ .qualType = "int" }, .range = f.range(pos, pos + 1) }, "int");
        const right = try f.readNode(f.lastNode("DeclRefExpr", "x", "int"), "int");
        const binary: Node = .{ .kind = "BinaryOperator", .inner = try f.list(&.{ left, right }) };
        const output = (try f.lower(&.{}, &.{ variable, binary, left })).?;
        defer t.allocator.free(output);
        try contains(output, "\"+r\"(__tfimr_r_1)");
        try contains(output, "\"+r\"(__tfimr_r_2)");
        try t.expectEqual(@as(usize, 2), std.mem.count(u8, output, "(unsigned int)(x)"));
        try t.expectEqual(@as(usize, 2), std.mem.count(u8, output, "(__typeof__(x))"));
        const join = try std.fmt.allocPrint(f.a, "; }}){s}({{ unsigned int", .{op});
        try contains(output, join);
    }
}

test "nested array reads consume child edits and cached children cannot overlap their owner" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "int main(void){ int a[3]={1}; int i=0; return a[i]; }" };
    var init = f.node("InitListExpr", "{1}", "int[3]");
    init.inner = try f.list(&.{f.node("IntegerLiteral", "1", "int")});
    const array = try f.variable("int a[3]={1}", "a", "int[3]", init);
    const index = try f.variable("int i=0", "i", "int", f.node("IntegerLiteral", "0", "int"));
    const inner = try f.readNode(f.lastNode("DeclRefExpr", "i", "int"), "int");
    var subscript = f.node("ArraySubscriptExpr", "a[i]", "int");
    subscript.inner = try f.list(&.{ .{ .kind = "DeclRefExpr", .type = .{ .qualType = "int[3]" }, .range = f.range(f.at("a[i]"), f.at("a[i]") + 1) }, inner });
    const outer = try f.readNode(subscript, "int");
    const output = (try f.lower(&.{}, &.{ array, index, outer, inner, outer })).?;
    defer t.allocator.free(output);
    const unique = (try f.lower(&.{}, &.{ array, index, outer })).?;
    defer t.allocator.free(unique);
    try t.expectEqualStrings(unique, output);
    try contains(output, "unsigned int __tfimr_r_3 = (unsigned int)(a[({ unsigned int __tfimr_r_2");
}

test "wide scalar reads use two RV32 register outputs without reevaluating the operand" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_][]const u8{ "long long", "unsigned long long" }) |typ| {
        const source = try std.fmt.allocPrint(arena.allocator(), "int main(void){{ {s} x=9LL; return x; }}", .{typ});
        const f: Fixture = .{ .a = arena.allocator(), .source = source };
        const text = try std.fmt.allocPrint(f.a, "{s} x=9LL", .{typ});
        const variable = try f.variable(text, "x", typ, f.node("IntegerLiteral", "9LL", "long long"));
        const read = try f.readNode(f.lastNode("DeclRefExpr", "x", typ), typ);
        const output = (try f.lower(&.{}, &.{ variable, read })).?;
        defer t.allocator.free(output);
        try contains(output, "\"+r\"(__tfimr_lo_1), \"+r\"(__tfimr_hi_1)");
        try contains(output, "(__typeof__(x))((unsigned long long)__tfimr_lo_1 | ((unsigned long long)__tfimr_hi_1 << 32))");
        try t.expect(std.mem.indexOf(u8, output, "\"+r\"(__tfimr_r_1)") == null);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, output, "(unsigned long long)(x)"));
    }
}

test "pointer read operands with side effects evaluate once and restore their original scalar type" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f: Fixture = .{ .a = arena.allocator(), .source = "static int a[2]; int main(void){ int *p=a; p; return *p++; }" };
    const array = try f.variable("static int a[2]", "a", "int[2]", null);
    const pointer = try f.variable("int *p=a", "p", "int *", f.lastNode("DeclRefExpr", "a", "int[2]"));
    const use_at = f.at("p;");
    const pointer_use = try f.readNode(.{ .kind = "DeclRefExpr", .type = .{ .qualType = "int *" }, .range = f.range(use_at, use_at + 1) }, "int *");
    var deref = f.node("UnaryOperator", "*p++", "int");
    var postfix = f.node("UnaryOperator", "p++", "int *");
    postfix.inner = try f.list(&.{f.lastNode("DeclRefExpr", "p", "int *")});
    deref.inner = try f.list(&.{postfix});
    const read = try f.readNode(deref, "int");
    const output = (try f.lower(&.{array}, &.{ pointer, pointer_use, read })).?;
    defer t.allocator.free(output);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, output, "(unsigned int)(*p++)"));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, output, "(__typeof__(*p++))"));
    try contains(output, "(unsigned int)(p)");
    try contains(output, "(__typeof__(p))__tfimr_r_0");
    try t.expectEqual(@as(?usize, 4), support.readBytes("int *"));
    try t.expectEqual(@as(?usize, 4), support.readBytes("int (*)[3]"));
    try t.expect(support.expressionType("int (*)[3]"));
    for ([_][]const u8{ "int[3]", "int (*)[n]", "volatile float", "struct X", "int (*)(void)" }) |typ|
        try t.expectEqual(@as(?usize, null), support.readBytes(typ));
}
