const std = @import("std");
const A = std.mem.Allocator;
const V = std.json.Value;
const support = @import("runner_frontend_c_inputs_support.zig");
const jsonDepth = support.jsonDepth;
const integerType = support.integerType;
const objectType = support.objectType;
const Error = A.Error || error{Unsupported};
const Span = struct { lo: usize, hi: usize };
const Edit = struct { span: Span, text: []const u8 };

// Source is allocator-owned; cost includes global copies and wrapper startup, excluding all local initialization.
pub const Lowered = struct { source: []u8, restoration_steps: usize };

pub fn lower(a: A, source: []const u8, ast_json: []const u8, source_offset: usize) A.Error!?[]u8 {
    const lowered = try lowerWithCost(a, source, ast_json, source_offset) orelse return null;
    return lowered.source;
}

// The caller supplies the original, successfully typechecked Clang AST for this exact source.
pub fn lowerWithCost(a: A, source: []const u8, ast_json: []const u8, source_offset: usize) A.Error!?Lowered {
    if (source.len > 16 * 1024 or ast_json.len > 4 * 1024 * 1024) return null;
    if (!jsonDepth(ast_json) or std.mem.indexOf(u8, source, "__tfimr_") != null) return null;
    if (std.mem.indexOf(u8, source, "\\\n") != null or std.mem.indexOf(u8, source, "\\\r\n") != null) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const parsed = std.json.parseFromSlice(V, scratch, ast_json, .{ .ignore_unknown_fields = true }) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return null;
    };
    defer parsed.deinit();
    var p: Pass = .{ .a = scratch, .source = source, .offset = source_offset };
    const result = p.run(parsed.value) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return null;
    };
    if (result.len > 256 * 1024) return null;
    return .{ .source = try a.dupe(u8, result), .restoration_steps = p.restoration_steps };
}

fn field(v: V, key: []const u8) V {
    return if (v == .object) v.object.get(key) orelse .null else .null;
}
fn string(v: V) []const u8 {
    return if (v == .string) v.string else "";
}
fn eq(v: V, s: []const u8) bool {
    return std.mem.eql(u8, string(v), s);
}
fn items(v: V) []const V {
    return if (v == .array) v.array.items else &.{};
}
fn number(v: V) Error!usize {
    if (v != .integer or v.integer < 0) return error.Unsupported;
    return std.math.cast(usize, v.integer) orelse error.Unsupported;
}
fn ty(v: V) []const u8 {
    return string(field(field(v, "type"), "qualType"));
}
fn typeName(v: V) []const u8 {
    const desugared = string(field(v, "desugaredQualType"));
    return if (desugared.len != 0) desugared else string(field(v, "qualType"));
}
fn canonical(v: V) []const u8 {
    return typeName(field(v, "type"));
}
fn narrowString(v: V) bool {
    return std.mem.startsWith(u8, ty(v), "char[") or std.mem.startsWith(u8, ty(v), "const char[");
}
fn identifier(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}
fn fixedMetadata(node: V, depth: usize) bool {
    if (depth > 64) return false;
    const typ = canonical(node);
    if (std.mem.indexOfScalar(u8, typ, '[') != null and !objectType(typ)) return false;
    const arg = typeName(field(node, "argType"));
    if (arg.len != 0 and !objectType(arg)) return false;
    for (items(field(node, "inner"))) |child| if (!fixedMetadata(child, depth + 1)) return false;
    return true;
}
fn constantContext(node: V, depth: usize) bool {
    if (depth > 64) return false;
    const kind = field(node, "kind");
    if (eq(kind, "UnaryExprOrTypeTraitExpr")) return fixedMetadata(node, depth + 1);
    const allowed = [_][]const u8{
        "ConstantExpr",   "IntegerLiteral",      "CharacterLiteral", "ParenExpr",      "UnaryOperator",
        "BinaryOperator", "ConditionalOperator", "ImplicitCastExpr", "CStyleCastExpr",
    };
    var supported = false;
    for (allowed) |name| {
        if (eq(kind, name)) supported = true;
    }
    if (!supported) return false;
    for (items(field(node, "inner"))) |child| if (!constantContext(child, depth + 1)) return false;
    return true;
}

const Pass = struct {
    a: A,
    source: []const u8,
    offset: usize,
    edits: std.ArrayList(Edit) = .empty,
    data: std.ArrayList(u8) = .empty,
    startup: std.ArrayList(u8) = .empty,
    restoration_steps: usize = 32,
    restoration_bytes: usize = 0,
    next: usize = 0,
    main_found: bool = false,
    literals: std.ArrayList(struct { span: Span, key: []const u8, replacement: []const u8 }) = .empty,
    reads: std.ArrayList(struct { span: Span, typ: []const u8, replacement: []const u8 }) = .empty,

    fn fmt(p: *Pass, comptime format: []const u8, args: anytype) A.Error![]const u8 {
        return std.fmt.allocPrint(p.a, format, args);
    }
    fn append(p: *Pass, list: *std.ArrayList(u8), s: []const u8) A.Error!void {
        try list.appendSlice(p.a, s);
    }
    fn id(p: *Pass) Error!usize {
        if (p.next >= 4096) return error.Unsupported;
        const n = p.next;
        p.next += 1;
        return n;
    }
    fn location(p: *Pass, raw: V) Error!struct { pos: usize, len: usize, macro: bool } {
        const expansion = field(raw, "expansionLoc");
        const loc = if (expansion == .null) raw else expansion;
        if (field(loc, "isMacroArgExpansion") == .bool and field(loc, "isMacroArgExpansion").bool)
            return error.Unsupported;
        const off = try number(field(loc, "offset"));
        const len = try number(field(loc, "tokLen"));
        if (off < p.offset) return error.Unsupported;
        const pos = off - p.offset;
        if (pos > p.source.len or len > p.source.len - pos or len == 0) return error.Unsupported;
        return .{ .pos = pos, .len = len, .macro = expansion != .null };
    }
    fn span(p: *Pass, node: V) Error!Span {
        const range = field(node, "range");
        const begin = try p.location(field(range, "begin"));
        const end = try p.location(field(range, "end"));
        if (begin.pos > end.pos or begin.macro != end.macro) return error.Unsupported;
        if (begin.macro and begin.pos != end.pos) return error.Unsupported;
        const s: Span = .{ .lo = begin.pos, .hi = end.pos + end.len };
        if (begin.macro) {
            for (p.source[s.lo..s.hi]) |c| if (!identifier(c)) return error.Unsupported;
            var after = s.hi;
            while (after < p.source.len and std.ascii.isWhitespace(p.source[after])) : (after += 1) {}
            if (after < p.source.len and p.source[after] == '(') return error.Unsupported;
        }
        return s;
    }
    fn edit(p: *Pass, s: Span, text: []const u8) Error!void {
        for (p.edits.items) |*old| {
            if (s.lo != old.span.lo or s.hi != old.span.hi) continue;
            if (std.mem.eql(u8, text, old.text)) return;
            if (s.lo != s.hi) return error.Unsupported;
            old.text = try p.fmt("{s}{s}", .{ old.text, text });
            return;
        }
        if (p.edits.items.len >= 4096) return error.Unsupported;
        try p.edits.append(p.a, .{ .span = s, .text = text });
    }
    fn less(_: void, x: Edit, y: Edit) bool {
        return x.span.lo < y.span.lo or (x.span.lo == y.span.lo and x.span.hi < y.span.hi);
    }
    fn render(p: *Pass, s: Span, edits: []Edit) Error![]const u8 {
        std.mem.sort(Edit, edits, {}, less);
        var result: std.ArrayList(u8) = .empty;
        var pos = s.lo;
        for (edits) |e| {
            if (e.span.lo < pos or e.span.hi > s.hi) return error.Unsupported;
            try p.append(&result, p.source[pos..e.span.lo]);
            try p.append(&result, e.text);
            pos = e.span.hi;
        }
        try p.append(&result, p.source[pos..s.hi]);
        return result.items;
    }
    fn slot(p: *Pass, node: V) Error!void {
        const s = try p.span(node);
        const kind = field(node, "kind");
        const value = field(node, "value");
        const key = try p.fmt("{s}|{s}|{s}|{d}", .{
            string(kind), ty(node), string(value), if (value == .integer) value.integer else @as(i64, 0),
        });
        for (p.literals.items) |old| {
            if (old.span.lo != s.lo or old.span.hi != s.hi) continue;
            if (!std.mem.eql(u8, key, old.key)) return error.Unsupported;
            return p.edit(s, old.replacement);
        }
        const n = try p.id();
        var token = p.source[s.lo..s.hi];
        if (eq(kind, "UnaryExprOrTypeTraitExpr")) {
            var operand = typeName(field(node, "argType"));
            const children = items(field(node, "inner"));
            if (operand.len == 0 and children.len == 1) operand = canonical(children[0]);
            if (!objectType(operand)) return error.Unsupported;
            token = try p.fmt("{s}({s})", .{
                if (eq(field(node, "name"), "sizeof")) @as([]const u8, "sizeof") else "_Alignof", operand,
            });
        }
        var replacement: []const u8 = undefined;
        if (eq(kind, "StringLiteral")) {
            if (!narrowString(node)) return error.Unsupported;
            try p.append(&p.data, try p.fmt(
                "volatile char __tfimr_s_{d}[] __attribute__((section(\".data\"))) = {s};\n" ++
                    "char * volatile __tfimr_p_{d} = (char *)__tfimr_s_{d};\n",
                .{ n, token, n, n },
            ));
            replacement = try p.fmt("(__tfimr_p_{d})", .{n});
        } else {
            if (!integerType(ty(node))) return error.Unsupported;
            try p.append(&p.data, try p.fmt(
                "volatile {s} __tfimr_v_{d} __attribute__((section(\".data\"))) = {s};\n",
                .{ ty(node), n, token },
            ));
            replacement = try p.fmt("(__tfimr_v_{d})", .{n});
        }
        try p.literals.append(p.a, .{ .span = s, .key = key, .replacement = replacement });
        try p.edit(s, replacement);
    }
    fn scalarRead(p: *Pass, node: V, local: bool, no_insert: bool, depth: usize) Error!void {
        const typ = canonical(node);
        const bytes = support.readBytes(typ) orelse return error.Unsupported;
        const children = items(field(node, "inner"));
        if (children.len != 1) return error.Unsupported;
        const region = try p.span(node);
        const child_region = try p.span(children[0]);
        if (region.lo != child_region.lo or region.hi != child_region.hi) return error.Unsupported;
        for (p.reads.items) |old| {
            if (old.span.lo != region.lo or old.span.hi != region.hi) continue;
            if (!std.mem.eql(u8, typ, old.typ)) return error.Unsupported;
            for (p.edits.items) |e| {
                if (e.span.lo > region.lo or e.span.hi < region.hi) continue;
                for (p.reads.items) |owner| {
                    if (owner.span.lo == e.span.lo and owner.span.hi == e.span.hi and
                        std.mem.eql(u8, owner.replacement, e.text)) return;
                }
            }
            return p.edit(region, old.replacement);
        }
        const first = p.edits.items.len;
        try p.walk(children[0], local, no_insert, depth + 1);
        const expr = try p.render(region, p.edits.items[first..]);
        p.edits.shrinkRetainingCapacity(first);
        const replacement = try support.readExpression(p.a, expr, p.source[region.lo..region.hi], bytes, try p.id());
        try p.reads.append(p.a, .{ .span = region, .typ = typ, .replacement = replacement });
        try p.edit(region, replacement);
    }
    fn groupEnd(p: *Pass, from: usize) Error!usize {
        var i = from;
        var depth: usize = 0;
        while (i < p.source.len) : (i += 1) {
            const c = p.source[i];
            if (c == '"' or c == '\'') {
                const quote = c;
                i += 1;
                while (i < p.source.len and p.source[i] != quote) : (i += 1) {
                    if (p.source[i] == '\\') i += 1;
                }
                if (i >= p.source.len) return error.Unsupported;
            } else if (c == '/' and i + 1 < p.source.len and p.source[i + 1] == '*') {
                const end = std.mem.indexOf(u8, p.source[i + 2 ..], "*/") orelse return error.Unsupported;
                i += end + 3;
            } else if (c == '/' and i + 1 < p.source.len and p.source[i + 1] == '/') {
                while (i < p.source.len and p.source[i] != '\n') : (i += 1) {}
                if (i == p.source.len) return error.Unsupported;
            } else if (c == '(' or c == '[' or c == '{') {
                depth += 1;
            } else if (c == ')' or c == ']' or c == '}') {
                if (depth == 0) return error.Unsupported;
                depth -= 1;
            } else if (c == ';' and depth == 0) return i + 1;
        }
        return error.Unsupported;
    }
    fn mutable(p: *Pass, decl: Span, name: usize) Error!void {
        var i = decl.lo;
        while (i < name) {
            if (try p.trivia(&i)) continue;
            const c = p.source[i];
            if (c == '=' or c == ',' or c == '[') break;
            if (identifier(c)) {
                const start = i;
                while (i < name and identifier(p.source[i])) : (i += 1) {}
                if (std.mem.eql(u8, p.source[start..i], "const")) try p.edit(.{ .lo = start, .hi = i }, "");
            } else {
                i += 1;
            }
        }
    }
    fn fixArray(p: *Pass, node: V, name_end: usize, init_start: usize) Error!void {
        var i = name_end;
        while (i < init_start) : (i += 1) {
            if (try p.trivia(&i)) {
                if (i >= init_start) break;
            }
            if (p.source[i] == '\'' or p.source[i] == '"') return error.Unsupported;
            if (p.source[i] != '[') continue;
            var end = i + 1;
            while (end < init_start and try p.trivia(&end)) {}
            if (end >= init_start or p.source[end] != ']') continue;
            const typ = canonical(node);
            const begin = std.mem.indexOfScalar(u8, typ, '[') orelse return error.Unsupported;
            const close = std.mem.indexOfScalarPos(u8, typ, begin, ']') orelse return error.Unsupported;
            try p.edit(.{ .lo = i, .hi = end + 1 }, typ[begin .. close + 1]);
            return;
        }
    }
    fn barrier(p: *Pass, name: []const u8) A.Error![]const u8 {
        return p.fmt(" __asm__ volatile(\"\" : : \"r\"(&{s}) : \"memory\"); ", .{name});
    }
    fn variable(p: *Pass, node: V, local: bool, no_insert: bool, depth: usize) Error!void {
        const typ = canonical(node);
        const bytes = support.objectBytes(typ) orelse return error.Unsupported;
        const children = items(field(node, "inner"));
        if (children.len > 1 or field(node, "previousDecl") != .null) return error.Unsupported;
        const storage = field(node, "storageClass");
        if (field(node, "tls") != .null or eq(storage, "extern")) return error.Unsupported;
        const array = std.mem.indexOfScalar(u8, typ, '[') != null;
        const string_array = array and children.len == 1 and eq(field(children[0], "kind"), "StringLiteral");
        const persistent = !local or eq(storage, "static");
        if (persistent) {
            if (std.mem.indexOfScalar(u8, typ, '*')) |star| {
                if ((!array and children.len == 0) or std.mem.indexOf(u8, typ[star + 1 ..], "const") != null)
                    return error.Unsupported;
            }
        }
        if (!persistent and !array) {
            for (children) |child| try p.walk(child, true, no_insert, depth + 1);
            return;
        }
        if (no_insert) return error.Unsupported;
        const decl = try p.span(node);
        const loc = try p.location(field(node, "loc"));
        const name = string(field(node, "name"));
        if (name.len == 0 or !std.mem.eql(u8, p.source[loc.pos .. loc.pos + loc.len], name))
            return error.Unsupported;
        const group = try p.groupEnd(decl.hi);
        if (local and !persistent) {
            var tail = decl.hi;
            while (tail < group and try p.trivia(&tail)) {}
            if (tail + 1 != group or p.source[tail] != ';') return error.Unsupported;
        }
        try p.mutable(decl, loc.pos);
        const fence = if (array) try p.barrier(name) else "";
        if (array and (children.len == 0 or (persistent and string_array))) {
            if (string_array and !narrowString(children[0])) return error.Unsupported;
            if (!local) try p.append(&p.startup, fence) else try p.edit(.{ .lo = group, .hi = group }, fence);
            return;
        }
        if (!persistent and !string_array) {
            for (children) |child| try p.walk(child, true, false, depth + 1);
            try p.edit(.{ .lo = group, .hi = group }, fence);
            return;
        }
        const n = try p.id();
        var init: []const u8 = "(__tfimr_zero)";
        if (children.len == 1) {
            const child = children[0];
            const init_span = try p.span(child);
            if (init_span.lo < loc.pos + loc.len or init_span.hi > decl.hi) return error.Unsupported;
            if (array) try p.fixArray(node, loc.pos + loc.len, init_span.lo);
            if (string_array) {
                if (!narrowString(child)) return error.Unsupported;
                const bracket = std.mem.indexOfScalar(u8, typ, '[') orelse return error.Unsupported;
                try p.append(&p.data, try p.fmt(
                    "volatile char __tfimr_b_{d}{s} __attribute__((section(\".data\"))) = {s};\n",
                    .{ n, typ[bracket..], p.source[init_span.lo..init_span.hi] },
                ));
            } else {
                const first = p.edits.items.len;
                try p.walk(child, true, false, depth + 1);
                init = try p.render(init_span, p.edits.items[first..]);
                p.edits.shrinkRetainingCapacity(first);
            }
            try p.edit(init_span, if (array) "{0}" else "0");
        }
        const setup = if (string_array)
            try p.fmt("__tfimr_copy((volatile unsigned char *)&{s}, (volatile unsigned char *)__tfimr_b_{d}, sizeof({s}));", .{ name, n, name })
        else
            try p.fmt("{{ __typeof__({s}) __tfimr_t_{d} = {s}; __tfimr_copy((volatile unsigned char *)&{s}, (volatile unsigned char *)&__tfimr_t_{d}, sizeof({s})); }}", .{ name, n, init, name, n, name });
        if (!local) {
            if (bytes > 32768 - p.restoration_bytes) return error.Unsupported;
            p.restoration_bytes += bytes;
            p.restoration_steps += bytes * 8 + 32;
            try p.append(&p.startup, setup);
            try p.append(&p.startup, fence);
        } else if (persistent) {
            try p.edit(.{ .lo = group, .hi = group }, try p.fmt(
                " static int __tfimr_g_{d}; if (!__tfimr_g_{d}) {{ {s} __tfimr_g_{d} = 1; }} {s}",
                .{ n, n, setup, n, fence },
            ));
        } else {
            try p.edit(.{ .lo = group, .hi = group }, try p.fmt(" {s}{s}", .{ setup, fence }));
        }
    }
    fn walk(p: *Pass, node: V, local: bool, no_insert: bool, depth: usize) Error!void {
        if (depth > 64 or node != .object) return error.Unsupported;
        const kind = field(node, "kind");
        const range = field(node, "range");
        if (!local and !eq(kind, "TranslationUnitDecl")) {
            const raw = field(range, "end");
            const loc = if (field(raw, "expansionLoc") == .null) raw else field(raw, "expansionLoc");
            const end = field(loc, "offset");
            if (end == .integer and end.integer >= 0) {
                const position = std.math.cast(usize, end.integer) orelse return error.Unsupported;
                if (position < p.offset) return;
            }
            if (end == .null and field(node, "isImplicit") == .bool and field(node, "isImplicit").bool) return;
            _ = try p.span(node);
        }
        if (std.mem.indexOfScalar(u8, canonical(node), '[') != null and !support.expressionType(canonical(node))) return error.Unsupported;
        if (eq(kind, "VarDecl")) return p.variable(node, local, no_insert, depth);
        if (eq(kind, "IntegerLiteral") or eq(kind, "CharacterLiteral"))
            return p.slot(node);
        const children = items(field(node, "inner"));
        if (eq(kind, "DeclStmt") and children.len > 1) {
            for (children) |child| {
                if (eq(field(child, "kind"), "VarDecl") and
                    std.mem.indexOfScalar(u8, canonical(child), '[') != null and
                    !eq(field(child, "storageClass"), "static")) return error.Unsupported;
            }
        }
        if (eq(kind, "ImplicitCastExpr")) {
            if (eq(field(node, "castKind"), "LValueToRValue")) return p.scalarRead(node, local, no_insert, depth);
            if (eq(field(node, "castKind"), "NullToPointer")) return error.Unsupported;
            if (eq(field(node, "castKind"), "ArrayToPointerDecay") and children.len == 1) {
                var child = children[0];
                var nesting: usize = 0;
                while (eq(field(child, "kind"), "ParenExpr")) {
                    const inner = items(field(child, "inner"));
                    nesting += 1;
                    if (inner.len != 1 or nesting > 64) return error.Unsupported;
                    child = inner[0];
                }
                if (eq(field(child, "kind"), "StringLiteral")) return p.slot(child);
            }
        }
        if (eq(kind, "DeclRefExpr") and std.mem.startsWith(u8, string(field(field(node, "referencedDecl"), "name")), "__builtin_"))
            return error.Unsupported;
        if (eq(kind, "UnaryExprOrTypeTraitExpr")) {
            if (!eq(field(node, "name"), "sizeof") and !eq(field(node, "name"), "alignof"))
                return error.Unsupported;
            if (!fixedMetadata(node, 0)) return error.Unsupported;
            return p.slot(node);
        }
        if (eq(kind, "FunctionDecl")) {
            if (eq(field(node, "name"), "main")) {
                if (!std.mem.eql(u8, ty(node), "int (void)") and !std.mem.eql(u8, ty(node), "int ()"))
                    return error.Unsupported;
                for (children) |child| if (eq(field(child, "kind"), "CompoundStmt")) {
                    if (p.main_found) return error.Unsupported;
                    const body = try p.span(child);
                    if (p.source[body.hi - 1] != '}') return error.Unsupported;
                    try p.edit(.{ .lo = body.hi - 1, .hi = body.hi - 1 }, " return __tfimr_zero; ");
                    p.main_found = true;
                };
            }
            for (children) |child| try p.walk(child, true, false, depth + 1);
            return;
        }
        if (eq(kind, "ParmVarDecl") and !objectType(canonical(node))) return error.Unsupported;
        const allowed = [_][]const u8{
            "TranslationUnitDecl", "ParmVarDecl",  "CompoundStmt",          "DeclStmt",               "ImplicitCastExpr",
            "CStyleCastExpr",      "ParenExpr",    "BinaryOperator",        "CompoundAssignOperator", "UnaryOperator",
            "ReturnStmt",          "IfStmt",       "ForStmt",               "WhileStmt",              "DoStmt",
            "BreakStmt",           "ContinueStmt", "NullStmt",              "CallExpr",               "ArraySubscriptExpr",
            "ConditionalOperator", "InitListExpr", "ImplicitValueInitExpr", "DeclRefExpr",            "SwitchStmt",
            "CaseStmt",            "DefaultStmt",
        };
        var supported = false;
        for (allowed) |name| {
            if (eq(kind, name)) {
                supported = true;
                break;
            }
        }
        if (!supported) return error.Unsupported;
        if (eq(kind, "InitListExpr") and std.mem.indexOfScalar(u8, ty(node), '*') == null) {
            for (children) |child| if (eq(field(child, "kind"), "StringLiteral")) return error.Unsupported;
        }
        if (eq(kind, "CaseStmt") and children.len != 2) return error.Unsupported;
        if (eq(kind, "CaseStmt") and !constantContext(children[0], 0)) return error.Unsupported;
        for (children, 0..) |child, i| {
            if (eq(kind, "CaseStmt") and i == 0) continue;
            if (child == .object and child.object.count() == 0 and eq(kind, "ForStmt")) continue;
            try p.walk(child, local, no_insert or eq(kind, "SwitchStmt") or
                (eq(kind, "ForStmt") and i == 0), depth + 1);
        }
        for (items(field(node, "array_filler"))) |child| try p.walk(child, local, no_insert, depth + 1);
    }
    fn trivia(p: *Pass, pos: *usize) Error!bool {
        const i = pos.*;
        if (std.ascii.isWhitespace(p.source[i])) {
            pos.* += 1;
        } else if (p.source[i] == '/' and i + 1 < p.source.len and p.source[i + 1] == '*') {
            const end = std.mem.indexOf(u8, p.source[i + 2 ..], "*/") orelse return error.Unsupported;
            pos.* = i + end + 4;
        } else if (p.source[i] == '/' and i + 1 < p.source.len and p.source[i + 1] == '/') {
            pos.* = std.mem.indexOfScalarPos(u8, p.source, i, '\n') orelse p.source.len;
        } else return false;
        return true;
    }
    fn preprocessor(p: *Pass) Error![]const u8 {
        var macros: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        var prefix = true;
        while (i < p.source.len) {
            if (try p.trivia(&i)) continue;
            if (p.source[i] == '#') {
                if (!prefix) return error.Unsupported;
                const end = std.mem.indexOfScalarPos(u8, p.source, i, '\n') orelse p.source.len;
                var words = std.mem.tokenizeAny(u8, p.source[i + 1 .. end], " \t\r");
                if (!std.mem.eql(u8, words.next() orelse return error.Unsupported, "define")) return error.Unsupported;
                const name = words.next() orelse return error.Unsupported;
                if (name.len == 0 or std.ascii.isDigit(name[0]) or std.mem.eql(u8, name, "main")) return error.Unsupported;
                for (name) |c| if (!identifier(c)) return error.Unsupported;
                const reserved = [_][]const u8{
                    "volatile",      "char",    "unsigned", "signed", "short", "long",   "int",    "const", "_Bool",
                    "__attribute__", "section", "static",   "void",   "while", "sizeof", "return", "if",    "__typeof__",
                    "_Alignof",
                };
                for (reserved) |word| if (std.mem.eql(u8, name, word)) return error.Unsupported;
                const body = std.mem.trim(u8, words.rest(), " \t\r");
                if (body.len == 0) return error.Unsupported;
                if (body[0] == '"' or body[0] == '\'') {
                    var j: usize = 1;
                    while (j < body.len and body[j] != body[0]) : (j += 1) {
                        if (body[j] == '\\') j += 1;
                    }
                    if (j + 1 != body.len) return error.Unsupported;
                } else {
                    if (!std.ascii.isDigit(body[0])) return error.Unsupported;
                    for (body) |c| if (!std.ascii.isAlphanumeric(c)) return error.Unsupported;
                }
                try p.append(&macros, p.source[i..end]);
                try p.append(&macros, "\n");
                try p.edit(.{ .lo = i, .hi = end }, "");
                i = end;
            } else {
                prefix = false;
                if (p.source[i] == '"' or p.source[i] == '\'') {
                    const quote = p.source[i];
                    i += 1;
                    while (i < p.source.len and p.source[i] != quote) : (i += 1) {
                        if (p.source[i] == '\\') i += 1;
                    }
                    if (i >= p.source.len) return error.Unsupported;
                }
                i += 1;
            }
        }
        return macros.items;
    }
    fn run(p: *Pass, root: V) Error![]const u8 {
        if (!eq(field(root, "kind"), "TranslationUnitDecl")) return error.Unsupported;
        const macros = try p.preprocessor();
        try p.walk(root, false, false, 0);
        if (!p.main_found) return error.Unsupported;
        const body = try p.render(.{ .lo = 0, .hi = p.source.len }, p.edits.items);
        return p.fmt(
            "{s}volatile int __tfimr_zero __attribute__((section(\".data\"))) = 0;\n{s}" ++
                "static void __tfimr_copy(volatile unsigned char *__tfimr_d, volatile unsigned char *__tfimr_s, unsigned long __tfimr_n) {{ while (__tfimr_n--) *__tfimr_d++ = *__tfimr_s++; }}\n" ++
                "#define main __tfimr_user_main\n{s}\n#undef main\n" ++
                "int main(void) {{ {s} return __tfimr_user_main(); }}\n",
            .{ macros, p.data.items, body, p.startup.items },
        );
    }
};
