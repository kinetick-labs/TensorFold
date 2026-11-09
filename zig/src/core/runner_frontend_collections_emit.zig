const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const Allocator = std.mem.Allocator;
const Error = Allocator.Error;
pub const builtins = [_][]const u8{ "len", "sum", "min", "max", "range", "list", "tuple", "set", "dict", "enumerate", "zip", "sorted", "str", "int", "bool", "abs", "any", "all", "map" };
pub const methods = [_][]const u8{ "append", "pop", "reverse", "get", "setdefault", "keys", "items", "values", "add", "remove", "discard", "find", "rfind", "join", "split", "copy" };
pub fn lookup(names: []const []const u8, name: []const u8) ?usize {
    for (names, 0..) |s, i| if (std.mem.eql(u8, s, name)) return i;
    return null;
}

pub fn emit(a: Allocator, p: *const ast.Program) Error![]u8 {
    var e: Emitter = .{ .a = a, .p = p };
    errdefer e.out.deinit(a);
    defer e.bindings.deinit(a);
    try e.put(@embedFile("runner_frontend_collections_runtime.c"));
    try e.put(@embedFile("runner_frontend_collections_support.c"));
    try e.put("\nstatic volatile int inputs[]={");
    var input_count: usize = 0;
    for (p.nodes.items) |n| if (n.tag == .integer or n.tag == .boolean) {
        try e.out.print(a, "{d},", .{n.value});
        input_count += 1;
    };
    if (input_count == 0) try e.put("0");
    try e.put("};\n");
    for (p.nodes.items, 0..) |n, id| if (n.tag == .string) {
        try e.out.print(a, "static volatile char text{d}[]=", .{id});
        try e.quoted(n.text);
        try e.out.print(a, ";\nstatic V literal{d};\n", .{id});
    };
    for (p.globals.names.items, 0..) |_, i| try e.out.print(a, "static V g{d}={{-1,0}};\n", .{i});
    for (p.functions.items, 0..) |f, i| {
        try e.out.print(a, "static unsigned defined{d};\n", .{i});
        try e.signature(f, i);
        try e.put(";\n");
    }
    for (p.functions.items, 0..) |f, i| {
        e.scope = &p.functions.items[i].scope;
        try e.signature(f, i);
        try e.put("{\n");
        for (f.scope.names.items, 0..) |_, j| try e.out.print(a, "V l{d}={{-1,0}};\n", .{j});
        for (f.params, 0..) |name, j| try e.out.print(a, "l{d}=p{d};\n", .{ f.scope.find(name).?, j });
        try e.statements(f.body);
        try e.put("return none();\n}\n");
    }
    e.scope = null;
    try e.put("int main(void){\n");
    for (p.nodes.items, 0..) |n, i| if (n.tag == .string) {
        try e.out.print(a, "literal{d}=text(text{d},{d});\n", .{ i, i, n.text.len });
    };
    if (p.globals.find("__name__")) |i| try e.out.print(a, "g{d}=text(\"__main__\",8);\n", .{i});
    try e.statements(p.body);
    try e.put("return 0;\n}\n");
    return e.out.toOwnedSlice(a);
}

const Binding = struct { name: []const u8, id: usize };
const Emitter = struct {
    a: Allocator,
    p: *const ast.Program,
    out: std.ArrayList(u8) = .empty,
    scope: ?*const ast.Scope = null,
    next: usize = 0,
    bindings: std.ArrayList(Binding) = .empty,

    fn put(e: *Emitter, s: []const u8) Error!void {
        try e.out.appendSlice(e.a, s);
    }
    fn fresh(e: *Emitter) usize {
        const i = e.next;
        e.next += 1;
        return i;
    }
    fn quoted(e: *Emitter, s: []const u8) Error!void {
        try e.put("\"");
        for (s) |c| try e.out.print(e.a, "\\{o:0>3}", .{c});
        try e.put("\"");
    }
    fn signature(e: *Emitter, f: ast.Function, i: usize) Error!void {
        try e.out.print(e.a, "static V f{d}(", .{i});
        if (f.params.len == 0) try e.put("void");
        for (f.params, 0..) |_, j| {
            if (j != 0) try e.put(",");
            try e.out.print(e.a, "V p{d}", .{j});
        }
        try e.put(")");
    }
    fn variable(e: *Emitter, name: []const u8) Error!void {
        var i = e.bindings.items.len;
        while (i != 0) {
            i -= 1;
            const b = e.bindings.items[i];
            if (std.mem.eql(u8, b.name, name)) {
                try e.out.print(e.a, "v{d}", .{b.id});
                return;
            }
        }
        if (e.scope) |s| if (s.find(name)) |j| {
            try e.out.print(e.a, "l{d}", .{j});
            return;
        };
        if (e.p.globals.find(name)) |j| {
            try e.out.print(e.a, "g{d}", .{j});
        } else try e.put("(V){-1,0}");
    }
    fn decl(e: *Emitter, i: usize) Error!void {
        try e.out.print(e.a, "V v{d}=", .{i});
    }
    fn expr(e: *Emitter, id: ast.Id) Error!usize {
        const n = e.p.nodes.items[id];
        const r = e.fresh();
        switch (n.tag) {
            .integer, .boolean => {
                try e.decl(r);
                var slot: usize = 0;
                for (e.p.nodes.items[0..id]) |prior| if (prior.tag == .integer or prior.tag == .boolean) {
                    slot += 1;
                };
                try e.out.print(e.a, "val({d},inputs[{d}]);\n", .{ @intFromBool(n.tag == .boolean), slot });
            },
            .none => {
                try e.decl(r);
                try e.put("none();\n");
            },
            .string => {
                try e.decl(r);
                try e.out.print(e.a, "py_hold(literal{d});\n", .{id});
            },
            .name => {
                try e.decl(r);
                if ((std.mem.eql(u8, n.text, "sum") or std.mem.eql(u8, n.text, "str")) and e.p.globals.find(n.text) == null and
                    (e.scope == null or e.scope.?.find(n.text) == null))
                {
                    try e.out.print(e.a, "val(9,{d});\n", .{if (std.mem.eql(u8, n.text, "sum")) @as(usize, 1) else 2});
                } else {
                    try e.put("checked(");
                    try e.variable(n.text);
                    try e.put(");\n");
                }
            },
            .list, .tuple, .dict, .set => {
                try e.decl(r);
                const tag: usize = switch (n.tag) {
                    .list => 4,
                    .tuple => 5,
                    .dict => 6,
                    else => 7,
                };
                try e.out.print(e.a, "fresh({d});\n", .{tag});
                var i: usize = 0;
                while (i < n.args.len) : (i += 1) {
                    const x = try e.expr(n.args[i]);
                    if (n.tag == .dict) {
                        i += 1;
                        const y = try e.expr(n.args[i]);
                        try e.out.print(e.a, "store(v{d},v{d},v{d});\n", .{ r, x, y });
                    } else try e.out.print(e.a, "{s}(v{d},v{d});\n", .{ if (n.tag == .set) "setadd" else "append", r, x });
                }
            },
            .logical_and, .logical_or => {
                const x = try e.expr(n.a);
                try e.decl(r);
                try e.out.print(e.a, "v{d};\nif({s}truth(v{d})){{\n", .{ x, if (n.tag == .logical_or) "!" else "", x });
                const y = try e.expr(n.b);
                try e.out.print(e.a, "v{d}=py_hold(v{d});\n}}\n", .{ r, y });
            },
            .conditional => {
                const cond = try e.expr(n.b);
                try e.out.print(e.a, "V v{d};if(truth(v{d})){{\n", .{ r, cond });
                const x = try e.expr(n.a);
                try e.out.print(e.a, "v{d}=py_hold(v{d});\n}}else{{\n", .{ r, x });
                const y = try e.expr(n.c.?);
                try e.out.print(e.a, "v{d}=py_hold(v{d});\n}}\n", .{ r, y });
            },
            .neg, .pos, .logical_not => {
                const x = try e.expr(n.a);
                try e.decl(r);
                const op: usize = if (n.tag == .neg) 0 else if (n.tag == .pos) 1 else 2;
                try e.out.print(e.a, "unary({d},v{d});\n", .{ op, x });
            },
            .index => {
                const x = try e.expr(n.a);
                const y = try e.expr(n.b);
                try e.decl(r);
                try e.out.print(e.a, "get(v{d},v{d});\n", .{ x, y });
            },
            .slice => {
                const x = try e.expr(n.a);
                var parts: [3]usize = undefined;
                for (&parts, 0..) |*part, i| part.* = try e.expr(n.args[i]);
                try e.decl(r);
                try e.out.print(e.a, "slice(v{d},v{d},v{d},v{d});\n", .{ x, parts[0], parts[1], parts[2] });
            },
            .call, .method => try e.call(n, r),
            .comprehension => try e.comprehension(n, r),
            else => {
                const x = try e.expr(n.a);
                const y = try e.expr(n.b);
                try e.decl(r);
                try e.out.print(e.a, "binary({d},v{d},v{d});\n", .{ binaryId(n.tag), x, y });
            },
        }
        return r;
    }
    fn call(e: *Emitter, n: ast.Node, r: usize) Error!void {
        if (n.tag == .call) if (e.p.functionId(n.text)) |f| {
            try e.out.print(e.a, "if(!defined{d})fail();\n", .{f});
        };
        if (n.tag == .call and n.args.len == 1 and
            (std.mem.eql(u8, n.text, "any") or std.mem.eql(u8, n.text, "all")))
        {
            const gen = e.p.nodes.items[n.args[0]];
            if (gen.tag == .comprehension and gen.value == 1) return e.generatorTruth(gen, r, std.mem.eql(u8, n.text, "all"));
        }
        const base = if (n.tag == .method) try e.expr(n.a) else 0;
        var args: std.ArrayList(usize) = .empty;
        defer args.deinit(e.a);
        for (n.args) |arg| try args.append(e.a, try e.expr(arg));
        try e.decl(r);
        if (n.tag == .call) if (e.p.functionId(n.text)) |f| {
            try e.out.print(e.a, "f{d}(", .{f});
            for (args.items, 0..) |x, i| {
                if (i != 0) try e.put(",");
                try e.out.print(e.a, "v{d}", .{x});
            }
            try e.put(");\n");
            return;
        };
        const id = lookup(if (n.tag == .method) &methods else &builtins, n.text);
        var valid_keywords = true;
        var positional: usize = 0;
        for (n.keywords) |key| {
            if (key.len == 0) {
                positional += 1;
            } else if (n.tag != .call or !std.mem.eql(u8, n.text, "sorted") or
                (!std.mem.eql(u8, key, "key") and !std.mem.eql(u8, key, "reverse"))) valid_keywords = false;
        }
        if (std.mem.eql(u8, n.text, "sorted") and positional > 1) valid_keywords = false;
        if (id == null or args.items.len > 3 or !valid_keywords) {
            try e.put("none();fail();\n");
            return;
        }
        try e.out.print(e.a, "{s}({d},", .{ if (n.tag == .method) "method" else "builtin", id.? });
        if (n.tag == .method) try e.out.print(e.a, "v{d},", .{base});
        try e.out.print(e.a, "{d},", .{args.items.len});
        var slots: [3]?usize = .{ null, null, null };
        if (n.tag == .call and id.? == 11) {
            var pos: usize = 0;
            for (args.items, 0..) |x, i| {
                const key = if (i < n.keywords.len) n.keywords[i] else "";
                if (std.mem.eql(u8, key, "key")) slots[1] = x else if (std.mem.eql(u8, key, "reverse")) slots[2] = x else {
                    if (pos < 3) slots[pos] = x;
                    pos += 1;
                }
            }
        } else for (args.items, 0..) |x, i| slots[i] = x;
        for (slots, 0..) |x, i| {
            if (i != 0) try e.put(",");
            if (x) |j| try e.out.print(e.a, "v{d}", .{j}) else try e.put("none()");
        }
        try e.put(");\n");
    }
    fn generatorTruth(e: *Emitter, n: ast.Node, r: usize, all: bool) Error!void {
        const seq = try e.expr(n.b);
        const loop = e.fresh();
        const item = e.fresh();
        try e.decl(r);
        try e.out.print(e.a, "boolean({d});\n", .{@intFromBool(all)});
        try e.iterHead(seq, loop, item);
        try e.bindings.append(e.a, .{ .name = n.text, .id = item });
        defer _ = e.bindings.pop();
        if (n.c) |filter| {
            const f = try e.expr(filter);
            try e.out.print(e.a, "if(!truth(v{d}))continue;\n", .{f});
        }
        const x = try e.expr(n.a);
        try e.out.print(e.a, "if({s}truth(v{d})){{v{d}=boolean({d});break;}}\n}}\n", .{ if (all) "!" else "", x, r, @intFromBool(!all) });
    }
    fn comprehension(e: *Emitter, n: ast.Node, r: usize) Error!void {
        const seq = try e.expr(n.b);
        const loop = e.fresh();
        const item = e.fresh();
        try e.decl(r);
        try e.put("fresh(4);\n");
        try e.iterHead(seq, loop, item);
        try e.bindings.append(e.a, .{ .name = n.text, .id = item });
        defer _ = e.bindings.pop();
        if (n.c) |filter| {
            const f = try e.expr(filter);
            try e.out.print(e.a, "if(!truth(v{d}))continue;\n", .{f});
        }
        const x = try e.expr(n.a);
        try e.out.print(e.a, "append(v{d},v{d});\n}}\n", .{ r, x });
    }
    fn iterHead(e: *Emitter, seq: usize, loop: usize, item: usize) Error!void {
        try e.out.print(e.a, "int i{d}=0,guard{d}=dict_size(v{d});V v{d};\nwhile(1){{\n", .{ loop, loop, seq, item });
        try e.out.print(e.a, "iter_guard(v{d},guard{d});if(!cg_iter(v{d},&i{d},&v{d}))break;\n", .{ seq, loop, seq, loop, item });
    }
    fn target(e: *Emitter, id: usize, value: usize) Error!void {
        const n = e.p.nodes.items[id];
        switch (n.tag) {
            .name => {
                try e.variable(n.text);
                try e.out.print(e.a, "=py_hold(v{d});\n", .{value});
            },
            .tuple, .list => {
                const cursor = e.fresh();
                try e.out.print(e.a, "int unpack{d}=0;\n", .{cursor});
                var values: std.ArrayList(usize) = .empty;
                defer values.deinit(e.a);
                for (n.args) |_| {
                    const x = e.fresh();
                    try values.append(e.a, x);
                    try e.out.print(e.a, "V v{d};if(!cg_iter(v{d},&unpack{d},&v{d}))fail();\n", .{ x, value, cursor, x });
                }
                const extra = e.fresh();
                try e.out.print(e.a, "V v{d};if(cg_iter(v{d},&unpack{d},&v{d}))fail();\n", .{ extra, value, cursor, extra });
                for (n.args, values.items) |arg, x| try e.target(arg, x);
            },
            .index => {
                const x = try e.expr(n.a);
                const y = try e.expr(n.b);
                try e.out.print(e.a, "put(v{d},v{d},v{d});\n", .{ x, y, value });
            },
            else => try e.put("fail();\n"),
        }
    }
    fn statements(e: *Emitter, body: []const ast.Stmt) Error!void {
        for (body) |s| switch (s.tag) {
            .assign => {
                const x = try e.expr(s.expr.?);
                for (s.targets) |t| try e.target(t, x);
            },
            .augassign => {
                const n = e.p.nodes.items[s.targets[0]];
                if (n.tag == .index) {
                    const a = try e.expr(n.a);
                    const b = try e.expr(n.b);
                    const old = e.fresh();
                    try e.decl(old);
                    try e.out.print(e.a, "get(v{d},v{d});\n", .{ a, b });
                    const x = try e.expr(s.expr.?);
                    try e.out.print(e.a, "put(v{d},v{d},inplace({d},v{d},v{d}));\n", .{ a, b, binaryId(s.op), old, x });
                } else {
                    const old = try e.expr(s.targets[0]);
                    const x = try e.expr(s.expr.?);
                    const r = e.fresh();
                    try e.decl(r);
                    try e.out.print(e.a, "inplace({d},v{d},v{d});\n", .{ binaryId(s.op), old, x });
                    try e.target(s.targets[0], r);
                }
            },
            .expression => _ = try e.expr(s.expr.?),
            .print => {
                var values: std.ArrayList(usize) = .empty;
                defer values.deinit(e.a);
                for (s.targets) |arg| try values.append(e.a, try e.expr(arg));
                if (s.expr) |arg| try values.append(e.a, try e.expr(arg));
                for (values.items, 0..) |x, i| {
                    if (i != 0) try e.put("outchar(32);\n");
                    try e.out.print(e.a, "repr(v{d},1);\n", .{x});
                }
                try e.put("outchar(10);\n");
            },
            .conditional => {
                const x = try e.expr(s.expr.?);
                try e.out.print(e.a, "if(truth(v{d})){{\n", .{x});
                try e.statements(s.body);
                try e.put("}else{\n");
                try e.statements(s.otherwise);
                try e.put("}\n");
            },
            .while_loop => {
                try e.put("while(1){\n");
                const x = try e.expr(s.expr.?);
                try e.out.print(e.a, "if(!truth(v{d}))break;\n", .{x});
                try e.statements(s.body);
                try e.put("}\n");
            },
            .for_loop => {
                const x = try e.expr(s.expr.?);
                const i = e.fresh();
                const item = e.fresh();
                try e.iterHead(x, i, item);
                try e.target(s.targets[0], item);
                try e.statements(s.body);
                try e.put("}\n");
            },
            .ret => {
                if (s.expr) |arg| {
                    const x = try e.expr(arg);
                    try e.out.print(e.a, "return py_hold(v{d});\n", .{x});
                } else try e.put("return none();\n");
            },
            .assertion => {
                const x = try e.expr(s.expr.?);
                try e.out.print(e.a, "if(!truth(v{d}))fail();\n", .{x});
            },
            .break_loop => try e.put("break;\n"),
            .continue_loop => try e.put("continue;\n"),
            .function => try e.out.print(e.a, "defined{d}=1;\n", .{s.function}),
            .pass => {},
        };
    }
};
fn binaryId(tag: ast.Tag) usize {
    return switch (tag) {
        .add => 0,
        .sub => 1,
        .mul => 2,
        .floor => 3,
        .modulo => 4,
        .eq => 5,
        .ne => 6,
        .lt => 7,
        .le => 8,
        .gt => 9,
        .ge => 10,
        .contains => 11,
        .not_contains => 12,
        .is_identity => 13,
        .is_not => 14,
        else => 99,
    };
}
