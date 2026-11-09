const std = @import("std");
const ast = @import("runner_frontend_python_ast.zig");
const Allocator = std.mem.Allocator;

const runtime =
    \\typedef struct { int n; int b; } py_value;
    \\typedef struct { py_value items[128]; volatile int len; } py_list;
    \\__attribute__((always_inline)) static inline py_value py_hold(py_value v) { __asm__ volatile("" : "+r"(v.n)); return v; }
    \\__attribute__((noreturn)) static void py_assert_fail(void) {
    \\ register int a0 __asm__("a0")=1; register int a7 __asm__("a7")=93;
    \\ __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory"); for(;;) {}
    \\}
    \\static py_value py_int(int n) { py_value r = {n,0}; return r; }
    \\static py_value py_bool(int n) { py_value r = {!!n,1}; return r; }
    \\static py_value py_add(py_value a,py_value b) { return py_int(a.n+b.n); }
    \\static py_value py_sub(py_value a,py_value b) { return py_int(a.n-b.n); }
    \\static py_value py_mul(py_value a,py_value b) { return py_int(a.n*b.n); }
    \\static py_value py_neg(py_value a) { return py_int(-a.n); }
    \\static py_value py_reverse_digits(py_value a) {
    \\ int n=a.n,r=0; while(n) { r=r*10+n%10; n/=10; } return py_int(r);
    \\}
    \\static py_value py_floor(py_value a,py_value b) {
    \\ int q=a.n/b.n,r=a.n%b.n; if(r && ((r<0)!=(b.n<0))) --q; return py_int(q);
    \\}
    \\static py_value py_modulo(py_value a,py_value b) {
    \\ int r=a.n%b.n; if(r && ((r<0)!=(b.n<0))) r+=b.n; return py_int(r);
    \\}
    \\static py_value py_eq(py_value a,py_value b) { return py_bool(a.n==b.n); }
    \\static py_value py_ne(py_value a,py_value b) { return py_bool(a.n!=b.n); }
    \\static py_value py_lt(py_value a,py_value b) { return py_bool(a.n<b.n); }
    \\static py_value py_le(py_value a,py_value b) { return py_bool(a.n<=b.n); }
    \\static py_value py_gt(py_value a,py_value b) { return py_bool(a.n>b.n); }
    \\static py_value py_ge(py_value a,py_value b) { return py_bool(a.n>=b.n); }
    \\static py_value py_gcd(py_value a,py_value b) {
    \\ unsigned int x=a.n<0?0u-(unsigned int)a.n:(unsigned int)a.n;
    \\ unsigned int y=b.n<0?0u-(unsigned int)b.n:(unsigned int)b.n;
    \\ while(y) { unsigned int r=x%y; x=y; y=r; } return py_int((int)x);
    \\}
    \\static py_value py_len(const py_list *a) { return py_int(a->len); }
    \\static py_value py_sum(const py_list *a) {
    \\ __asm__ volatile("" : : "r"(a) : "memory");
    \\ int n=0; for(int i=0;i<a->len;++i) n+=a->items[i].n; return py_int(n);
    \\}
    \\static py_value py_get(const py_list *a,py_value i) {
    \\ return a->items[i.n<0?a->len+i.n:i.n];
    \\}
    \\static void py_append(py_list *a,py_value v) { a->items[a->len++]=v; }
    \\static void py_out(int c) {
    \\ register int a0 __asm__("a0")=c; register int a7 __asm__("a7")=1;
    \\ __asm__ volatile("ecall" : "+r"(a0) : "r"(a7));
    \\}
    \\static void py_text(const volatile char *s,int n) { for(int i=0;i<n;++i) py_out((unsigned char)s[i]); }
    \\static void py_print(py_value a) { char d[10]; tf_py_print(a.n,a.b,d); }
;

pub fn emit(a: Allocator, p: *const ast.Program) Allocator.Error![]u8 {
    var e: Emitter = .{ .a = a, .p = p };
    errdefer e.out.deinit(a);
    defer e.literals.deinit(a);
    try e.put(runtime);
    try e.put("\n");
    try e.put("static volatile int py_inputs[]={");
    var inputs: usize = 0;
    for (p.nodes.items) |node| {
        if (node.tag == .integer or node.tag == .boolean) {
            try e.out.print(a, "{d},", .{node.value});
            inputs += 1;
        }
    }
    if (inputs == 0) try e.put("0");
    try e.put("};\n");
    for (p.nodes.items, 0..) |node, id| {
        if (node.tag == .string) {
            try e.out.print(a, "static volatile char py_text_input{d}[]=", .{id});
            try e.quoted(node.text);
            try e.put(";\n");
        }
    }
    for (p.globals.symbols.items, 0..) |s, i| {
        try e.out.print(a, "static {s} py_g{d};\n", .{ typeName(s.kind), i });
    }
    for (p.functions.items, 0..) |f, id| {
        try e.signature(f, id, false);
        try e.put(";\n");
    }
    for (p.functions.items, 0..) |f, id| {
        e.scope = &p.functions.items[id].scope;
        e.params = f.params;
        try e.signature(f, id, true);
        try e.put(" {\n");
        for (f.scope.symbols.items, 0..) |s, i| {
            try e.out.print(a, "{s} py_l{d};\n", .{ if (s.kind == .list and e.isParam(s.name)) "const py_list *" else typeName(s.kind), i });
        }
        for (f.params, 0..) |name, i| {
            if (f.scope.symbols.items[f.scope.find(name).?].kind == .list) {
                try e.out.print(a, "py_l{d}", .{f.scope.find(name).?});
            } else try e.variable(name);
            try e.out.print(a, "=py_p{d};\n", .{i});
        }
        try e.statements(f.body);
        try e.put("return py_int(0);\n}\n");
    }
    e.scope = null;
    e.params = &.{};
    try e.put("int main(void) {\n");
    try e.statements(p.body);
    try e.put("return 0;\n}\n");
    return e.out.toOwnedSlice(a);
}

fn typeName(kind: ast.Kind) []const u8 {
    return if (kind == .list) "py_list" else "py_value";
}

const Generator = struct { name: []const u8, id: usize };
const Emitter = struct {
    a: Allocator,
    p: *const ast.Program,
    out: std.ArrayList(u8) = .empty,
    scope: ?*const ast.Scope = null,
    params: []const []const u8 = &.{},
    next_id: usize = 0,
    generators: [8]Generator = undefined,
    generators_len: usize = 0,
    literals: std.ArrayList(usize) = .empty, // list literals passed by pointer, declared around their statement

    fn isParam(e: *const Emitter, name: []const u8) bool {
        for (e.params) |param| if (std.mem.eql(u8, param, name)) return true;
        return false;
    }

    fn put(e: *Emitter, s: []const u8) Allocator.Error!void {
        try e.out.appendSlice(e.a, s);
    }

    fn fresh(e: *Emitter) usize {
        const id = e.next_id;
        e.next_id += 1;
        return id;
    }

    fn signature(e: *Emitter, f: ast.Function, id: usize, names: bool) Allocator.Error!void {
        try e.out.print(e.a, "static py_value py_f{d}(", .{id});
        if (f.params.len == 0) try e.put("void");
        for (f.params, 0..) |param, i| {
            if (i != 0) try e.put(",");
            try e.put(if (f.scope.symbols.items[f.scope.find(param).?].kind == .list) "const py_list *" else "py_value");
            if (names) try e.out.print(e.a, " py_p{d}", .{i});
        }
        try e.put(")");
    }

    fn variable(e: *Emitter, name: []const u8) Allocator.Error!void {
        var i = e.generators_len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, e.generators[i].name, name)) {
                try e.out.print(e.a, "py_gen{d}", .{e.generators[i].id});
                return;
            }
        }
        if (e.scope) |scope| {
            if (scope.find(name)) |id| {
                if (scope.symbols.items[id].kind == .list and e.isParam(name))
                    try e.out.print(e.a, "(*py_l{d})", .{id})
                else
                    try e.out.print(e.a, "py_l{d}", .{id});
                return;
            }
        }
        try e.out.print(e.a, "py_g{d}", .{e.p.globals.find(name).?});
    }

    fn kind(e: *const Emitter, id: usize) ast.Kind {
        const n = e.p.nodes.items[id];
        if (n.tag == .list) return .list;
        if (n.tag != .name) return .scalar;
        for (e.generators[0..e.generators_len]) |g| {
            if (std.mem.eql(u8, g.name, n.text)) return .scalar;
        }
        if (e.scope) |scope| {
            if (scope.find(n.text)) |sym| return scope.symbols.items[sym].kind;
        }
        return e.p.globals.symbols.items[e.p.globals.find(n.text).?].kind;
    }

    fn truth(e: *Emitter, id: usize) Allocator.Error!void {
        if (e.kind(id) == .list) {
            try e.put("(");
            try e.listPointer(id);
            try e.put(")->len");
        } else {
            try e.put("(");
            try e.expr(id);
            try e.put(").n");
        }
    }

    fn listPointer(e: *Emitter, id: usize) Allocator.Error!void {
        const n = e.p.nodes.items[id];
        if (n.tag == .name) {
            try e.put("&");
            return e.variable(n.text);
        }
        std.debug.assert(n.tag == .list);
        // only len and the items are written: a compound literal would zero all 128 items, a memset per call
        const tmp = e.fresh();
        try e.literals.append(e.a, tmp);
        try e.out.print(e.a, "({{py_literal{d}.len={d};", .{ tmp, n.args.len });
        for (n.args, 0..) |item, i| {
            try e.out.print(e.a, "py_literal{d}.items[{d}]=", .{ tmp, i });
            try e.expr(item);
            try e.put(";");
        }
        try e.out.print(e.a, "&py_literal{d};}})", .{tmp});
    }

    fn listCall(e: *Emitter, helper: []const u8, list: usize, index: ?usize) Allocator.Error!void {
        const n = e.p.nodes.items[list];
        const tmp = e.fresh();
        if (n.tag == .list) {
            try e.out.print(e.a, "({{py_list py_literal{d};py_literal{d}.len={d};", .{ tmp, tmp, n.args.len });
            for (n.args, 0..) |item, i| {
                try e.out.print(e.a, "py_literal{d}.items[{d}]=", .{ tmp, i });
                try e.expr(item);
                try e.put(";");
            }
        }
        try e.out.print(e.a, "py_{s}(", .{helper});
        if (n.tag == .list) try e.out.print(e.a, "&py_literal{d}", .{tmp}) else try e.listPointer(list);
        if (index) |id| {
            try e.put(",");
            try e.expr(id);
        }
        try e.put(")");
        if (n.tag == .list) try e.put(";})");
    }

    fn expr(e: *Emitter, id: usize) Allocator.Error!void {
        try e.put("py_hold(");
        try e.exprBody(id);
        try e.put(")");
    }

    fn exprBody(e: *Emitter, id: usize) Allocator.Error!void {
        const n = e.p.nodes.items[id];
        switch (n.tag) {
            .integer, .boolean => {
                var slot: usize = 0;
                for (e.p.nodes.items[0..id]) |prior| {
                    if (prior.tag == .integer or prior.tag == .boolean) slot += 1;
                }
                try e.out.print(e.a, "py_{s}(py_inputs[{d}])", .{
                    if (n.tag == .integer) @as([]const u8, "int") else "bool", slot,
                });
            },
            .name => try e.variable(n.text),
            .neg, .reverse_digits => {
                try e.out.print(e.a, "py_{s}(", .{@tagName(n.tag)});
                try e.expr(n.lhs);
                try e.put(")");
            },
            .logical_not => {
                try e.put("py_bool(!(");
                try e.truth(n.lhs);
                try e.put("))");
            },
            .add, .sub, .mul, .floor, .modulo, .eq, .ne, .lt, .le, .gt, .ge => {
                try e.out.print(e.a, "py_{s}(", .{@tagName(n.tag)});
                try e.expr(n.lhs);
                try e.put(",");
                try e.expr(n.rhs);
                try e.put(")");
            },
            .logical_and, .logical_or => {
                const tmp = e.fresh();
                try e.out.print(e.a, "({{py_value py_tmp{d}=", .{tmp});
                try e.expr(n.lhs);
                try e.out.print(e.a, ";py_tmp{d}.n?", .{tmp});
                if (n.tag == .logical_and) try e.expr(n.rhs) else try e.out.print(e.a, "py_tmp{d}", .{tmp});
                try e.put(":");
                if (n.tag == .logical_or) try e.expr(n.rhs) else try e.out.print(e.a, "py_tmp{d}", .{tmp});
                try e.put(";})");
            },
            .index => try e.listCall("get", n.lhs, n.rhs),
            .call => {
                if (std.mem.eql(u8, n.text, "len") or std.mem.eql(u8, n.text, "sum")) {
                    try e.listCall(n.text, n.args[0], null);
                    return;
                } else {
                    if (std.mem.eql(u8, n.text, "gcd")) try e.put("py_gcd(") else try e.out.print(e.a, "py_f{d}(", .{e.p.functionId(n.text).?});
                    for (n.args, 0..) |arg, i| {
                        if (i != 0) try e.put(",");
                        if (e.kind(arg) == .list) try e.listPointer(arg) else try e.expr(arg);
                    }
                }
                try e.put(")");
            },
            .sum_filtered => try e.sumFiltered(n),
            .all_range => try e.allRange(n),
            .string, .format, .list => unreachable,
        }
    }

    fn sumFiltered(e: *Emitter, n: ast.Expr) Allocator.Error!void {
        const id = e.fresh();
        const seq = e.p.nodes.items[n.rhs];
        const is_range = seq.tag == .call and std.mem.eql(u8, seq.text, "range");
        try e.out.print(e.a, "({{py_value py_sum{d}=py_int(0);", .{id});
        if (is_range) {
            try e.rangeParts(seq, id);
            try e.rangeHead(id);
            try e.out.print(e.a, "py_value py_gen{d}=py_hold(py_int((int)py_i{d}));", .{ id, id });
        } else {
            try e.out.print(e.a, "const py_list *py_seq{d}=", .{id});
            try e.listPointer(n.rhs);
            try e.out.print(e.a, ";for(int py_i{d}=0;py_i{d}<py_seq{d}->len;++py_i{d}){{", .{ id, id, id, id });
            try e.out.print(e.a, "py_value py_gen{d}=py_hold(py_seq{d}->items[py_i{d}]);", .{ id, id, id });
        }
        e.generators[e.generators_len] = .{ .name = n.text, .id = id };
        e.generators_len += 1;
        if (n.filter) |filter| {
            try e.put("if(");
            try e.truth(filter);
            try e.put("){");
        }
        try e.out.print(e.a, "py_sum{d}=py_hold(py_add(py_sum{d},", .{ id, id });
        try e.expr(n.lhs);
        e.generators_len -= 1;
        try e.put("));");
        if (n.filter != null) try e.put("}");
        try e.out.print(e.a, "}}py_sum{d};}})", .{id});
    }

    fn rangeParts(e: *Emitter, n: ast.Expr, id: usize) Allocator.Error!void {
        try e.out.print(e.a, "long long py_start{d}=", .{id});
        if (n.args.len > 1) {
            try e.put("(");
            try e.expr(n.args[0]);
            try e.put(").n");
        } else try e.put("0");
        try e.out.print(e.a, ",py_stop{d}=(", .{id});
        try e.expr(n.args[if (n.args.len > 1) 1 else 0]);
        try e.out.print(e.a, ").n,py_step{d}=", .{id});
        if (n.args.len == 3) {
            try e.put("(");
            try e.expr(n.args[2]);
            try e.put(").n");
        } else try e.put("1");
        try e.put(";\n");
    }

    fn rangeHead(e: *Emitter, id: usize) Allocator.Error!void {
        try e.out.print(e.a, "for(long long py_i{d}=py_start{d};", .{ id, id });
        try e.out.print(e.a, "py_step{d}>0?py_i{d}<py_stop{d}:py_i{d}>py_stop{d};", .{ id, id, id, id, id });
        try e.out.print(e.a, "py_i{d}+=py_step{d}){{\n", .{ id, id });
    }

    fn allRange(e: *Emitter, n: ast.Expr) Allocator.Error!void {
        const id = e.fresh();
        try e.out.print(e.a, "({{int py_all{d}=1;", .{id});
        try e.rangeParts(e.p.nodes.items[n.rhs], id);
        try e.rangeHead(id);
        try e.out.print(e.a, "py_value py_gen{d}=py_int((int)py_i{d});\nif(!(", .{ id, id });
        e.generators[e.generators_len] = .{ .name = n.text, .id = id };
        e.generators_len += 1;
        try e.truth(n.lhs);
        e.generators_len -= 1;
        try e.out.print(e.a, ")){{py_all{d}=0;break;}}\n}}py_bool(py_all{d});}})", .{ id, id });
    }

    fn quoted(e: *Emitter, s: []const u8) Allocator.Error!void {
        try e.put("\"");
        for (s) |c| {
            try e.out.print(e.a, "\\{o:0>3}", .{c});
        }
        try e.put("\"");
    }

    fn printPart(e: *Emitter, id: usize) Allocator.Error!void {
        const n = e.p.nodes.items[id];
        if (n.tag == .string) {
            try e.out.print(e.a, "py_text(py_text_input{d},{d});\n", .{ id, n.text.len });
        } else if (n.tag == .format) {
            for (n.args) |arg| try e.printPart(arg);
        } else {
            try e.put("py_print(");
            try e.expr(id);
            try e.put(");\n");
        }
    }

    fn statements(e: *Emitter, body: []const ast.Stmt) Allocator.Error!void {
        for (body) |s| {
            const at = e.out.items.len;
            const mark = e.literals.items.len;
            try e.statement(s);
            // the statement's own list literals live in a block around it, so their pointers outlive every use
            if (e.literals.items.len == mark) continue;
            var decls: std.ArrayList(u8) = .empty;
            defer decls.deinit(e.a);
            try decls.append(e.a, '{');
            for (e.literals.items[mark..]) |tmp| try decls.print(e.a, "py_list py_literal{d};", .{tmp});
            try decls.append(e.a, '\n');
            try e.out.insertSlice(e.a, at, decls.items);
            try e.put("}\n");
            e.literals.shrinkRetainingCapacity(mark);
        }
    }

    fn statement(e: *Emitter, s: ast.Stmt) Allocator.Error!void {
        switch (s.tag) {
            .assign => {
                const n = e.p.nodes.items[s.expr];
                if (n.tag == .list) {
                    const id = e.fresh();
                    try e.put("{\n");
                    if (n.args.len != 0) {
                        try e.out.print(e.a, "py_value py_assign{d}[{d}];\n", .{ id, n.args.len });
                    }
                    for (n.args, 0..) |item, i| {
                        try e.out.print(e.a, "py_assign{d}[{d}]=", .{ id, i });
                        try e.expr(item);
                        try e.put(";\n");
                    }
                    try e.variable(s.name);
                    try e.put(".len=0;\n");
                    for (n.args, 0..) |_, i| {
                        try e.put("py_append(&");
                        try e.variable(s.name);
                        try e.out.print(e.a, ",py_assign{d}[{d}]);\n", .{ id, i });
                    }
                    try e.put("}\n");
                } else {
                    try e.variable(s.name);
                    try e.put("=");
                    try e.expr(s.expr);
                    try e.put(";\n");
                }
            },
            .append => {
                try e.put("py_append(&");
                try e.variable(s.name);
                try e.put(",");
                try e.expr(s.expr);
                try e.put(");\n");
            },
            .conditional => {
                try e.put("if(");
                try e.truth(s.expr);
                try e.put("){\n");
                try e.statements(s.body);
                if (s.otherwise.len != 0) {
                    try e.put("}else{\n");
                    try e.statements(s.otherwise);
                }
                try e.put("}\n");
            },
            .loop => {
                try e.put("while(");
                try e.truth(s.expr);
                try e.put("){\n");
                try e.statements(s.body);
                try e.put("}\n");
            },
            .each => try e.each(s),
            .ret => {
                try e.put("return ");
                try e.expr(s.expr);
                try e.put(";\n");
            },
            .print => {
                for (s.args, 0..) |arg, i| {
                    if (i != 0) try e.put("py_out(' ');\n");
                    try e.printPart(arg);
                }
                try e.put("py_out('\\n');\n");
            },
            .assertion => {
                try e.put("if(!(");
                try e.truth(s.expr);
                try e.put("))py_assert_fail();\n");
            },
            .raise => try e.put(if (e.scope == null) "return 1;\n" else "return py_int(0);\n"),
            .function, .math_import => {},
        }
    }

    fn each(e: *Emitter, s: ast.Stmt) Allocator.Error!void {
        const n = e.p.nodes.items[s.expr];
        const id = e.fresh();
        try e.put("{\n");
        if (n.tag == .call) {
            try e.rangeParts(n, id);
            try e.rangeHead(id);
            try e.variable(s.name);
            try e.out.print(e.a, "=py_int((int)py_i{d});\n", .{id});
        } else {
            try e.out.print(e.a, "struct {{ py_value items[{d}]; int len; }} py_seq{d};\npy_seq{d}.len={d};\n", .{
                @max(n.args.len, 1), id, id, n.args.len,
            });
            for (n.args, 0..) |item, i| {
                try e.out.print(e.a, "py_seq{d}.items[{d}]=", .{ id, i });
                try e.expr(item);
                try e.put(";\n");
            }
            try e.out.print(e.a, "for(int py_i{d}=0;py_i{d}<py_seq{d}.len;++py_i{d}){{\n", .{ id, id, id, id });
            try e.variable(s.name);
            try e.out.print(e.a, "=py_seq{d}.items[py_i{d}];\n", .{ id, id });
        }
        try e.statements(s.body);
        try e.put("}\n}\n");
    }
};
