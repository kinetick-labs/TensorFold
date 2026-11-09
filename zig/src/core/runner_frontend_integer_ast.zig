const std = @import("std");
const list = @import("runner_frontend_integer_list.zig");

pub const Invalid = error{Unsupported};
pub const limit: i512 = std.math.pow(i512, 10, 62);
/// Elements all of a script's lists may hold together (each one a node, so the node budget binds first).
pub const max_items = 64;
/// Nodes a whole script may make (loops unroll into them); other lowerings keep `capacity`'s 64, the C runtime's.
pub const max_nodes = 512;
/// A name bound but not yet assigned: a function's local before its first assignment (UnboundLocalError).
pub const unbound = std.math.maxInt(usize);
pub const Op = enum(u8) { literal, neg, add, sub, mul, floor, mod, pow, powmod, digits, divmod, gcd, isqrt, factorial, range_product, digit_sum, eq, ne, lt, le, gt, ge };
pub const Node = struct {
    op: Op,
    l: usize = 0,
    r: usize = 0,
    k: usize = 0,
    value: i512 = 0,
    text: []const u8 = "",
    base: u8 = 10,
};
pub const Parser = struct {
    source: []const u8,
    pos: usize = 0,
    nodes: [max_nodes]Node = undefined,
    count: usize = 0,
    capacity: usize = 64,
    proved: [max_nodes]bool = @splat(false),
    names: [64][]const u8 = undefined,
    roots: [64]usize = undefined,
    binding_count: usize = 0,
    require_math_import: bool = false,
    math_module: bool = false,
    math_imports: u8 = 0,
    functions: [8]Function = undefined,
    function_count: usize = 0,
    // inside an inlined body: its parameters are bindings[frame..], the module's are bindings[0..globals]
    frame: ?Frame = null,
    active: u8 = 0, // functions being inlined, one bit each: a call to one of them is recursion
    check_only: bool = false, // a def's body at its definition: syntax only, names are not looked up
    // static int lists: parse-time values over element nodes; a name of kind list holds its index in lists
    items: [max_items]usize = undefined,
    item_count: usize = 0,
    lists: [16]list.List = undefined,
    list_count: usize = 0,
    kinds: [64]bool = undefined,
    module_count: usize = 0, // the module's names: bindings[0..module_count] outside any frame or comprehension
    allow_lists: bool = false, // the whole-script lowering's lists and min/max; every other parser keeps its grammar
    listed: bool = false, // a list or min/max parsed: the graph route's script only (calc and ELF keep their own)
    asserted: bool = false, // an `assert` proved: like a def or a list, the graph route's script only
    looped: bool = false, // a loop, a suite body or a digit reversal: the graph route's script only
    calls: u8 = 0, // nested calls being inlined: a suite's recursion stops at 24
    no_recurse: bool = false, // a branch tried only if it needs no recursion (a schedule's last level)
    recursions: u32 = 0, // recursive calls inlined so far: a branch that recursed is not a schedule's last level
    cut: bool = false, // a recursion no_recurse refused: the untaken branch is past the schedule, not unsupported

    /// `def name(params): return <expr>`, inlined at each call; `body` borrows the script.
    pub const Function = struct { name: []const u8, params: [4][]const u8 = undefined, arity: usize = 0, body: []const u8, suite: []const u8 = "" };
    const Frame = struct { start: usize, globals: usize };
    /// An int node, or a list's index in `lists`.
    pub const Operand = struct { index: usize, list: bool };
    pub const Mark = struct { pos: usize, count: usize, items: usize, lists: usize };

    // Names and literal slices borrow source storage; bind after parsing the RHS to preserve prior roots.
    pub fn bind(self: *Parser, name: []const u8, root: usize) Invalid!void {
        return self.bindAs(name, .{ .index = root, .list = false });
    }

    pub fn bindAs(self: *Parser, name: []const u8, v: Operand) Invalid!void {
        if (name.len == 0 or v.index >= (if (v.list) self.list_count else self.count) or self.function(name) != null)
            return error.Unsupported;
        for (self.names[0..self.binding_count], self.roots[0..self.binding_count], 0..) |bound_name, *bound_root, i| {
            if (std.mem.eql(u8, name, bound_name)) {
                bound_root.* = v.index;
                self.kinds[i] = v.list;
                return;
            }
        }
        if (self.binding_count == self.names.len) return error.Unsupported;
        self.names[self.binding_count] = name;
        self.roots[self.binding_count] = v.index;
        self.kinds[self.binding_count] = v.list;
        self.binding_count += 1;
        self.module_count = self.binding_count;
    }

    /// Where to drop what a parse added (a failed attempt, or nodes only a proof reads).
    pub fn mark(self: *const Parser) Mark {
        return .{ .pos = self.pos, .count = self.count, .items = self.item_count, .lists = self.list_count };
    }

    pub fn reset(self: *Parser, m: Mark) void {
        self.pos, self.count, self.item_count, self.list_count = .{ m.pos, m.count, m.items, m.lists };
    }

    /// A call argument or a whole right side: a list when one parses up to `,`, `)` or the end, else an int.
    pub fn operand(self: *Parser, depth: usize) Invalid!Operand {
        if (list.ahead(self)) {
            const m = self.mark();
            if (list.parse(self, depth)) |li| {
                self.space();
                if (self.pos == self.source.len or self.source[self.pos] == ',' or self.source[self.pos] == ')')
                    return .{ .index = li, .list = true };
            } else |_| {}
            self.reset(m);
        }
        return .{ .index = try self.expr(depth), .list = false };
    }

    /// A def as Python compiles it: the body must parse now (names resolve at each call), the name must be new.
    pub fn define(self: *Parser, f: Function) Invalid!void {
        if (self.function_count == self.functions.len or self.function(f.name) != null) return error.Unsupported;
        for (self.names[0..self.binding_count]) |held| if (std.mem.eql(u8, held, f.name)) return error.Unsupported;
        for (f.params[0..f.arity], 0..) |p, i| {
            if (std.mem.eql(u8, p, f.name)) return error.Unsupported;
            for (f.params[0..i]) |q| if (std.mem.eql(u8, p, q)) return error.Unsupported;
        }
        const saved = .{ self.source, self.pos, self.count, self.item_count, self.list_count };
        defer {
            self.source, self.pos, self.count, self.item_count, self.list_count = saved;
            self.check_only = false;
        }
        self.check_only = true;
        if (f.suite.len != 0) try @import("runner_frontend_integer_suite.zig").check(self, f.suite) else {
            self.source = f.body;
            self.pos = 0;
            _ = try self.expr(0);
            self.space();
            if (self.pos != self.source.len) return error.Unsupported;
        }
        self.functions[self.function_count] = f;
        self.function_count += 1;
    }

    fn function(self: *const Parser, name: []const u8) ?usize {
        for (self.functions[0..self.function_count], 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return i;
        return null;
    }

    /// A name's binding: a body's params, then module names (never a caller's); newest first, so targets shadow.
    pub fn slot(self: *const Parser, name: []const u8) ?usize {
        const visible: [2][2]usize = if (self.frame) |f| .{ .{ f.start, self.binding_count }, .{ 0, f.globals } } else .{ .{ 0, self.binding_count }, .{ 0, 0 } };
        for (visible) |range| {
            var i = range[1];
            while (i > range[0]) {
                i -= 1;
                if (std.mem.eql(u8, name, self.names[i])) return i;
            }
        }
        return null;
    }

    fn lookup(self: *const Parser, name: []const u8) ?usize {
        const s = self.slot(name) orelse return null;
        return if (self.kinds[s] or self.roots[s] == unbound) null else self.roots[s];
    }

    pub fn lookupList(self: *const Parser, name: []const u8) ?usize {
        const s = self.slot(name) orelse return null;
        return if (self.kinds[s]) self.roots[s] else null;
    }

    /// `f(args)` with f's body parsed over the argument nodes: pure, so inlining is exact; recursion refuses.
    fn call(self: *Parser, index: usize, depth: usize) Invalid!usize {
        const f = self.functions[index];
        const bit = @as(u8, 1) << @intCast(index);
        // only a suite recurses, within its schedule (runner_frontend_integer_suite.zig)
        if (self.active & bit != 0) {
            if (f.suite.len == 0 or self.calls >= 24) return error.Unsupported;
            if (self.no_recurse) {
                self.cut = true;
                return error.Unsupported;
            }
            self.recursions += 1;
        }
        var args: [4]Operand = undefined;
        var n: usize = 0;
        try self.need("(");
        if (!self.take(")")) while (true) {
            if (n == args.len) return error.Unsupported;
            args[n] = try self.operand(depth + 1);
            n += 1;
            if (self.take(")")) break;
            try self.need(",");
        };
        if (n != f.arity or self.binding_count + n > self.names.len) return error.Unsupported;
        const saved = .{ self.source, self.pos, self.binding_count, self.frame, self.active, self.calls };
        defer {
            self.source, self.pos, self.binding_count, self.frame, self.active, self.calls = saved;
        }
        // read the outer frame before writing: a struct literal assigned in place may alias self.frame
        const globals = if (self.frame) |outer| outer.globals else self.module_count;
        self.frame = .{ .start = self.binding_count, .globals = globals };
        for (f.params[0..n], args[0..n]) |p, arg| {
            self.names[self.binding_count] = p;
            self.roots[self.binding_count] = arg.index;
            self.kinds[self.binding_count] = arg.list;
            self.binding_count += 1;
        }
        self.active |= bit;
        self.calls += 1;
        if (f.suite.len != 0) return @import("runner_frontend_integer_suite.zig").call(self, f.suite);
        self.source = f.body;
        self.pos = 0;
        const root = try self.expr(depth + 1);
        self.space();
        if (self.pos != self.source.len) return error.Unsupported;
        return root;
    }

    pub fn parse(self: *Parser) Invalid!usize {
        self.space();
        if (self.pos == self.source.len or self.source[self.pos] == '+') return error.Unsupported;
        const root = try self.expr(0);
        self.space();
        if (self.pos != self.source.len) return error.Unsupported;
        _ = try self.prove(root, root);
        return root;
    }

    // Project both results from the original operands; proof values never become literal nodes.
    pub fn divisionPair(self: *Parser, root: usize) Invalid![2]usize {
        if (root >= self.count or self.nodes[root].op != .divmod or self.count > self.capacity - 2)
            return error.Unsupported;
        _ = try self.prove(root, root);
        const n = self.nodes[root];
        const q = try self.add(.{ .op = .floor, .l = n.l, .r = n.r });
        const r = try self.add(.{ .op = .mod, .l = n.l, .r = n.r });
        _ = try self.prove(q, q);
        _ = try self.prove(r, r);
        return .{ q, r };
    }

    pub fn space(self: *Parser) void {
        while (self.pos < self.source.len and (self.source[self.pos] == ' ' or self.source[self.pos] == '\t'))
            self.pos += 1;
    }

    pub fn take(self: *Parser, s: []const u8) bool {
        self.space();
        if (!std.mem.startsWith(u8, self.source[self.pos..], s)) return false;
        self.pos += s.len;
        return true;
    }

    pub fn need(self: *Parser, s: []const u8) Invalid!void {
        if (!self.take(s)) return error.Unsupported;
    }

    pub fn add(self: *Parser, node: Node) Invalid!usize {
        if (self.count == self.capacity) return error.Unsupported;
        self.nodes[self.count] = node;
        self.proved[self.count] = false; // a reset may reuse the slot
        self.count += 1;
        return self.count - 1;
    }

    fn comparison(self: *Parser) ?Op {
        const tokens = [_][]const u8{ "==", "!=", "<", "<=", ">", ">=" };
        const ops = [_]Op{ .eq, .ne, .lt, .le, .gt, .ge };
        for ([_]usize{ 0, 1, 3, 5, 2, 4 }) |i| {
            if (self.take(tokens[i])) return ops[i];
        }
        return null;
    }

    pub fn expr(self: *Parser, depth: usize) Invalid!usize {
        const left = try self.sum(depth);
        const op = self.comparison() orelse return left;
        const right = try self.sum(depth);
        if (self.comparison() != null) return error.Unsupported;
        return self.add(.{ .op = op, .l = left, .r = right });
    }

    fn sum(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 12) return error.Unsupported;
        var left = try self.term(depth);
        while (true) {
            const op: Op = if (self.take("+")) .add else if (self.take("-")) .sub else break;
            left = try self.add(.{ .op = op, .l = left, .r = try self.term(depth) });
        }
        return left;
    }

    fn term(self: *Parser, depth: usize) Invalid!usize {
        var left = try self.factor(depth);
        while (true) {
            const op: Op = if (self.take("//")) .floor else if (self.take("%")) .mod else if (self.take("*")) .mul else break;
            const right = try self.factor(depth);
            const n = self.nodes[left];
            if (op == .mod and n.op == .pow) {
                left = try self.add(.{ .op = .powmod, .l = n.l, .r = n.r, .k = right });
            } else if (op == .mod and n.op == .neg and self.nodes[n.l].op == .pow) {
                const p = self.nodes[n.l];
                const reduced = try self.add(.{ .op = .powmod, .l = p.l, .r = p.r, .k = right });
                const negative = try self.add(.{ .op = .neg, .l = reduced });
                left = try self.add(.{ .op = .mod, .l = negative, .r = right });
            } else {
                left = try self.add(.{ .op = op, .l = left, .r = right });
            }
        }
        return left;
    }

    pub fn boolean(self: *const Parser, index: usize) bool {
        return switch (self.nodes[index].op) {
            .eq, .ne, .lt, .le, .gt, .ge => true,
            else => false,
        };
    }

    fn factor(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 12) return error.Unsupported;
        if (self.take("+")) {
            const child = try self.factor(depth + 1);
            if (self.boolean(child)) return error.Unsupported;
            return child;
        }
        if (self.take("-")) return self.add(.{ .op = .neg, .l = try self.factor(depth + 1) });
        const left = try self.atom(depth);
        if (self.take("**")) return self.add(.{ .op = .pow, .l = left, .r = try self.factor(depth + 1) });
        return left;
    }

    fn atom(self: *Parser, depth: usize) Invalid!usize {
        if (self.take("(")) {
            const n = try self.expr(depth + 1);
            try self.need(")");
            return n;
        }
        if (list.ahead(self)) return list.index(self, try list.parse(self, depth), depth);
        if (try self.algorithm(depth)) |n| return n;
        if (self.word("sum")) {
            if (!self.allow_lists) return self.digitSum(depth);
            const m = self.mark();
            if (self.digitSum(depth)) |n| return n else |_| self.reset(m);
            return list.sum(self, depth);
        }
        if (self.allow_lists and self.word("min")) return list.extreme(self, depth, false);
        if (self.allow_lists and self.word("max")) return list.extreme(self, depth, true);
        if (self.word("divmod")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .divmod, .l = a, .r = b });
        }
        if (self.word("pow")) {
            try self.need("(");
            const a = try self.expr(depth + 1);
            try self.need(",");
            const b = try self.expr(depth + 1);
            if (!self.take(",")) return error.Unsupported;
            const m = try self.expr(depth + 1);
            try self.need(")");
            return self.add(.{ .op = .powmod, .l = a, .r = b, .k = m });
        }
        if (self.word("len")) {
            if (!self.allow_lists) return self.textLength(depth);
            const m = self.mark();
            if (self.textLength(depth)) |n| return n else |_| self.reset(m);
            return list.length(self, depth);
        }
        if (self.word("int")) return if (self.take("(str(")) @import("runner_frontend_integer_digits.zig").reversed(self, depth) else self.convert();
        self.space();
        if (self.pos < self.source.len and
            (std.ascii.isAlphabetic(self.source[self.pos]) or self.source[self.pos] == '_'))
        {
            const name = try self.identifier();
            self.space();
            const called = self.pos < self.source.len and self.source[self.pos] == '(';
            if (self.check_only) {
                // a def's body at its definition: any name, any call (defined later or not), resolved when it runs
                if (called) {
                    try self.need("(");
                    if (!self.take(")")) while (true) {
                        _ = try self.operand(depth + 1);
                        if (self.take(")")) break;
                        try self.need(",");
                    };
                } else if (self.take("[")) {
                    _ = try self.expr(depth + 1);
                    try self.need("]");
                }
                return self.add(.{ .op = .literal, .value = 1, .text = "1" });
            }
            if (called) return self.call(self.function(name) orelse return error.Unsupported, depth);
            return self.lookup(name) orelse error.Unsupported;
        }
        return self.literal();
    }

    fn algorithm(self: *Parser, depth: usize) Invalid!?usize {
        const start = self.pos;
        const qualified = self.word("math");
        if (qualified) try self.need(".");
        const op: Op = if (self.word("gcd")) .gcd else if (self.word("isqrt")) .isqrt else if (self.word("factorial")) .factorial else if (self.word("prod")) .range_product else {
            if (qualified) return error.Unsupported;
            return null;
        };
        if (self.require_math_import) {
            const mask: u8 = switch (op) {
                .gcd => 1,
                .isqrt => 2,
                .factorial => 4,
                .range_product => 8,
                else => unreachable,
            };
            if (!qualified and self.math_imports & mask == 0) {
                self.pos = start; // not imported: a def of that name, or nothing (the call refuses)
                return null;
            }
            if (qualified and !self.math_module) return error.Unsupported;
        }
        try self.need("(");
        if (op == .range_product) {
            try self.need("range");
            try self.need("(");
        }
        var a = try self.expr(depth + 1);
        var b: usize = 0;
        var k: usize = 0;
        if (op == .gcd) {
            try self.need(",");
            b = try self.expr(depth + 1);
        }
        if (op == .range_product) {
            if (self.take(",")) {
                b = try self.expr(depth + 1);
            } else {
                b = a;
                a = try self.add(.{ .op = .literal, .value = 0, .text = "0" });
            }
            k = if (self.take(",")) try self.expr(depth + 1) else try self.add(.{
                .op = .literal,
                .value = 1,
                .text = "1",
            });
            try self.need(")");
        }
        try self.need(")");
        return try self.add(.{ .op = op, .l = a, .r = b, .k = k });
    }

    pub fn word(self: *Parser, token: []const u8) bool {
        self.space();
        const rest = self.source[self.pos..];
        if (!std.mem.startsWith(u8, rest, token)) return false;
        if (rest.len > token.len and (std.ascii.isAlphanumeric(rest[token.len]) or rest[token.len] == '_'))
            return false;
        self.pos += token.len;
        return true;
    }

    pub fn identifier(self: *Parser) Invalid![]const u8 {
        self.space();
        const start = self.pos;
        while (self.pos < self.source.len and
            (std.ascii.isAlphanumeric(self.source[self.pos]) or self.source[self.pos] == '_')) self.pos += 1;
        if (start == self.pos or std.ascii.isDigit(self.source[start])) return error.Unsupported;
        const name = self.source[start..self.pos];
        if (std.mem.eql(u8, name, "int") or std.mem.eql(u8, name, "__debug__")) return error.Unsupported;
        const reserved = [_][]const u8{ "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield" };
        for (reserved) |token| if (std.mem.eql(u8, name, token)) return error.Unsupported;
        return name;
    }

    /// len(str(n)): n's digit count, its sign included.
    fn textLength(self: *Parser, depth: usize) Invalid!usize {
        try self.need("(");
        try self.need("str");
        try self.need("(");
        const n = try self.expr(depth + 1);
        try self.need(")");
        try self.need(")");
        return self.add(.{ .op = .digits, .l = n });
    }

    fn digitSum(self: *Parser, depth: usize) Invalid!usize {
        try self.need("(");
        const mapped = self.word("map");
        if (mapped) {
            try self.need("(");
            try self.need("int");
            try self.need(",");
        } else {
            try self.need("int");
            try self.need("(");
            const name = try self.identifier();
            try self.need(")");
            if (!self.word("for")) return error.Unsupported;
            const binding = try self.identifier();
            if (!std.mem.eql(u8, name, binding) or !self.word("in")) return error.Unsupported;
        }
        try self.need("str");
        try self.need("(");
        const value = try self.expr(depth + 1);
        try self.need(")");
        if (mapped) try self.need(")");
        try self.need(")");
        return self.add(.{ .op = .digit_sum, .l = value });
    }

    fn convert(self: *Parser) Invalid!usize {
        try self.need("(");
        self.space();
        if (self.pos == self.source.len) return error.Unsupported;
        const quote = self.source[self.pos];
        if (quote != '\'' and quote != '"') return error.Unsupported;
        self.pos += 1;
        const begin = self.pos;
        while (self.pos < self.source.len and self.source[self.pos] != quote) : (self.pos += 1) {}
        if (self.pos == self.source.len) return error.Unsupported;
        const text = std.mem.trim(u8, self.source[begin..self.pos], " \t");
        self.pos += 1;
        var base: u8 = 10;
        if (self.take(",")) {
            if (self.take("16")) base = 16 else if (self.take("2")) base = 2 else return error.Unsupported;
        }
        try self.need(")");
        var digits = text;
        var sign: i512 = 1;
        if (digits.len != 0 and (digits[0] == '-' or digits[0] == '+')) {
            if (digits[0] == '-') sign = -1;
            digits = digits[1..];
        }
        if (digits.len >= 2 and digits[0] == '0' and
            ((base == 16 and (digits[1] == 'x' or digits[1] == 'X')) or
                (base == 2 and (digits[1] == 'b' or digits[1] == 'B')))) digits = digits[2..];
        return self.add(.{ .op = .literal, .value = sign * try radix(digits, base), .text = digits, .base = base });
    }

    fn literal(self: *Parser) Invalid!usize {
        self.space();
        const begin = self.pos;
        var base: u8 = 10;
        if (self.pos + 2 <= self.source.len and self.source[self.pos] == '0') {
            const c = self.source[self.pos + 1];
            if (c == 'x' or c == 'X') base = 16 else if (c == 'b' or c == 'B') base = 2;
            if (base != 10) self.pos += 2;
        }
        const start = self.pos;
        while (self.pos < self.source.len and digit(self.source[self.pos]) < base) : (self.pos += 1) {}
        if (base == 10 and self.pos - begin > 1 and self.source[begin] == '0') return error.Unsupported;
        const text = self.source[start..self.pos];
        return self.add(.{ .op = .literal, .value = try radix(text, base), .text = text, .base = base });
    }

    /// The node's value, folded once (the checks and arithmetic in runner_frontend_integer_prove.zig).
    pub fn prove(self: *Parser, index: usize, root: usize) Invalid!i512 {
        return @import("runner_frontend_integer_prove.zig").prove(self, index, root);
    }
};

fn digit(c: u8) u8 {
    return if (c >= '0' and c <= '9') c - '0' else if (c >= 'a' and c <= 'f') c - 'a' + 10 else if (c >= 'A' and c <= 'F') c - 'A' + 10 else 255;
}

fn radix(text: []const u8, base: u8) Invalid!i512 {
    if (text.len == 0) return error.Unsupported;
    var n: i512 = 0;
    for (text) |c| {
        if (digit(c) >= base) return error.Unsupported;
        n = try bound(n * base + digit(c));
    }
    return n;
}

pub fn bound(n: i512) Invalid!i512 {
    if (n <= -limit or n >= limit) return error.Unsupported;
    return n;
}
