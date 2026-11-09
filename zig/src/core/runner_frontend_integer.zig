const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const cost = @import("runner_frontend_integer_cost.zig");

/// graph_only: only the graph route takes the table (lists, more than eight values); `source` is empty.
pub const Lowered = struct { source: []u8, estimated_steps: ?usize, table: ?Table = null, graph_only: bool = false };

/// A lowered script's DAG and ordered outputs, the data the runtime evaluates; literal text is dropped (use value).
pub const Table = struct {
    nodes: [ast.max_nodes]ast.Node = undefined,
    count: usize = 0,
    roots: [32]usize = undefined,
    endings: [32]u8 = undefined, // ' ', '\n', or ',' between a printed list's elements
    styles: [32]u8 = @splat(0), // 0 integer/bool/tuple, 1 exact integral float, 2 fixed Decimal
    scales: [32]u8 = @splat(0),
    brackets: [32]u8 = @splat(0), // 1: a list's `[` before the value, 2: its `]` after
    outputs: usize = 0,
};

/// The table lowerFormatted renders as C; a non-literal node's value is a proof, never a result to print.
pub fn table(parser: *const ast.Parser, roots: []const usize, endings: []const u8, styles: ?[]const u8, scales: ?[]const u8) Table {
    var t: Table = .{ .count = parser.count, .outputs = roots.len };
    for (parser.nodes[0..parser.count], t.nodes[0..parser.count]) |n, *m| m.* = .{ .op = n.op, .l = n.l, .r = n.r, .k = n.k, .value = n.value };
    @memcpy(t.roots[0..roots.len], roots);
    @memcpy(t.endings[0..roots.len], endings);
    if (styles) |s| @memcpy(t.styles[0..roots.len], s);
    if (scales) |s| @memcpy(t.scales[0..roots.len], s);
    return t;
}

pub fn ambiguousPercent(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!bool {
    if (std.mem.indexOfScalar(u8, source, '%') == null) return false;
    var parser = ast.Parser{ .source = source };
    _ = parser.parse() catch return false;
    const percentage = try @import("runner_frontend_inline.zig").lower(a, source, true) orelse return false;
    defer a.free(percentage);
    return true;
}

// Proof values decide admission; emitted nodes retain the requested arithmetic and original literal values.
pub fn lower(a: std.mem.Allocator, source: []const u8) std.mem.Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, source, 0) orelse return null;
    return result.source;
}

pub fn lowerWithCost(a: std.mem.Allocator, source: []const u8, budget: usize) std.mem.Allocator.Error!?Lowered {
    var code = std.mem.trim(u8, source, " \t\r\n");
    if (std.mem.startsWith(u8, code, ">>> ")) code = std.mem.trim(u8, code[4..], " \t\r\n");
    if (std.mem.startsWith(u8, code, "print")) {
        const rest = std.mem.trimStart(u8, code[5..], " \t");
        if (rest.len < 3 or rest[0] != '(' or rest[rest.len - 1] != ')') return null;
        code = rest[1 .. rest.len - 1];
    }
    if (code.len == 0 or code.len > 4096) return null;
    var parser = ast.Parser{ .source = code };
    const root = parser.parse() catch return null;
    var steps: ?usize = null;
    if (budget != 0) steps = cost.estimate(&parser, root, budget) orelse return null;
    return .{ .source = try lowerParser(a, &parser, root), .estimated_steps = steps };
}

// Emit only a successfully proved root; proof values never replace operations or literal text.
pub fn lowerParser(a: std.mem.Allocator, parser: *ast.Parser, root: usize) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, @embedFile("runner_frontend_integer_runtime.c"));
    try out.appendSlice(a, "\nstatic const struct tf_node tf_nodes[] = {\n");
    var emitter = Emitter{ .a = a, .parser = parser, .out = &out };
    _ = try emitter.node(root);
    try out.print(a, "}};\nint main(void) {{ return tf_run(tf_nodes, {d}); }}\n", .{emitter.count});
    return try out.toOwnedSlice(a);
}

// All ordered print roots share one original DAG and are evaluated before the first output.
pub fn lowerParsers(a: std.mem.Allocator, parser: *ast.Parser, roots: []const usize) std.mem.Allocator.Error![]u8 {
    const endings: [8]u8 = @splat('\n');
    return lowerPrints(a, parser, roots, endings[0..roots.len]);
}

// A space continues a grouped print and a newline ends it; all roots share their original runtime nodes.
pub fn lowerPrints(a: std.mem.Allocator, parser: *ast.Parser, roots: []const usize, endings: []const u8) std.mem.Allocator.Error![]u8 {
    const defaults: [8]u8 = @splat(0);
    return lowerFormatted(a, parser, roots, endings, defaults[0..roots.len], defaults[0..roots.len]);
}

// Styles: 0 ordinary integer/bool/tuple, 1 exact integral float, 2 fixed Decimal.
pub fn lowerFormatted(a: std.mem.Allocator, parser: *ast.Parser, roots: []const usize, endings: []const u8, styles: []const u8, scales: []const u8) std.mem.Allocator.Error![]u8 {
    std.debug.assert(roots.len > 0 and roots.len <= 8 and endings.len == roots.len);
    for (endings) |ending| std.debug.assert(ending == ' ' or ending == '\n');
    std.debug.assert(styles.len == roots.len and scales.len == roots.len);
    var formatted = false;
    for (styles, scales) |style, scale| {
        std.debug.assert(style <= 2 and scale <= 6);
        formatted = formatted or style != 0;
    }
    if (!formatted and roots.len == 1 and endings[0] == '\n') return lowerParser(a, parser, roots[0]);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, @embedFile("runner_frontend_integer_runtime.c"));
    try out.appendSlice(a, "\nstatic const struct tf_node tf_nodes[] = {\n");
    var emitter = Emitter{ .a = a, .parser = parser, .out = &out };
    var refs: [8]usize = undefined;
    for (roots, 0..) |root, i| refs[i] = try emitter.node(root);
    try out.appendSlice(a, "};\nstatic const int tf_roots[] = {");
    for (refs[0..roots.len]) |ref| try out.print(a, "{d},", .{ref});
    try out.appendSlice(a, "};\nstatic const unsigned char tf_endings[] = {");
    for (endings) |ending| try out.print(a, "{d},", .{ending});
    if (formatted) {
        try out.appendSlice(a, "};\nstatic const unsigned char tf_styles[] = {");
        for (styles) |style| try out.print(a, "{d},", .{style});
        try out.appendSlice(a, "};\nstatic const unsigned char tf_scales[] = {");
        for (scales) |scale| try out.print(a, "{d},", .{scale});
        try out.print(a, "}};\nint main(void) {{ return tf_run_formatted(tf_nodes, {d}, tf_roots, tf_endings, " ++
            "tf_styles, tf_scales, {d}); }}\n", .{ emitter.count, roots.len });
    } else {
        try out.print(a, "}};\nint main(void) {{ return tf_run_prints(tf_nodes, {d}, tf_roots, tf_endings, {d}); }}\n", .{
            emitter.count, roots.len,
        });
    }
    return try out.toOwnedSlice(a);
}

pub fn compile(a: std.mem.Allocator, io: std.Io, code: []const u8, config: compiler.Config) std.mem.Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code, config.step_budget) orelse return null;
    defer a.free(result.source);
    var runtime_config = config;
    runtime_config.estimated_steps = result.estimated_steps;
    if (runtime_config.optimize != .O0) runtime_config.optimize = .O2;
    runtime_config.opaque_inputs = true;
    return compiler.compile(a, io, result.source, runtime_config);
}

const Emitter = struct {
    a: std.mem.Allocator,
    parser: *const ast.Parser,
    out: *std.ArrayList(u8),
    count: usize = 0,
    mapped: [64]?usize = @splat(null),

    fn node(self: *Emitter, index: usize) std.mem.Allocator.Error!usize {
        if (self.mapped[index]) |mapped| return mapped;
        const n = self.parser.nodes[index];
        var l: usize = 0;
        var r: usize = 0;
        var k: usize = 0;
        if (n.op != .literal) l = try self.node(n.l);
        switch (n.op) {
            .literal, .neg, .digits, .isqrt, .factorial, .digit_sum => {},
            else => r = try self.node(n.r),
        }
        if (n.op == .powmod or n.op == .range_product) k = try self.node(n.k);
        if (n.op == .literal) {
            try self.out.print(self.a, "{{0,0,0,{d},\"{s}{s}\"}},\n", .{
                n.base, if (n.value < 0) "-" else "", n.text,
            });
        } else {
            var op: u8 = @backingInt(n.op);
            if (n.op == .mod) {
                for (self.parser.nodes[0..self.parser.count], 0..) |prior, j| {
                    if ((prior.op == .floor or prior.op == .divmod) and prior.l == n.l and prior.r == n.r) {
                        if (self.mapped[j]) |ref| {
                            op = 22;
                            l = ref;
                            r = 0;
                            break;
                        }
                    }
                }
            }
            try self.out.print(self.a, "{{{d},{d},{d},{d},0}},\n", .{ op, l, r, k });
        }
        const result = self.count;
        self.count += 1;
        self.mapped[index] = result;
        return result;
    }
};
