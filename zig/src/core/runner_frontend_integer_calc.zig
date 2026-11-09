const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const calc = @import("runner_frontend_calc.zig");

// Expand immutable roots with original decimal digits; calc retains its grammar and firmware safety checks.
pub fn lower(a: std.mem.Allocator, parser: *const ast.Parser, root: usize) std.mem.Allocator.Error!?[]u8 {
    var renderer = Renderer{ .parser = parser };
    renderer.node(root, 0) catch return null;
    return calc.lower(a, renderer.bytes[0..renderer.len]);
}

const Renderer = struct {
    parser: *const ast.Parser,
    bytes: [calc.max_input]u8 = undefined,
    len: usize = 0,
    visits: usize = 0,

    fn emit(self: *Renderer, text: []const u8) ast.Invalid!void {
        if (text.len > self.bytes.len - self.len) return error.Unsupported;
        @memcpy(self.bytes[self.len..][0..text.len], text);
        self.len += text.len;
    }

    fn node(self: *Renderer, index: usize, depth: usize) ast.Invalid!void {
        if (index >= self.parser.count or depth > 12 or self.visits == 128) return error.Unsupported;
        self.visits += 1;
        const n = self.parser.nodes[index];
        if (n.op == .literal) {
            if (n.base != 10 or n.text.len == 0) return error.Unsupported;
            if (n.value < 0) try self.emit("-");
            return self.emit(n.text);
        }
        if (!self.parser.proved[index]) return error.Unsupported;
        const token = switch (n.op) {
            .neg => "-",
            .add => "+",
            .sub => "-",
            .mul => "*",
            .floor => "//",
            .mod => "%",
            .pow => "**",
            .gcd => ",",
            else => return error.Unsupported,
        };
        if (n.l >= index or (n.op != .neg and n.r >= index)) return error.Unsupported;
        try self.emit(if (n.op == .gcd) "gcd(" else "(");
        if (n.op == .neg) {
            try self.emit(token);
            try self.node(n.l, depth + 1);
        } else {
            try self.node(n.l, depth + 1);
            try self.emit(token);
            try self.node(n.r, depth + 1);
        }
        try self.emit(")");
    }
};
