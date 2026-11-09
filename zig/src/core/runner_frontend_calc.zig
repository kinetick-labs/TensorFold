const std = @import("std");

pub const max_input = 255;
pub const max_digits = 62;

// Returns firmware input, never a host-computed result; null leaves the model's continuation untouched.
pub fn lower(allocator: std.mem.Allocator, source: []const u8) !?[]u8 {
    var code = std.mem.trim(u8, source, " \t\r\n");
    if (std.mem.startsWith(u8, code, ">>> ")) code = std.mem.trim(u8, code[4..], " \t\r\n");
    if (std.mem.startsWith(u8, code, "print")) {
        const call = std.mem.trimStart(u8, code[5..], " \t");
        if (call.len < 3 or call[0] != '(' or call[call.len - 1] != ')') return null;
        code = call[1 .. call.len - 1];
    }
    if (code.len == 0 or code.len > 4096) return null;
    var parser: Parser = .{ .source = code };
    parser.expression(0) catch return null;
    parser.space();
    if (parser.pos != code.len or parser.len == 0) return null;
    if (parser.numbers > (if (parser.rational) @as(usize, 4) else 6)) return null;
    if (!@import("runner_frontend_value.zig").valid(parser.output[0..parser.len])) return null;
    return try allocator.dupe(u8, parser.output[0..parser.len]);
}

const Invalid = error{Unsupported};
const Parser = struct {
    source: []const u8,
    pos: usize = 0,
    output: [max_input]u8 = undefined,
    len: usize = 0,
    atoms: usize = 0,
    numbers: usize = 0,
    rational: bool = false,

    fn space(self: *Parser) void {
        while (self.pos < self.source.len and (self.source[self.pos] == ' ' or self.source[self.pos] == '\t'))
            self.pos += 1;
    }

    fn emit(self: *Parser, bytes: []const u8) Invalid!void {
        if (self.len + bytes.len > max_input) return error.Unsupported;
        @memcpy(self.output[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
    }

    fn take(self: *Parser, token: []const u8) Invalid!bool {
        self.space();
        if (!std.mem.startsWith(u8, self.source[self.pos..], token)) return false;
        self.pos += token.len;
        try self.emit(token);
        return true;
    }

    fn expression(self: *Parser, depth: usize) Invalid!void {
        if (depth > 6) return error.Unsupported;
        try self.term(depth);
        while (true) {
            if (try self.take("+") or try self.take("-")) try self.term(depth) else break;
        }
    }

    fn term(self: *Parser, depth: usize) Invalid!void {
        try self.factor(depth);
        while (true) {
            self.space();
            const rest = self.source[self.pos..];
            if (std.mem.startsWith(u8, rest, "**")) return error.Unsupported;
            const integral = try self.take("//") or try self.take("%");
            if (!integral) {
                if (try self.take("/")) self.rational = true else if (!try self.take("*")) break;
            }
            try self.factor(depth);
        }
    }

    fn factor(self: *Parser, depth: usize) Invalid!void {
        if (depth > 6) return error.Unsupported;
        if (try self.take("+") or try self.take("-")) return self.factor(depth + 1);
        try self.atom(depth);
        if (try self.take("**")) {
            try self.factor(depth + 1);
        }
    }

    fn atom(self: *Parser, depth: usize) Invalid!void {
        self.space();
        self.atoms += 1;
        if (self.atoms > 12) return error.Unsupported;
        if (try self.take("gcd")) {
            if (!try self.take("(")) return error.Unsupported;
            try self.expression(depth + 1);
            if (!try self.take(",")) return error.Unsupported;
            try self.expression(depth + 1);
            if (!try self.take(")")) return error.Unsupported;
        } else if (try self.take("(")) {
            try self.expression(depth + 1);
            if (!try self.take(")")) return error.Unsupported;
        } else {
            const start = self.pos;
            var digits: usize = 0;
            while (self.pos < self.source.len and std.ascii.isDigit(self.source[self.pos])) : (self.pos += 1)
                digits += 1;
            if (digits == 0) return error.Unsupported;
            if (self.pos - start > 1 and self.source[start] == '0') return error.Unsupported;
            if (self.pos < self.source.len and self.source[self.pos] == '.') {
                self.pos += 1;
                const fractional = self.pos;
                while (self.pos < self.source.len and std.ascii.isDigit(self.source[self.pos])) : (self.pos += 1)
                    digits += 1;
                if (fractional == self.pos) return error.Unsupported;
            }
            if (digits > max_digits) return error.Unsupported;
            self.numbers += 1;
            try self.emit(self.source[start..self.pos]);
        }
    }
};

test "model arithmetic becomes calc firmware input" {
    const cases = [_][2][]const u8{
        .{ "print(123456789012345678901234567890 * 987654321098765432109876543210)", "123456789012345678901234567890*987654321098765432109876543210" },
        .{ ">>> 2**100\n", "2**100" },
        .{ "print((1/3 + 2/3))", "(1/3+2/3)" },
        .{ "gcd (48, 18)", "gcd(48,18)" },
        .{ "print(1.25 * 2.4)", "1.25*2.4" },
        .{ "-2**2", "-2**2" },
        .{ "7 // -3 % 2", "7//-3%2" },
    };
    for (cases) |case| {
        const result = (try lower(std.testing.allocator, case[0])).?;
        defer std.testing.allocator.free(result);
        try std.testing.expectEqualStrings(case[1], result);
    }
}

test "unsupported cells abstain instead of extracting a plausible subexpression" {
    const cases = [_][]const u8{
        "",            "print(x*y)",       "import os",          "print(2); print(3)",                                              "2^100",                                       "sum(i*i for i in range(10))",
        "print(2, 3)", "print(2, end='')", "print(2) # comment", "2\n3",                                                            ">>> 2**100\n1267650600228229401496703205376", "1e10",
        "0x10",        "1_000",            "012",                "2***3",                                                           "gcd(1)",                                      "gcd(1,2,3)",
        "1.2//2",      "1.2/2",            "print(2)garbage",    "123456789012345678901234567890123456789012345678901234567890123", "((((((((((((1))))))))))))",
    };
    for (cases) |code| try std.testing.expectEqual(@as(?[]u8, null), try lower(std.testing.allocator, code));
}

test "firmware resource limits abstain before an error could replace model text" {
    const bad = [_][]const u8{
        "99999999999999999999999999999999999999999999999999999999999999+0",
        "99999999999999999999999999999999999999999999999999999999999999*0",
        "1+(2+(3+(4+(5+(6+7)))))",
        "(1/2)+(1/3)+(1/4)",
    };
    for (bad) |source| {
        const result = try lower(std.testing.allocator, source);
        defer if (result) |bytes| std.testing.allocator.free(bytes);
        try std.testing.expect(result == null);
    }
}
