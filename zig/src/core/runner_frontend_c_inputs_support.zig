const std = @import("std");

pub fn jsonDepth(s: []const u8) bool {
    var depth: usize = 0;
    var quoted = false;
    var escape = false;
    for (s) |c| {
        if (quoted) {
            if (escape) escape = false else if (c == '\\') escape = true else if (c == '"') quoted = false;
        } else if (c == '"') {
            quoted = true;
        } else if (c == '{' or c == '[') {
            depth += 1;
            if (depth > 64) return false;
        } else if (c == '}' or c == ']') {
            if (depth == 0) return false;
            depth -= 1;
        }
    }
    return !quoted and depth == 0;
}

pub fn integerType(s: []const u8) bool {
    const types = [_][]const u8{ "char", "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long", "_Bool" };
    for (types) |t| if (std.mem.eql(u8, s, t)) return true;
    return false;
}
pub fn objectType(s: []const u8) bool {
    if (s.len == 0 or s.len > 512) return false;
    const bracket = std.mem.indexOfScalar(u8, s, '[') orelse s.len;
    var tokens = std.mem.tokenizeAny(u8, s[0..bracket], " *");
    var base = false;
    while (tokens.next()) |token| {
        if (!integerType(token) and !std.mem.eql(u8, token, "unsigned") and
            !std.mem.eql(u8, token, "signed") and !std.mem.eql(u8, token, "const") and
            !std.mem.eql(u8, token, "volatile")) return false;
        if (integerType(token)) base = true;
    }
    var i = bracket;
    while (i < s.len) {
        if (s[i] != '[') return false;
        i += 1;
        const begin = i;
        while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) {}
        if (i == begin or i == s.len or s[i] != ']') return false;
        i += 1;
    }
    return base;
}

// RV32 ILP32 object sizes; reject incomplete, variable or oversized shapes before generating copies.
pub fn objectBytes(s: []const u8) ?usize {
    if (!objectType(s)) return null;
    const bracket = std.mem.indexOfScalar(u8, s, '[') orelse s.len;
    const base = s[0..bracket];
    var bytes: usize = 4;
    if (std.mem.indexOfScalar(u8, base, '*') == null) {
        var tokens = std.mem.tokenizeScalar(u8, base, ' ');
        var longs: usize = 0;
        while (tokens.next()) |word| {
            if (std.mem.eql(u8, word, "char") or std.mem.eql(u8, word, "_Bool")) bytes = 1;
            if (std.mem.eql(u8, word, "short")) bytes = 2;
            if (std.mem.eql(u8, word, "long")) longs += 1;
        }
        if (longs == 2) bytes = 8;
    }
    var i = bracket;
    while (i < s.len) {
        const end = std.mem.indexOfScalarPos(u8, s, i, ']') orelse return null;
        const count = std.fmt.parseInt(usize, s[i + 1 .. end], 10) catch return null;
        if (count == 0 or count > 32768 / bytes) return null;
        bytes *= count;
        i = end + 1;
    }
    return bytes;
}

pub fn expressionType(s: []const u8) bool {
    if (objectType(s)) return true;
    const pointer = std.mem.indexOf(u8, s, " (*)") orelse return false;
    var buffer: [512]u8 = undefined;
    const tail = s[pointer + 4 ..];
    if (pointer + tail.len > buffer.len) return false;
    @memcpy(buffer[0..pointer], s[0..pointer]);
    @memcpy(buffer[pointer..][0..tail.len], tail);
    return objectType(buffer[0 .. pointer + tail.len]);
}

pub fn readBytes(s: []const u8) ?usize {
    if (std.mem.indexOf(u8, s, " (*)") != null) return if (expressionType(s)) 4 else null;
    if (std.mem.indexOfScalar(u8, s, '[') != null) return null;
    return objectBytes(s);
}

// The expression runs once; typeof is unevaluated because supported scalar types exclude variably modified types.
pub fn readExpression(a: std.mem.Allocator, expr: []const u8, type_expr: []const u8, bytes: usize, id: usize) (std.mem.Allocator.Error || error{Unsupported})![]const u8 {
    if (expr.len > 65536 or type_expr.len > 16384) return error.Unsupported;
    if (bytes == 8) return std.fmt.allocPrint(a, "({{ unsigned long long __tfimr_r_{d} = (unsigned long long)({s}); " ++
        "unsigned int __tfimr_lo_{d} = (unsigned int)__tfimr_r_{d}; " ++
        "unsigned int __tfimr_hi_{d} = (unsigned int)(__tfimr_r_{d} >> 32); " ++
        "__asm__ volatile(\"\" : \"+r\"(__tfimr_lo_{d}), \"+r\"(__tfimr_hi_{d})); " ++
        "(__typeof__({s}))((unsigned long long)__tfimr_lo_{d} | ((unsigned long long)__tfimr_hi_{d} << 32)); }})", .{ id, expr, id, id, id, id, id, id, type_expr, id, id });
    if (bytes != 1 and bytes != 2 and bytes != 4) return error.Unsupported;
    return std.fmt.allocPrint(a, "({{ unsigned int __tfimr_r_{d} = (unsigned int)({s}); " ++
        "__asm__ volatile(\"\" : \"+r\"(__tfimr_r_{d})); (__typeof__({s}))__tfimr_r_{d}; }})", .{ id, expr, id, type_expr, id });
}
