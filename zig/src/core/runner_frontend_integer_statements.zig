const std = @import("std");
const Invalid = @import("runner_frontend_integer_ast.zig").Invalid;

pub const Statement = struct { text: []const u8, after_semicolon: bool };

// Split only physical top-level statements. Quoted conversion literals retain their original bytes.
pub const Iterator = struct {
    code: []const u8,
    pos: usize = 0,
    continuation: bool = false,

    pub fn next(self: *Iterator) Invalid!?Statement {
        if (self.pos == 0 and std.mem.indexOfScalar(u8, self.code, 0) != null) return error.Unsupported;
        if (self.pos == self.code.len) return null;
        const start = self.pos;
        const continuation = self.continuation;
        var quote: u8 = 0;
        var depth: usize = 0;
        while (self.pos < self.code.len) {
            const c = self.code[self.pos];
            if (quote != 0) {
                if (c == '\n') return error.Unsupported;
                self.pos += 1;
                if (c == '\\') {
                    if (self.pos == self.code.len) return error.Unsupported;
                    self.pos += 1;
                } else if (c == quote) quote = 0;
                continue;
            }
            if (c == '\'' or c == '"') {
                quote = c;
            } else if (c == '(' or c == '[' or c == '{') {
                depth += 1;
                if (depth > 16) return error.Unsupported;
            } else if (c == ')' or c == ']' or c == '}') {
                if (depth == 0) return error.Unsupported;
                depth -= 1;
            } else if (c == '#' or c == '\n' or (c == ';' and depth == 0)) {
                if (depth != 0) return error.Unsupported;
                const end = self.pos;
                if (c == ';' and std.mem.trim(u8, self.code[start..end], " \t").len == 0)
                    return error.Unsupported;
                if (c == '#') {
                    while (self.pos < self.code.len and self.code[self.pos] != '\n') : (self.pos += 1) {}
                }
                if (self.pos < self.code.len) self.pos += 1;
                self.continuation = c == ';';
                return .{ .text = self.code[start..end], .after_semicolon = continuation };
            }
            self.pos += 1;
        }
        if (quote != 0 or depth != 0) return error.Unsupported;
        self.continuation = false;
        return .{ .text = self.code[start..], .after_semicolon = continuation };
    }
};

// Commas and assignment signs inside calls/conversion strings are not statement syntax.
pub fn separator(text: []const u8, token: u8) ?usize {
    var quote: u8 = 0;
    var escaped = false;
    var depth: usize = 0;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
            continue;
        }
        if (c == '\'' or c == '"') quote = c else if (c == '(' or c == '[' or c == '{') {
            depth += 1;
        } else if (c == ')' or c == ']' or c == '}') {
            if (depth == 0) return null;
            depth -= 1;
        } else if (depth == 0 and c == token) {
            if (token == '=' and ((i != 0 and std.mem.indexOfScalar(u8, "=!<>", text[i - 1]) != null) or
                (i + 1 < text.len and text[i + 1] == '='))) continue;
            return i;
        }
    }
    return null;
}
