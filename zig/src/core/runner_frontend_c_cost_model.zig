const std = @import("std");
pub const V = std.json.Value;
pub const Error = std.mem.Allocator.Error || error{Unsupported};
pub const Range = struct {
    lo: i128,
    hi: i128,

    pub fn exact(n: i128) Range {
        return .{ .lo = n, .hi = n };
    }
    pub fn join(x: Range, y: Range) Range {
        return .{ .lo = @min(x.lo, y.lo), .hi = @max(x.hi, y.hi) };
    }
};
pub const Type = struct {
    bits: u8,
    signed: bool,
    plain_char: bool = false,

    pub fn range(t: Type) Range {
        if (t.plain_char) return .{ .lo = -128, .hi = 255 };
        const n = @as(i128, 1) << @intCast(t.bits - @intFromBool(t.signed));
        return .{ .lo = if (t.signed) -n else 0, .hi = n - 1 };
    }
    pub fn fits(t: Type, r: Range) bool {
        if (t.plain_char) return r.lo >= 0 and r.hi <= 127;
        const full = t.range();
        return r.lo >= full.lo and r.hi <= full.hi;
    }
};
pub const Array = struct {
    count: usize,
    element: Range,
    element_type: Type,
    text: ?[]const u8 = null,
    readable: bool = true,
};
pub const Value = struct {
    number: ?Range = null,
    array: ?Array = null,
    text: ?[]const u8 = null,

    pub fn integer(n: i128) Value {
        return .{ .number = Range.exact(n) };
    }
    pub fn numeric(v: Value) Error!Range {
        return v.number orelse error.Unsupported;
    }
    pub fn join(x: Value, y: Value) Error!Value {
        if ((x.number == null and x.array == null and x.text == null) or
            (y.number == null and y.array == null and y.text == null)) return .{};
        if (x.number) |r| return .{ .number = r.join(try y.numeric()) };
        if (x.array) |r| {
            const s = y.array orelse return error.Unsupported;
            if (r.count != s.count or r.element_type.bits != s.element_type.bits or
                r.element_type.signed != s.element_type.signed or
                r.element_type.plain_char != s.element_type.plain_char) return error.Unsupported;
            var result = r;
            result.element = r.element.join(s.element);
            result.readable = r.readable and s.readable;
            if (r.text == null or s.text == null or !std.mem.eql(u8, r.text.?, s.text.?)) result.text = null;
            return .{ .array = result };
        }
        if (x.text) |s| {
            if (y.text) |t| {
                if (std.mem.eql(u8, s, t)) return x;
            }
        }
        return error.Unsupported;
    }
};
pub const Binding = struct {
    id: []const u8,
    value: Value,
    typ: ?Type,
    global: bool = false,
    indirect: bool = false,
};
pub const Env = struct {
    bindings: std.ArrayList(Binding) = .empty,

    pub fn find(e: *Env, id: []const u8) ?*Binding {
        for (e.bindings.items) |*b| if (std.mem.eql(u8, b.id, id)) return b;
        return null;
    }
    pub fn add(e: *Env, a: std.mem.Allocator, b: Binding) Error!void {
        if (e.bindings.items.len >= 128 or e.find(b.id) != null) return error.Unsupported;
        try e.bindings.append(a, b);
    }
    pub fn clone(e: *const Env, a: std.mem.Allocator) Error!Env {
        var out: Env = .{};
        try out.bindings.appendSlice(a, e.bindings.items);
        return out;
    }
    pub fn merge(e: *Env, other: *const Env) Error!void {
        for (e.bindings.items) |*b| {
            for (other.bindings.items) |s| {
                if (!std.mem.eql(u8, b.id, s.id)) continue;
                b.value = try b.value.join(s.value);
                break;
            }
        }
    }
};
pub fn field(v: V, name: []const u8) V {
    return if (v == .object) v.object.get(name) orelse .null else .null;
}
pub fn text(v: V) []const u8 {
    return if (v == .string) v.string else "";
}
pub fn is(v: V, kind: []const u8) bool {
    return std.mem.eql(u8, text(field(v, "kind")), kind);
}
pub fn children(v: V) []const V {
    const inner = field(v, "inner");
    return if (inner == .array) inner.array.items else &.{};
}
pub fn op(v: V, s: []const u8) bool {
    return std.mem.eql(u8, text(field(v, "opcode")), s);
}
pub fn strip(v: V) V {
    var n = v;
    for (0..64) |_| {
        if (!is(n, "ParenExpr") and !is(n, "ImplicitCastExpr") and !is(n, "CStyleCastExpr") and
            !is(n, "ConstantExpr")) return n;
        const c = children(n);
        if (c.len != 1) return n;
        n = c[0];
    }
    return .null;
}
pub fn ref(v: V) []const u8 {
    const n = strip(v);
    if (!is(n, "DeclRefExpr")) return "";
    return text(field(field(n, "referencedDecl"), "id"));
}
pub fn typ(v: V) []const u8 {
    const t = field(v, "type");
    const desugared = text(field(t, "desugaredQualType"));
    return if (desugared.len != 0) desugared else text(field(t, "qualType"));
}
pub fn integerType(raw: []const u8) ?Type {
    var s = raw;
    while (true) {
        var changed = false;
        for ([_][]const u8{ "const ", "volatile ", "restrict " }) |prefix| {
            if (std.mem.startsWith(u8, s, prefix)) {
                s = s[prefix.len..];
                changed = true;
            }
        }
        if (!changed) break;
    }
    if (std.mem.eql(u8, s, "char")) return .{ .bits = 8, .signed = true, .plain_char = true };
    if (std.mem.eql(u8, s, "_Bool")) return .{ .bits = 1, .signed = false };
    var signed = true;
    if (std.mem.startsWith(u8, s, "unsigned ")) {
        signed = false;
        s = s[9..];
    } else if (std.mem.startsWith(u8, s, "signed ")) s = s[7..];
    const bits: u8 = if (std.mem.eql(u8, s, "char")) 8 else if (std.mem.eql(u8, s, "short")) 16 else if (std.mem.eql(u8, s, "int") or std.mem.eql(u8, s, "long")) 32 else if (std.mem.eql(u8, s, "long long")) 64 else return null;
    return .{ .bits = bits, .signed = signed };
}
pub fn arrayType(raw: []const u8) ?struct { element: Type, count: usize } {
    const start = std.mem.indexOfScalar(u8, raw, '[') orelse return null;
    if (raw.len < start + 3 or raw[raw.len - 1] != ']') return null;
    const count = std.fmt.parseInt(usize, raw[start + 1 .. raw.len - 1], 10) catch return null;
    if (count == 0 or count > 32768) return null;
    return .{ .element = integerType(std.mem.trim(u8, raw[0..start], " ")) orelse return null, .count = count };
}
pub fn pointerType(raw: []const u8) ?Type {
    const s = std.mem.trim(u8, raw, " ");
    if (s.len == 0 or s[s.len - 1] != '*') return null;
    return integerType(std.mem.trim(u8, s[0 .. s.len - 1], " "));
}
pub fn size(raw: []const u8) ?usize {
    if (integerType(raw)) |t| return @max(@as(usize, 1), t.bits / 8);
    if (arrayType(raw)) |t| return t.count * @max(@as(usize, 1), t.element.bits / 8);
    if (pointerType(raw) != null) return 4;
    return null;
}
pub fn integer(v: V) Error!i128 {
    const n = field(v, "value");
    return switch (n) {
        .integer => n.integer,
        .string => std.fmt.parseInt(i128, n.string, 10) catch error.Unsupported,
        else => error.Unsupported,
    };
}
pub fn string(a: std.mem.Allocator, v: V) Error![]const u8 {
    const raw = text(field(strip(v), "value"));
    if (!is(strip(v), "StringLiteral") or raw.len < 2 or raw.len > 16386 or
        raw[0] != '"' or raw[raw.len - 1] != '"')
        return error.Unsupported;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 1;
    while (i + 1 < raw.len) : (i += 1) {
        var c = raw[i];
        if (c == '\\') {
            i += 1;
            if (i + 1 >= raw.len) return error.Unsupported;
            c = switch (raw[i]) {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                '0'...'7' => blk: {
                    var n: u16 = raw[i] - '0';
                    var digits: usize = 1;
                    while (digits < 3 and i + 2 < raw.len and raw[i + 1] >= '0' and raw[i + 1] <= '7') : (digits += 1) {
                        i += 1;
                        n = n * 8 + raw[i] - '0';
                    }
                    if (n > 255) return error.Unsupported;
                    break :blk @intCast(n);
                },
                else => return error.Unsupported,
            };
        }
        if (c == 0) return error.Unsupported;
        if (c >= 128) return error.Unsupported;
        try out.append(a, c);
    }
    return out.items;
}
pub fn validateJson(s: []const u8) bool {
    var depth: usize = 0;
    var quoted = false;
    var escaped = false;
    var objects: usize = 0;
    for (s) |c| {
        if (quoted) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == '"') quoted = false;
        } else if (c == '"') {
            quoted = true;
        } else if (c == '[' or c == '{') {
            depth += 1;
            if (c == '{') objects += 1;
            if (depth > 64 or objects > 32768) return false;
        } else if (c == ']' or c == '}') {
            if (depth == 0) return false;
            depth -= 1;
        }
    }
    return depth == 0 and !quoted;
}
