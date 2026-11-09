const std = @import("std");
const m = @import("runner_frontend_c_cost_model.zig");
const Error = m.Error;
const Value = m.Value;

pub fn call(p: anytype, name: []const u8, args: []const Value) Error!Value {
    if (std.mem.eql(u8, name, "putchar")) {
        if (args.len != 1) return error.Unsupported;
        _ = try args[0].numeric();
        try p.charge(20);
        return .{ .number = .{ .lo = 0, .hi = 255 } };
    }
    if (std.mem.eql(u8, name, "puts")) {
        if (args.len != 1) return error.Unsupported;
        const s = try terminated(args[0]);
        try p.charge(24 + (s.len + 1) * 16);
        return Value.integer(0);
    }
    if (std.mem.eql(u8, name, "tf_print_int")) {
        if (args.len != 1) return error.Unsupported;
        _ = try decimal(p, args[0], false, false);
        try p.charge(20);
        return Value.integer(0);
    }
    if (!std.mem.eql(u8, name, "printf") or args.len == 0) return error.Unsupported;
    const format = args[0].text orelse return error.Unsupported;
    var i: usize = 0;
    var argument: usize = 1;
    var bytes: usize = 0;
    try p.charge(32);
    while (i < format.len) : (i += 1) {
        if (format[i] != '%') {
            try p.charge(16);
            bytes += 1;
            continue;
        }
        i += 1;
        if (i >= format.len) return error.Unsupported;
        if (format[i] == '%') {
            try p.charge(20);
            bytes += 1;
            continue;
        }
        var wide = false;
        if (i + 1 < format.len and format[i] == 'l' and format[i + 1] == 'l') {
            wide = true;
            i += 2;
        }
        if (i >= format.len or argument >= args.len) return error.Unsupported;
        const value = args[argument];
        argument += 1;
        try p.charge(16);
        switch (format[i]) {
            'u', 'i', 'd' => bytes += try decimal(p, value, format[i] == 'u', wide),
            'c' => {
                if (wide) return error.Unsupported;
                _ = try value.numeric();
                bytes += 1;
                try p.charge(20);
            },
            's' => {
                if (wide) return error.Unsupported;
                const s = try terminated(value);
                bytes += s.len;
                try p.charge(20 + s.len * 16);
            },
            else => return error.Unsupported,
        }
    }
    if (argument != args.len) return error.Unsupported;
    return .{ .number = .{ .lo = 0, .hi = @intCast(bytes) } };
}

fn terminated(v: Value) Error![]const u8 {
    if (v.text) |s| return s;
    if (v.array) |a| if (a.text) |s| return s;
    return error.Unsupported;
}

fn decimal(p: anytype, value: Value, unsigned: bool, wide: bool) Error!usize {
    const r = try value.numeric();
    const t: m.Type = .{ .bits = if (wide) 64 else 32, .signed = !unsigned };
    if (!t.fits(r)) return error.Unsupported;
    var n = @max(@abs(r.lo), @abs(r.hi));
    const big = n > std.math.maxInt(u32);
    var digits: usize = 1;
    while (n >= 10) : (n /= 10) digits += 1;
    try p.charge(40 + digits * 28);
    if (wide) {
        try p.charge(12);
        if (big) try p.charge(digits * 64 * 24);
    }
    if (r.lo < 0) try p.charge(24);
    return digits + @intFromBool(r.lo < 0);
}
