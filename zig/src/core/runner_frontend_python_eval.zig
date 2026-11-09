const std = @import("std");
const ast = @import("runner_frontend_python_ast.zig");
const Error = ast.Error;
const Scalar = struct { n: i64 = 0, boolean: bool = false };
const List = struct { items: [128]Scalar = undefined, len: usize = 0 };
const Value = union(enum) { scalar: Scalar, list: *List };
const Frame = struct {
    scope: *ast.Scope,
    function: ?usize = null,
    values: [64]Value = undefined,
    defined: [64]bool = @splat(false),
};
const Binding = struct { name: []const u8, value: Scalar = .{}, parent: ?*const Binding = null };
const Range = struct { start: i64 = 0, stop: i64, step: i64 = 1 };

pub fn prove(p: *ast.Program) Error!void {
    p.estimated_steps = 0;
    if (p.functions.items.len > 32) return error.Unsupported;
    try infer(p, &p.globals, p.body);
    for (p.functions.items) |*f| {
        if (p.globals.find(f.name) != null) return error.Unsupported;
        for (f.params) |name| {
            const id = f.scope.find(name) orelse return error.Unsupported;
            f.scope.symbols.items[id].kind = .scalar;
        }
        try inferGeneratorParams(p, f, f.body);
        try infer(p, &f.scope, f.body);
        for (f.scope.symbols.items) |s| if (p.functionId(s.name) != null) return error.Unsupported;
    }
    var v: Validator = .{ .p = p };
    try v.block(&p.globals, null, p.body);
    for (p.functions.items, 0..) |*f, id| try v.block(&f.scope, id, f.body);
    for (p.functions.items, 0..) |_, id| try v.acyclic(id, 0);
    var globals: Frame = .{ .scope = &p.globals };
    var r: Runtime = .{ .p = p, .globals = &globals };
    if (try r.block(&globals, p.body) != null) return error.Unsupported;
    p.estimated_steps = r.estimated_steps;
}

fn inferGeneratorParams(p: *ast.Program, f: *ast.Function, body: []const ast.Stmt) Error!void {
    for (body) |s| {
        switch (s.tag) {
            .assign, .ret, .assertion, .conditional, .loop, .each => try generatorParams(p, f, s.expr),
            else => {},
        }
        try inferGeneratorParams(p, f, s.body);
        try inferGeneratorParams(p, f, s.otherwise);
    }
}

fn generatorParams(p: *ast.Program, f: *ast.Function, id: usize) Error!void {
    const e = p.nodes.items[id];
    if (e.tag == .sum_filtered and p.nodes.items[e.rhs].tag == .name) {
        const name = p.nodes.items[e.rhs].text;
        for (f.params) |param| if (std.mem.eql(u8, param, name)) {
            f.scope.symbols.items[f.scope.find(param).?].kind = .list;
        };
    }
    switch (e.tag) {
        .integer, .boolean, .name, .string => {},
        .format, .list, .call => for (e.args) |arg| try generatorParams(p, f, arg),
        .neg, .logical_not, .reverse_digits => try generatorParams(p, f, e.lhs),
        else => {
            try generatorParams(p, f, e.lhs);
            try generatorParams(p, f, e.rhs);
            if (e.filter) |filter| try generatorParams(p, f, filter);
        },
    }
}

fn infer(p: *ast.Program, scope: *ast.Scope, body: []const ast.Stmt) Error!void {
    for (body) |s| {
        if (s.tag == .assign or s.tag == .each) {
            const id = scope.find(s.name) orelse return error.Unsupported;
            const kind: ast.Kind = if (s.tag == .assign and p.nodes.items[s.expr].tag == .list) .list else .scalar;
            const previous = scope.symbols.items[id].kind;
            if (previous != .unknown and previous != kind) return error.Unsupported;
            scope.symbols.items[id].kind = kind;
        }
        if (s.tag != .function) {
            try infer(p, scope, s.body);
            try infer(p, scope, s.otherwise);
        }
    }
}

const Validator = struct {
    p: *ast.Program,
    edges: [32]u32 = @splat(0),
    checked: u32 = 0,
    depth: usize = 0,

    fn symbol(v: *Validator, scope: *ast.Scope, name: []const u8, gen: ?*const Binding) Error!ast.Kind {
        var binding = gen;
        while (binding) |g| : (binding = g.parent) if (std.mem.eql(u8, g.name, name)) return .scalar;
        if (scope.find(name)) |id| {
            const kind = scope.symbols.items[id].kind;
            if (kind == .unknown) return error.Unsupported;
            return kind;
        }
        if (scope != &v.p.globals) if (v.p.globals.find(name)) |id| {
            const kind = v.p.globals.symbols.items[id].kind;
            if (kind != .scalar) return error.Unsupported;
            return kind;
        };
        return error.Unsupported;
    }

    fn expr(v: *Validator, scope: *ast.Scope, fun: ?usize, id: usize, gen: ?*const Binding) Error!ast.Kind {
        if (v.depth >= 64) return error.Unsupported;
        v.depth += 1;
        defer v.depth -= 1;
        const e = v.p.nodes.items[id];
        switch (e.tag) {
            .integer, .boolean => return .scalar,
            .name => return v.symbol(scope, e.text, gen),
            .string, .format => return error.Unsupported,
            .list => {
                if (e.args.len > 128) return error.Unsupported;
                for (e.args) |arg| try v.scalar(scope, fun, arg, gen);
                return .list;
            },
            .index => {
                if (try v.expr(scope, fun, e.lhs, gen) != .list) return error.Unsupported;
                try v.scalar(scope, fun, e.rhs, gen);
                return .scalar;
            },
            .neg, .logical_not, .reverse_digits => try v.scalar(scope, fun, e.lhs, gen),
            .call => {
                if (std.mem.eql(u8, e.text, "len") or std.mem.eql(u8, e.text, "sum")) {
                    if (e.args.len != 1 or try v.expr(scope, fun, e.args[0], gen) != .list)
                        return error.Unsupported;
                } else if (std.mem.eql(u8, e.text, "gcd")) {
                    if (!v.p.gcd_imported or e.args.len != 2) return error.Unsupported;
                    for (e.args) |arg| try v.scalar(scope, fun, arg, gen);
                } else {
                    const callee = v.p.functionId(e.text) orelse return error.Unsupported;
                    if (e.args.len != v.p.functions.items[callee].params.len) return error.Unsupported;
                    if (fun) |caller| v.edges[caller] |= @as(u32, 1) << @intCast(callee);
                    const f = &v.p.functions.items[callee];
                    for (e.args, f.params) |arg, param| {
                        if (try v.expr(scope, fun, arg, gen) != f.scope.symbols.items[f.scope.find(param).?].kind)
                            return error.Unsupported;
                    }
                }
            },
            .sum_filtered => {
                if (!ast.validName(e.text) or v.p.functionId(e.text) != null) return error.Unsupported;
                const seq = v.p.nodes.items[e.rhs];
                if (seq.tag == .call and std.mem.eql(u8, seq.text, "range")) {
                    try v.range(scope, fun, e.rhs, gen);
                } else if (try v.expr(scope, fun, e.rhs, gen) != .list) return error.Unsupported;
                const local: Binding = .{ .name = e.text, .parent = gen };
                if (e.filter) |filter| try v.scalar(scope, fun, filter, &local);
                try v.scalar(scope, fun, e.lhs, &local);
            },
            .all_range => {
                if (!ast.validName(e.text) or v.p.functionId(e.text) != null) return error.Unsupported;
                try v.range(scope, fun, e.rhs, gen);
                const local: Binding = .{ .name = e.text, .parent = gen };
                try v.scalar(scope, fun, e.lhs, &local);
            },
            else => {
                try v.scalar(scope, fun, e.lhs, gen);
                try v.scalar(scope, fun, e.rhs, gen);
            },
        }
        return .scalar;
    }

    fn scalar(v: *Validator, scope: *ast.Scope, fun: ?usize, id: usize, gen: ?*const Binding) Error!void {
        if (try v.expr(scope, fun, id, gen) != .scalar) return error.Unsupported;
    }

    fn range(v: *Validator, scope: *ast.Scope, fun: ?usize, id: usize, gen: ?*const Binding) Error!void {
        const e = v.p.nodes.items[id];
        if (e.tag != .call or !std.mem.eql(u8, e.text, "range") or e.args.len < 1 or e.args.len > 3)
            return error.Unsupported;
        for (e.args) |arg| try v.scalar(scope, fun, arg, gen);
    }

    fn printed(v: *Validator, scope: *ast.Scope, fun: ?usize, id: usize) Error!void {
        const e = v.p.nodes.items[id];
        if (e.tag == .string) return;
        if (e.tag == .format) {
            for (e.args) |arg| {
                if (v.p.nodes.items[arg].tag != .string) try v.scalar(scope, fun, arg, null);
            }
        } else try v.scalar(scope, fun, id, null);
    }

    fn block(v: *Validator, scope: *ast.Scope, fun: ?usize, body: []const ast.Stmt) Error!void {
        for (body) |s| {
            switch (s.tag) {
                .assign => {
                    const id = scope.find(s.name) orelse return error.Unsupported;
                    if (try v.expr(scope, fun, s.expr, null) != scope.symbols.items[id].kind)
                        return error.Unsupported;
                    if (scope.symbols.items[id].kind == .list and v.p.nodes.items[s.expr].tag != .list)
                        return error.Unsupported;
                    if (fun) |function| {
                        if (scope.symbols.items[id].kind == .list) {
                            for (v.p.functions.items[function].params) |param|
                                if (std.mem.eql(u8, param, s.name)) return error.Unsupported;
                        }
                    }
                },
                .append => {
                    if (fun != null or try v.symbol(scope, s.name, null) != .list) return error.Unsupported;
                    try v.scalar(scope, fun, s.expr, null);
                },
                .conditional, .loop, .assertion => try v.scalar(scope, fun, s.expr, null),
                .each => {
                    const e = v.p.nodes.items[s.expr];
                    if (e.tag == .list) {
                        _ = try v.expr(scope, fun, s.expr, null);
                    } else try v.range(scope, fun, s.expr, null);
                },
                .ret => {
                    if (fun == null) return error.Unsupported;
                    try v.scalar(scope, fun, s.expr, null);
                },
                .print => {
                    if (fun != null) return error.Unsupported;
                    for (s.args) |arg| try v.printed(scope, fun, arg);
                },
                .function => {
                    if (fun != null) return error.Unsupported;
                },
                .math_import => if (fun != null) return error.Unsupported,
                .raise => {},
            }
            if (s.tag != .function) {
                try v.block(scope, fun, s.body);
                try v.block(scope, fun, s.otherwise);
            }
        }
    }

    fn acyclic(v: *Validator, id: usize, active: u32) Error!void {
        const bit = @as(u32, 1) << @intCast(id);
        if (active & bit != 0) return error.Unsupported;
        if (v.checked & bit != 0) return;
        for (v.p.functions.items, 0..) |_, next| {
            if (v.edges[id] & (@as(u32, 1) << @intCast(next)) != 0) try v.acyclic(next, active | bit);
        }
        v.checked |= bit;
    }
};

const Runtime = struct {
    p: *ast.Program,
    globals: *Frame,
    fuel: usize = 2_000_000,
    iterations: usize = 0,
    depth: usize = 0,
    active: u32 = 0,
    declared: [32]bool = @splat(false),
    gcd_imported: bool = false,
    estimated_steps: usize = 200,

    // The caller applies its policy limit; existing fuel and iteration bounds still contain the proof.
    fn charge(r: *Runtime, steps: usize) void {
        r.estimated_steps +|= steps;
    }

    fn output(r: *Runtime, bytes: usize) void {
        r.charge(20 +| (bytes *| 10));
    }

    fn printedScalar(r: *Runtime, value: Scalar) void {
        if (value.boolean) {
            r.output(if (value.n == 0) 5 else 4);
            return;
        }
        var magnitude = @abs(value.n);
        var digits: usize = 1 + @as(usize, @intFromBool(value.n < 0));
        while (magnitude >= 10) : (magnitude /= 10) digits += 1;
        r.charge(20);
        r.output(digits);
    }

    fn tick(r: *Runtime) Error!void {
        if (r.iterations >= 10_000) return error.Unsupported;
        r.iterations += 1;
        r.charge(20);
    }

    fn get(r: *Runtime, frame: *Frame, name: []const u8, gen: ?*const Binding) Error!Value {
        var binding = gen;
        while (binding) |g| : (binding = g.parent) if (std.mem.eql(u8, g.name, name))
            return .{ .scalar = g.value };
        if (frame.scope.find(name)) |id| {
            if (!frame.defined[id]) return error.Unsupported;
            return frame.values[id];
        }
        if (frame != r.globals) return r.get(r.globals, name, null);
        return error.Unsupported;
    }

    fn put(frame: *Frame, name: []const u8, value: Value) Error!void {
        const id = frame.scope.find(name) orelse return error.Unsupported;
        frame.values[id] = value;
        frame.defined[id] = true;
    }

    fn scalar(r: *Runtime, frame: *Frame, id: usize, gen: ?*const Binding) Error!Scalar {
        return switch (try r.expr(frame, id, gen)) {
            .scalar => |s| s,
            else => error.Unsupported,
        };
    }

    fn list(r: *Runtime, frame: *Frame, id: usize, gen: ?*const Binding) Error!*List {
        return switch (try r.expr(frame, id, gen)) {
            .list => |l| l,
            else => error.Unsupported,
        };
    }

    fn expr(r: *Runtime, frame: *Frame, id: usize, gen: ?*const Binding) Error!Value {
        if (r.fuel == 0) return error.Unsupported;
        r.fuel -= 1;
        r.charge(10);
        const e = r.p.nodes.items[id];
        switch (e.tag) {
            .integer, .boolean => return .{ .scalar = try number(e.value, e.tag == .boolean) },
            .name => return r.get(frame, e.text, gen),
            .string, .format => return error.Unsupported,
            .list => {
                const l = try r.p.a.create(List);
                l.* = .{};
                r.charge(4 + e.args.len * 6);
                for (e.args) |arg| {
                    if (l.len >= 128) return error.Unsupported;
                    l.items[l.len] = try r.scalar(frame, arg, gen);
                    l.len += 1;
                }
                return .{ .list = l };
            },
            .index => {
                const l = try r.list(frame, e.lhs, gen);
                var i = (try r.scalar(frame, e.rhs, gen)).n;
                if (i < 0) i += @intCast(l.len);
                if (i < 0 or i >= @as(i64, @intCast(l.len))) return error.Unsupported;
                r.charge(6);
                return .{ .scalar = l.items[@intCast(i)] };
            },
            .call => return .{ .scalar = try r.call(frame, e, gen) },
            .sum_filtered => return .{ .scalar = try r.sumFiltered(frame, e, gen) },
            .all_range => {
                const bounds = try r.range(frame, e.rhs, gen);
                var i = bounds.start;
                while (if (bounds.step > 0) i < bounds.stop else i > bounds.stop) : (i += bounds.step) {
                    try r.tick();
                    const local: Binding = .{ .name = e.text, .value = try number(i, false), .parent = gen };
                    if ((try r.scalar(frame, e.lhs, &local)).n == 0)
                        return .{ .scalar = try number(0, true) };
                }
                return .{ .scalar = try number(1, true) };
            },
            else => {},
        }
        const x = try r.scalar(frame, e.lhs, gen);
        if (e.tag == .reverse_digits) {
            if (x.boolean or x.n < 0) return error.Unsupported;
            r.charge(20);
            var remaining = x.n;
            var reversed: i64 = 0;
            while (remaining != 0) : (remaining = @divTrunc(remaining, 10)) {
                r.charge(40);
                reversed = (try number(reversed * 10 + @mod(remaining, 10), false)).n;
            }
            return .{ .scalar = try number(reversed, false) };
        }
        if (e.tag == .neg) return .{ .scalar = try number(-x.n, false) };
        if (e.tag == .logical_not) return .{ .scalar = try number(@intFromBool(x.n == 0), true) };
        if (e.tag == .logical_and and x.n == 0) return .{ .scalar = x };
        if (e.tag == .logical_or and x.n != 0) return .{ .scalar = x };
        const y = try r.scalar(frame, e.rhs, gen);
        const result: Scalar = switch (e.tag) {
            .add => try number(x.n + y.n, false),
            .sub => try number(x.n - y.n, false),
            .mul => try number(x.n * y.n, false),
            .floor, .modulo => blk: {
                r.charge(8);
                if (y.n == 0 or (x.n == std.math.minInt(i32) and y.n == -1)) return error.Unsupported;
                const quotient = @divFloor(x.n, y.n);
                break :blk try number(if (e.tag == .floor) quotient else x.n - quotient * y.n, false);
            },
            .eq => try number(@intFromBool(x.n == y.n), true),
            .ne => try number(@intFromBool(x.n != y.n), true),
            .lt => try number(@intFromBool(x.n < y.n), true),
            .le => try number(@intFromBool(x.n <= y.n), true),
            .gt => try number(@intFromBool(x.n > y.n), true),
            .ge => try number(@intFromBool(x.n >= y.n), true),
            .logical_and, .logical_or => y,
            else => return error.Unsupported,
        };
        return .{ .scalar = result };
    }

    fn sumFiltered(r: *Runtime, frame: *Frame, e: ast.Expr, gen: ?*const Binding) Error!Scalar {
        r.charge(20);
        const seq = r.p.nodes.items[e.rhs];
        const is_range = seq.tag == .call and std.mem.eql(u8, seq.text, "range");
        const l = if (is_range) null else try r.list(frame, e.rhs, gen);
        if (seq.tag == .list) r.charge(128 * 6);
        const bounds = if (is_range) try r.range(frame, e.rhs, gen) else Range{ .stop = @intCast(l.?.len) };
        var sum: i64 = 0;
        var i = bounds.start;
        while (if (bounds.step > 0) i < bounds.stop else i > bounds.stop) : (i += bounds.step) {
            try r.tick();
            const value = if (l) |items| items.items[@intCast(i)] else try number(i, false);
            const local: Binding = .{ .name = e.text, .value = value, .parent = gen };
            if (e.filter) |filter| if ((try r.scalar(frame, filter, &local)).n == 0) continue;
            r.charge(12);
            sum = (try number(sum + (try r.scalar(frame, e.lhs, &local)).n, false)).n;
        }
        return number(sum, false);
    }

    fn range(r: *Runtime, frame: *Frame, id: usize, gen: ?*const Binding) Error!Range {
        r.charge(20);
        const e = r.p.nodes.items[id];
        if (e.tag != .call or !std.mem.eql(u8, e.text, "range") or e.args.len < 1 or e.args.len > 3)
            return error.Unsupported;
        var args: [3]i64 = .{ 0, 0, 1 };
        for (e.args, 0..) |arg, i| args[i] = (try r.scalar(frame, arg, gen)).n;
        const result: Range = if (e.args.len == 1) .{ .stop = args[0] } else .{ .start = args[0], .stop = args[1], .step = if (e.args.len == 3) args[2] else 1 };
        if (result.step == 0) return error.Unsupported;
        return result;
    }

    fn call(r: *Runtime, frame: *Frame, e: ast.Expr, gen: ?*const Binding) Error!Scalar {
        r.charge(20);
        if (std.mem.eql(u8, e.text, "len")) return number(@intCast((try r.list(frame, e.args[0], gen)).len), false);
        if (std.mem.eql(u8, e.text, "sum")) {
            const l = try r.list(frame, e.args[0], gen);
            var sum: i64 = 0;
            for (l.items[0..l.len]) |item| {
                r.charge(12);
                sum = (try number(sum + item.n, false)).n;
            }
            return number(sum, false);
        }
        if (std.mem.eql(u8, e.text, "gcd")) {
            if (!r.gcd_imported) return error.Unsupported;
            const x = (try r.scalar(frame, e.args[0], gen)).n;
            const y = (try r.scalar(frame, e.args[1], gen)).n;
            if (x == std.math.minInt(i32) or y == std.math.minInt(i32)) return error.Unsupported;
            var a = if (x < 0) -x else x;
            var b = if (y < 0) -y else y;
            while (b != 0) {
                try r.tick();
                r.charge(10);
                const rem = @mod(a, b);
                a = b;
                b = rem;
            }
            return number(a, false);
        }
        const id = r.p.functionId(e.text) orelse return error.Unsupported;
        const mask = @as(u32, 1) << @intCast(id);
        if (!r.declared[id] or r.depth >= 8 or r.active & mask != 0) return error.Unsupported;
        const f = &r.p.functions.items[id];
        var local: Frame = .{ .scope = &f.scope, .function = id };
        for (f.params, e.args) |param, arg| {
            if (r.p.nodes.items[arg].tag == .list) r.charge(128 * 6);
            try put(&local, param, try r.expr(frame, arg, gen));
        }
        r.depth += 1;
        r.active |= mask;
        defer {
            r.depth -= 1;
            r.active &= ~mask;
        }
        return try r.block(&local, f.body) orelse error.Unsupported;
    }

    fn printed(r: *Runtime, frame: *Frame, id: usize) Error!void {
        const e = r.p.nodes.items[id];
        if (e.tag == .string) {
            r.charge(10);
            r.output(e.text.len);
            return;
        }
        if (e.tag == .format) {
            r.charge(10);
            for (e.args) |arg| try r.printed(frame, arg);
        } else r.printedScalar(try r.scalar(frame, id, null));
    }

    fn block(r: *Runtime, frame: *Frame, body: []const ast.Stmt) Error!?Scalar {
        for (body) |s| {
            if (s.tag != .function and s.tag != .math_import) r.charge(4);
            switch (s.tag) {
                .assign => {
                    const value = try r.expr(frame, s.expr, null);
                    if (value == .list) r.charge(4 + value.list.len * 10);
                    try put(frame, s.name, value);
                },
                .append => {
                    const l = switch (try r.get(frame, s.name, null)) {
                        .list => |l| l,
                        else => return error.Unsupported,
                    };
                    const value = try r.scalar(frame, s.expr, null);
                    if (l.len >= 128) return error.Unsupported;
                    r.charge(12);
                    l.items[l.len] = value;
                    l.len += 1;
                },
                .conditional => {
                    const branch = if ((try r.scalar(frame, s.expr, null)).n != 0) s.body else s.otherwise;
                    if (try r.block(frame, branch)) |value| return value;
                },
                .loop => {
                    while ((try r.scalar(frame, s.expr, null)).n != 0) {
                        try r.tick();
                        if (try r.block(frame, s.body)) |value| return value;
                    }
                },
                .each => {
                    const e = r.p.nodes.items[s.expr];
                    if (e.tag == .list) {
                        const l = try r.list(frame, s.expr, null);
                        for (l.items[0..l.len]) |item| {
                            try r.tick();
                            try put(frame, s.name, .{ .scalar = item });
                            if (try r.block(frame, s.body)) |value| return value;
                        }
                    } else {
                        const bounds = try r.range(frame, s.expr, null);
                        var i = bounds.start;
                        while (if (bounds.step > 0) i < bounds.stop else i > bounds.stop) : (i += bounds.step) {
                            try r.tick();
                            try put(frame, s.name, .{ .scalar = try number(i, false) });
                            if (try r.block(frame, s.body)) |value| return value;
                        }
                    }
                },
                .function => r.declared[s.function] = true,
                .ret => return try r.scalar(frame, s.expr, null),
                .print => {
                    for (s.args, 0..) |arg, i| {
                        if (i != 0) r.output(1);
                        try r.printed(frame, arg);
                    }
                    r.output(1);
                },
                .assertion => {
                    r.charge(4);
                    if ((try r.scalar(frame, s.expr, null)).n == 0) return error.Unsupported;
                },
                .raise => return error.Unsupported,
                .math_import => r.gcd_imported = true,
            }
        }
        return null;
    }
};

fn number(n: i64, boolean: bool) Error!Scalar {
    if (n < std.math.minInt(i32) or n > std.math.maxInt(i32)) return error.Unsupported;
    return .{ .n = n, .boolean = boolean };
}
