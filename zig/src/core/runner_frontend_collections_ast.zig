const std = @import("std");
pub const Error = error{Unsupported} || std.mem.Allocator.Error;
pub const Id = usize;
pub const Tag = enum {
    integer,
    boolean,
    none,
    string,
    name,
    list,
    tuple,
    dict,
    set,
    call,
    method,
    index,
    slice,
    comprehension,
    conditional,
    add,
    sub,
    mul,
    floor,
    modulo,
    neg,
    pos,
    logical_not,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    contains,
    not_contains,
    is_identity,
    is_not,
    logical_and,
    logical_or,
};
pub const Node = struct {
    tag: Tag,
    value: i64 = 0,
    text: []const u8 = "",
    a: Id = 0,
    b: Id = 0,
    c: ?Id = null,
    args: []const Id = &.{},
    keywords: []const []const u8 = &.{},
};
pub const StmtTag = enum { assign, augassign, expression, print, conditional, for_loop, while_loop, function, ret, assertion, break_loop, continue_loop, pass };
pub const Stmt = struct {
    tag: StmtTag,
    targets: []const Id = &.{},
    expr: ?Id = null,
    op: Tag = .add,
    body: []const Stmt = &.{},
    otherwise: []const Stmt = &.{},
    function: usize = 0,
};
pub const Scope = struct {
    names: std.ArrayList([]const u8) = .empty,
    pub fn find(s: *const Scope, name: []const u8) ?usize {
        for (s.names.items, 0..) |old, i| if (std.mem.eql(u8, old, name)) return i;
        return null;
    }
    pub fn bind(s: *Scope, a: std.mem.Allocator, name: []const u8) Error!usize {
        if (s.find(name)) |i| return i;
        if (s.names.items.len == 64) return error.Unsupported;
        try s.names.append(a, name);
        return s.names.items.len - 1;
    }
};
pub const Function = struct { name: []const u8, params: []const []const u8, scope: Scope = .{}, body: []const Stmt = &.{} };
pub const Stats = struct { steps: usize = 0, cells: usize = 0, objects: usize = 0, chars: usize = 0, prints: usize = 0, depth: usize = 0 };
pub const Program = struct {
    a: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    functions: std.ArrayList(Function) = .empty,
    globals: Scope = .{},
    body: []const Stmt = &.{},
    stats: Stats = .{},
    pub fn node(p: *Program, n: Node) Error!Id {
        if (p.nodes.items.len == 1024) return error.Unsupported;
        try p.nodes.append(p.a, n);
        return p.nodes.items.len - 1;
    }
    pub fn functionId(p: *const Program, name: []const u8) ?usize {
        for (p.functions.items, 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return i;
        return null;
    }
};
pub fn validName(name: []const u8) bool {
    return @import("runner_frontend_python_ast.zig").validName(name);
}
