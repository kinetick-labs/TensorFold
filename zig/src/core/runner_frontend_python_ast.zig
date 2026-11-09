const std = @import("std");
pub const Invalid = error{Unsupported};
pub const Error = Invalid || std.mem.Allocator.Error;
pub const Tag = enum {
    integer,
    boolean,
    name,
    string,
    format,
    list,
    call,
    index,
    all_range,
    sum_filtered,
    add,
    sub,
    mul,
    floor,
    modulo,
    neg,
    reverse_digits,
    logical_not,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    logical_and,
    logical_or,
};
pub const Expr = struct {
    tag: Tag,
    value: i64 = 0,
    text: []const u8 = "",
    lhs: usize = 0,
    rhs: usize = 0,
    filter: ?usize = null,
    args: []const usize = &.{},
};
pub const Kind = enum { unknown, scalar, list };
pub const Symbol = struct { name: []const u8, kind: Kind = .unknown };
pub const Scope = struct {
    symbols: std.ArrayList(Symbol) = .empty,
    pub fn find(s: *const Scope, name: []const u8) ?usize {
        for (s.symbols.items, 0..) |sym, i| if (std.mem.eql(u8, sym.name, name)) return i;
        return null;
    }
    pub fn bind(s: *Scope, a: std.mem.Allocator, name: []const u8) Error!usize {
        if (!validName(name)) return error.Unsupported;
        if (s.find(name)) |id| return id;
        if (s.symbols.items.len >= 64) return error.Unsupported;
        try s.symbols.append(a, .{ .name = name });
        return s.symbols.items.len - 1;
    }
};
pub const StmtTag = enum { assign, append, conditional, loop, each, function, ret, print, assertion, raise, math_import };
pub const Stmt = struct {
    tag: StmtTag,
    name: []const u8 = "",
    expr: usize = 0,
    args: []const usize = &.{},
    body: []Stmt = &.{},
    otherwise: []Stmt = &.{},
    function: usize = 0,
};
pub const Function = struct {
    name: []const u8,
    params: []const []const u8,
    scope: Scope = .{},
    body: []Stmt = &.{},
};
pub const Program = struct {
    a: std.mem.Allocator,
    nodes: std.ArrayList(Expr) = .empty,
    functions: std.ArrayList(Function) = .empty,
    globals: Scope = .{},
    body: []Stmt = &.{},
    gcd_imported: bool = false,
    // Admission metadata never replaces source expressions or supplies emitted answers.
    estimated_steps: usize = 0,
    pub fn node(p: *Program, e: Expr) Error!usize {
        if (p.nodes.items.len >= 2048) return error.Unsupported;
        try p.nodes.append(p.a, e);
        return p.nodes.items.len - 1;
    }
    pub fn functionId(p: *const Program, name: []const u8) ?usize {
        for (p.functions.items, 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return i;
        return null;
    }
};
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    const reserved = [_][]const u8{
        "print",               "len",                "sum",                 "range",                     "gcd",                "all",                    "True",                   "False",                "None",             "ValueError",
        "and",                 "or",                 "not",                 "for",                       "while",              "if",                     "else",                   "elif",                 "return",           "def",
        "raise",               "class",              "global",              "nonlocal",                  "import",             "from",                   "in",                     "pass",                 "break",            "continue",
        "lambda",              "yield",              "try",                 "except",                    "finally",            "with",                   "as",                     "assert",               "del",              "is",
        "async",               "await",              "__debug__",           "__builtins__",              "__import__",         "abs",                    "any",                    "ascii",                "bin",              "bool",
        "breakpoint",          "bytearray",          "bytes",               "callable",                  "chr",                "classmethod",            "compile",                "complex",              "delattr",          "dict",
        "dir",                 "divmod",             "enumerate",           "eval",                      "exec",               "filter",                 "float",                  "format",               "frozenset",        "getattr",
        "globals",             "hasattr",            "hash",                "help",                      "hex",                "id",                     "input",                  "int",                  "isinstance",       "issubclass",
        "iter",                "list",               "locals",              "map",                       "max",                "memoryview",             "min",                    "next",                 "object",           "oct",
        "open",                "ord",                "pow",                 "property",                  "repr",               "reversed",               "round",                  "set",                  "setattr",          "slice",
        "sorted",              "staticmethod",       "str",                 "super",                     "tuple",              "type",                   "vars",                   "zip",                  "aiter",            "anext",
        "BaseException",       "BaseExceptionGroup", "Exception",           "ExceptionGroup",            "ArithmeticError",    "AssertionError",         "AttributeError",         "BufferError",          "EOFError",         "ImportError",
        "LookupError",         "MemoryError",        "NameError",           "OSError",                   "ReferenceError",     "RuntimeError",           "StopAsyncIteration",     "StopIteration",        "SyntaxError",      "SystemError",
        "SystemExit",          "TypeError",          "Warning",             "ZeroDivisionError",         "OverflowError",      "IndexError",             "KeyError",               "UnboundLocalError",    "NotImplemented",   "Ellipsis",
        "KeyboardInterrupt",   "GeneratorExit",      "NotImplementedError", "UnicodeError",              "UnicodeDecodeError", "UnicodeEncodeError",     "UnicodeTranslateError",  "RecursionError",       "IndentationError", "TabError",
        "ModuleNotFoundError", "BlockingIOError",    "BrokenPipeError",     "ChildProcessError",         "ConnectionError",    "ConnectionAbortedError", "ConnectionRefusedError", "ConnectionResetError", "FileExistsError",  "FileNotFoundError",
        "InterruptedError",    "IsADirectoryError",  "NotADirectoryError",  "PermissionError",           "ProcessLookupError", "TimeoutError",           "EnvironmentError",       "IOError",              "EncodingWarning",  "BytesWarning",
        "DeprecationWarning",  "FutureWarning",      "ImportWarning",       "PendingDeprecationWarning", "ResourceWarning",    "RuntimeWarning",         "SyntaxWarning",          "UnicodeWarning",       "UserWarning",      "copyright",
        "credits",             "exit",               "license",             "quit",
    };
    for (reserved) |r| if (std.mem.eql(u8, r, name)) return false;
    return true;
}
