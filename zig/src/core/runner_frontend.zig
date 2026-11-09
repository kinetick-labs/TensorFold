const std = @import("std");
const api = @import("runner_api.zig");
const calc = @import("runner_frontend_calc.zig");
const program = @import("runner_frontend_program.zig");
const inline_math = @import("runner_frontend_inline.zig");
const integer = @import("runner_frontend_integer.zig");
const integer_python = @import("runner_frontend_integer_python.zig");
const blocks = @import("runner_frontend_python_blocks.zig");
const firmware_cost = @import("runner_frontend_value.zig");

pub const Frontend = api.Frontend;
pub const Cell = api.Cell;
pub const Translation = api.Translation;
pub const Context = struct {
    io: std.Io,
    compiler: program.Config = .{ .clang = "/usr/bin/clang" },
    exact_tags: bool = false,
    exact_inline: bool = true,

    pub fn frontend(self: *Context) Frontend {
        return .{ .ctx = self, .translate = translate };
    }
};

pub fn frontend() Frontend {
    return .{ .translate = translate };
}

// EXACT binds by form: inline defaults on; explicit calc/exact code tags default off.
fn language(lang: []const u8) enum { python, exact, c, unsupported } {
    if (std.mem.eql(u8, lang, "python") or std.mem.eql(u8, lang, "python3") or std.mem.eql(u8, lang, "py"))
        return .python;
    if (std.mem.eql(u8, lang, "calc") or std.mem.eql(u8, lang, "exact")) return .exact;
    if (std.mem.eql(u8, lang, "c")) return .c;
    return .unsupported;
}

fn translate(raw: ?*anyopaque, a: std.mem.Allocator, cell: Cell) std.mem.Allocator.Error!Translation {
    if (raw == null) {
        var threaded = std.Io.Threaded.init(a, .{});
        defer threaded.deinit();
        var context: Context = .{ .io = threaded.io() };
        return translate(&context, a, cell);
    }
    const original: *Context = @ptrCast(@alignCast(raw.?));
    var context = original.*;
    const end = std.Io.Clock.awake.now(context.io).addDuration(.fromMilliseconds(1900));
    if (context.compiler.deadline == null or end.toNanoseconds() < context.compiler.deadline.?.toNanoseconds())
        context.compiler.deadline = end;
    return translateImpl(&context, a, cell);
}

fn translateImpl(raw: ?*anyopaque, a: std.mem.Allocator, cell: Cell) std.mem.Allocator.Error!Translation {
    const context: ?*Context = if (raw) |ptr| @ptrCast(@alignCast(ptr)) else null;
    const lang = language(cell.lang);
    if (lang == .unsupported) return .{ .abstain = .language };
    if (lang == .exact and (context == null or !context.?.exact_tags)) return .{ .abstain = .unsupported };
    if (cell.code.len > 4096) return .{ .abstain = .too_long };
    if (std.mem.indexOfScalar(u8, cell.code, 0) != null) return .{ .abstain = .syntax };
    for (cell.prior) |prior| if (std.mem.indexOfScalar(u8, prior, 0) != null) return .{ .abstain = .syntax };
    if (cell.form == .inline_expr) {
        if (lang != .python) return .{ .abstain = .unsupported };
        if (try integer.ambiguousPercent(a, cell.code)) return .{ .abstain = .unsupported };
        const source = std.mem.trim(u8, cell.code, " \t\r\n");
        if (std.mem.startsWith(u8, source, "print") or std.mem.startsWith(u8, source, ">>>"))
            return .{ .abstain = .unsupported };
        if (try inline_math.lower(a, cell.code, if (context) |ctx| ctx.exact_inline else true)) |bytes|
            return calcProgram(a, context, bytes);
        return integerProgram(a, context, cell.code, false);
    }
    if (cell.prior.len != 0 and (lang != .python or !try safePrior(a, cell.prior)))
        return .{ .abstain = .unsupported };
    if (lang == .python or lang == .exact) {
        const exact_only = std.mem.indexOfScalar(u8, cell.code, '.') != null or
            containsDivision(cell.code) or std.mem.indexOf(u8, cell.code, "gcd") != null;
        if (!exact_only or lang == .exact) {
            if (try calc.lower(a, cell.code)) |bytes| return calcProgram(a, context, bytes);
        }
    }
    if (lang == .python) {
        if (try integer_python.lowerCalc(a, cell.code)) |bytes| {
            const result = calcProgram(a, context, bytes);
            if (result == .program) return result;
        }
        const result = try integerProgram(a, context, cell.code, true);
        if (result == .program) return result;
        const budget = if (context) |ctx| ctx.compiler.step_budget else 10522;
        const decimal = try @import("runner_frontend_decimal.zig").lowerWithCost(a, cell.code, budget);
        if (decimal) |lowered| return compileNumeric(a, context.?, lowered);
        const fraction = try @import("runner_frontend_fraction.zig").lowerWithCost(a, cell.code, budget);
        if (fraction) |lowered| return compileNumeric(a, context.?, lowered);
    }
    if (lang == .exact) return .{ .abstain = .unsupported };
    var source: []const u8 = cell.code;
    var owned: ?[]u8 = null;
    var structured = false;
    var collections_mode = false;
    var estimated_steps: ?usize = null;
    defer if (owned) |bytes| a.free(bytes);
    if (lang == .python) {
        if (try program.lowerWithCost(a, cell.code)) |body| {
            defer a.free(body.source);
            owned = try program.wrapBody(a, body.source);
            estimated_steps = body.estimated_steps;
        } else {
            if (try blocks.lowerWithCost(a, cell.code)) |body| {
                owned = body.source;
                estimated_steps = body.estimated_steps;
            } else if (try @import("runner_frontend_collections.zig").lowerWithCost(a, cell.code)) |body| {
                owned = body.source;
                estimated_steps = body.estimated_steps;
                collections_mode = true;
            } else return .{ .abstain = .unsupported };
            structured = true;
        }
        source = owned.?;
    } else if (!program.compiler.hasMain(source)) {
        owned = try program.wrapBody(a, source);
        source = owned.?;
    }
    const bytes = if (context) |ctx| blk: {
        var settings = ctx.compiler;
        if (structured) settings.stack_limit = 4096;
        settings.collection_stack = collections_mode;
        if (collections_mode and settings.optimize != .O0) settings.optimize = .O1;
        settings.opaque_inputs = lang == .python;
        settings.estimated_steps = estimated_steps;
        break :blk try program.compile(a, ctx.io, source, settings);
    } else blk: {
        var threaded = std.Io.Threaded.init(a, .{});
        defer threaded.deinit();
        break :blk try program.compile(a, threaded.io(), source, .{
            .clang = "/usr/bin/clang",
            .stack_limit = if (structured) 4096 else 0,
            .collection_stack = collections_mode,
            .opaque_inputs = lang == .python,
            .estimated_steps = estimated_steps,
        });
    };
    if (bytes) |elf| return .{ .program = .{ .kind = .elf, .bytes = elf } };
    return .{ .abstain = .unsupported };
}

fn compileNumeric(a: std.mem.Allocator, ctx: *Context, lowered: integer.Lowered) std.mem.Allocator.Error!Translation {
    defer a.free(lowered.source);
    var settings = ctx.compiler;
    settings.opaque_inputs = true;
    settings.estimated_steps = lowered.estimated_steps;
    if (settings.optimize != .O0) settings.optimize = .O2;
    const bytes = try program.compile(a, ctx.io, lowered.source, settings) orelse return .{ .abstain = .unsupported };
    return .{ .program = .{ .kind = .elf, .bytes = bytes } };
}

fn calcProgram(a: std.mem.Allocator, context: ?*Context, bytes: []u8) Translation {
    const budget = if (context) |ctx| ctx.compiler.step_budget else 10522;
    if (budget != 0 and firmware_cost.estimate(bytes, budget) == null) {
        a.free(bytes);
        return .{ .abstain = .unsupported };
    }
    return .{ .program = .{ .kind = .calc, .bytes = bytes } };
}

fn integerProgram(a: std.mem.Allocator, context: ?*Context, code: []const u8, python: bool) std.mem.Allocator.Error!Translation {
    const bytes = if (context) |ctx|
        if (python) try integer_python.compile(a, ctx.io, code, ctx.compiler) else try integer.compile(a, ctx.io, code, ctx.compiler)
    else blk: {
        var threaded = std.Io.Threaded.init(a, .{});
        defer threaded.deinit();
        break :blk if (python) try integer_python.compile(a, threaded.io(), code, .{}) else try integer.compile(a, threaded.io(), code, .{});
    };
    return if (bytes) |elf| .{ .program = .{ .kind = .elf, .bytes = elf } } else .{ .abstain = .unsupported };
}

fn safePrior(a: std.mem.Allocator, prior: []const []const u8) std.mem.Allocator.Error!bool {
    if (prior.len > 8) return false;
    var total: usize = 0;
    for (prior) |code| {
        total += code.len;
        if (total > 4096) return false;
        if (try integer_python.lower(a, code)) |source| {
            a.free(source);
            continue;
        }
        const with_output = try std.fmt.allocPrint(a, "{s}\nprint(0)", .{code});
        defer a.free(with_output);
        const source = try integer_python.lower(a, with_output) orelse return false;
        a.free(source);
    }
    return true;
}

fn containsDivision(code: []const u8) bool {
    var i: usize = 0;
    while (i < code.len) : (i += 1) {
        if (code[i] == '/') {
            if (i + 1 == code.len or code[i + 1] != '/') return true;
            i += 1;
        }
    }
    return false;
}
