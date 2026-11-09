//! Stable template descriptors with once-only lazy prepared weights and validated named binding plans.
const std = @import("std");
const model = @import("steps_v1/runner_circuit_model.zig");
const binary = @import("runner_graph_binary.zig");
const packet = @import("runner_graph_packet.zig");
const lazy = @import("runner_graph_lazy.zig");
const Allocator = std.mem.Allocator;
const missing = std.math.maxInt(u32);
pub const Slots = lazy.Slots;

pub const Template = struct {
    name: []const u8,
    width: u16,
    circuit: model.Circuit,
    inputs: u8,
    outputs: u8,
    in_slots: Slots,
    out_slots: Slots,
    preparation: ?*Preparation = null,
    materialized: bool = true,

    /// Circuit fields and logical slots may be read only after this succeeds; prepared copies keep their own circuit.
    pub fn prepare(t: *const Template) !void {
        const p = t.preparation orelse return;
        if (p.ready.load(.acquire)) {
            if (t != p.owner and !t.materialized) return error.GraphUnpreparedCopy;
            return;
        }
        if (t != p.owner) return error.GraphUnpreparedCopy;
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        if (p.ready.load(.acquire)) return;
        if (p.failure) |err| return err;
        _ = p.attempts.fetchAdd(1, .monotonic);
        p.publish() catch |err| {
            p.failure = err;
            return err;
        };
    }

    pub fn isPrepared(t: *const Template) bool {
        const p = t.preparation orelse return true;
        return p.ready.load(.acquire) and (t == p.owner or t.materialized);
    }

    pub fn loadAttempts(t: *const Template) u32 {
        return if (t.preparation) |p| p.attempts.load(.monotonic) else 0;
    }
};

const Preparation = struct {
    gpa: Allocator,
    owner: *Template,
    path: []const u8,
    header: binary.Header,
    mutex: std.Io.Mutex = .init,
    ready: std.atomic.Value(bool) = .init(false),
    attempts: std.atomic.Value(u32) = .init(0),
    failure: ?anyerror = null,

    fn publish(p: *Preparation) !void {
        const t = p.owner;
        var circuit = try binary.loadPinned(p.gpa, p.path, t.name, t.width, &p.header);
        errdefer circuit.deinit(p.gpa);
        const layout = try lazy.Binding.inspect(&circuit, try binary.bindingWidth(t.width));
        if ((t.inputs != 0 and layout.inputs != t.inputs) or (t.outputs != 0 and layout.outputs != t.outputs))
            return error.GraphSlotArity;
        if (t.inputs == 0) t.inputs = layout.inputs;
        if (t.outputs == 0) t.outputs = layout.outputs;
        t.circuit = circuit;
        t.in_slots = layout.in_slots;
        t.out_slots = layout.out_slots;
        t.materialized = true;
        p.ready.store(true, .release);
    }
};

pub const LoadMode = enum { lazy, eager };

const Descriptor = struct {
    name: []const u8,
    width: u16,
    path: []const u8,
    inputs: ?u8 = null,
    outputs: ?u8 = null,
};
pub const PreparedPlan = struct {
    name: []const u8,
    inputs: u32,
    ops: []const packet.Op,
    result: u32,
    constants: []const packet.Value,
    total_values: usize,
    algorithm: ?[]const u8,
    width: u16,
    precision: u8,
};

const Manifest = struct {
    version: u32,
    templates: []const Descriptor,
    plans: []const packet.Plan = &.{},
};

pub const Bank = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    templates: []Template,
    plans: []const PreparedPlan,

    pub fn load(a: Allocator, path: []const u8) !Bank {
        return loadWithMode(a, path, .lazy);
    }

    pub fn loadEager(a: Allocator, path: []const u8) !Bank {
        return loadWithMode(a, path, .eager);
    }

    /// The allocator must support concurrent stream allocations; the bank owns all loaded circuits until deinit.
    pub fn loadWithMode(a: Allocator, path: []const u8, mode: LoadMode) !Bank {
        var threaded: std.Io.Threaded = .init_single_threaded;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, a, .limited(32 << 20));
        defer a.free(bytes);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const doc = try std.json.parseFromSliceLeaky(Manifest, arena.allocator(), bytes, .{ .allocate = .alloc_always });
        if (doc.version != 1 or doc.templates.len == 0 or doc.templates.len > 256) return error.GraphManifest;
        const templates = try arena.allocator().alloc(Template, doc.templates.len);
        const preparations = try arena.allocator().alloc(Preparation, doc.templates.len);
        const expected_schema = binary.schemaHash();
        var loaded: usize = 0;
        errdefer for (templates[0..loaded]) |*t| {
            if (t.isPrepared()) t.circuit.deinit(a);
        };
        for (doc.templates, templates, preparations) |d, *t, *preparation| {
            if (!packet.widthValid(d.width) or d.width == 255 or d.name.len == 0 or d.path.len == 0) return error.GraphTemplate;
            if (d.inputs) |n| if (n == 0 or n > packet.max_arity) return error.GraphSlotArity;
            if (d.outputs) |n| if (n == 0 or n > packet.max_arity) return error.GraphSlotArity;
            for (templates[0..loaded]) |prior| if (prior.width == d.width and std.mem.eql(u8, prior.name, d.name))
                return error.GraphDuplicateTemplate;
            const joined = if (std.fs.path.isAbsolute(d.path)) d.path else try std.fs.path.join(arena.allocator(), &.{ std.fs.path.dirname(path) orelse ".", d.path });
            const resolved = try std.Io.Dir.cwd().realPathFileAlloc(threaded.io(), joined, arena.allocator());
            const h = try binary.inspect(resolved, d.name, d.width, expected_schema);
            const legacy = binary.isLegacy(&h);
            const inputs = d.inputs orelse if (legacy) @as(u8, 0) else return error.GraphSlotArity;
            const outputs = d.outputs orelse if (legacy) @as(u8, 0) else return error.GraphSlotArity;
            if (!legacy and (inputs != h[80] or outputs != h[81])) return error.GraphSlotArity;
            preparation.* = .{ .gpa = a, .owner = t, .path = resolved, .header = h };
            t.* = .{ .name = d.name, .width = d.width, .circuit = undefined, .inputs = inputs, .outputs = outputs, .in_slots = @splat(@splat(missing)), .out_slots = @splat(@splat(missing)), .preparation = preparation, .materialized = false };
            loaded += 1;
            // Legacy headers omit bindings; validate the complete payload before exposing the descriptor.
            if (legacy) try t.prepare();
        }
        const plans = try arena.allocator().alloc(PreparedPlan, doc.plans.len);
        var bank: Bank = .{ .gpa = a, .arena = arena, .templates = templates, .plans = plans };
        for (doc.plans, 0..) |p, i| {
            if (p.name.len == 0) return error.GraphPlan;
            for (doc.plans[0..i]) |prior| if (std.mem.eql(u8, prior.name, p.name)) return error.GraphDuplicatePlan;
            if (p.inputs > packet.max_values or p.literals.len > packet.max_values - p.inputs)
                return error.GraphPlanSize;
            const constants = try arena.allocator().alloc(packet.Value, p.literals.len);
            for (p.literals, constants) |text, *v| v.* = try packet.literal(text);
            if (p.algorithm) |kind| {
                const family = std.meta.stringToEnum(packet.Algorithm, kind) orelse return error.GraphAlgorithm;
                if (p.inputs != packet.algorithmArity(family) or p.ops.len != 0 or constants.len != 0 or p.result != 0)
                    return error.GraphAlgorithm;
                try bank.validateAlgorithm(family, p.width, p.precision);
            } else if (p.precision != 0) {
                return error.GraphPrecision;
            }
            plans[i] = .{ .name = p.name, .inputs = p.inputs, .ops = p.ops, .result = p.result, .constants = constants, .total_values = if (p.algorithm != null) 32 else try bank.validate(p.inputs + p.literals.len, p.ops, p.result), .algorithm = p.algorithm, .width = p.width, .precision = p.precision };
        }
        const emitter = bank.find("emit", 62) orelse return error.GraphEmitMissing;
        if (emitter.inputs != 1 or emitter.outputs != 1) return error.GraphSlotArity;
        for (templates) |t| if (t.width == 128) {
            const wide = bank.find("emit", 128) orelse return error.GraphEmitMissing;
            if (wide.inputs != 1 or wide.outputs != 1) return error.GraphSlotArity;
            break;
        };
        for (templates) |t| if (t.width == 256) {
            try bank.require("emit", 256, 1, 1);
            try bank.require("format", 256, 2, 2);
            break;
        };
        bank.arena = arena;
        if (mode == .eager) try bank.prepareAll();
        return bank;
    }

    /// Destroy every backend and machine before freeing the bank; prepared copies borrow its circuit allocations.
    pub fn deinit(b: *Bank) void {
        for (b.templates) |*t| {
            if (t.isPrepared()) t.circuit.deinit(b.gpa);
        }
        b.arena.deinit();
        b.* = undefined;
    }

    pub fn prepareAll(b: *const Bank) !void {
        for (b.templates) |*t| try t.prepare();
    }

    pub fn preparedCount(b: *const Bank) usize {
        var count: usize = 0;
        for (b.templates) |*t| count += @intFromBool(t.isPrepared());
        return count;
    }

    pub fn loadAttempts(b: *const Bank) u64 {
        var count: u64 = 0;
        for (b.templates) |*t| count += t.loadAttempts();
        return count;
    }

    /// Clears only a cached failure; the next use retries the pinned pack and successful circuits are never replaced.
    pub fn retryFailed(b: *const Bank, name: []const u8, width: u16) !void {
        const t = b.find(name, width) orelse return error.GraphTemplate;
        const p = t.preparation orelse return error.GraphAlreadyPrepared;
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        if (p.ready.load(.acquire)) return error.GraphAlreadyPrepared;
        if (p.failure == null) return error.GraphNotFailed;
        p.failure = null;
    }

    /// Lookup reads metadata only; call Template.prepare before accessing its circuit or logical slots.
    pub fn find(b: *const Bank, name: []const u8, width: u16) ?*const Template {
        for (b.templates) |*t| if (t.width == width and std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    pub fn plan(b: *const Bank, name: []const u8) ?*const PreparedPlan {
        for (b.plans) |*p| if (std.mem.eql(u8, p.name, name)) return p;
        return null;
    }

    pub fn validate(b: *const Bank, inputs: usize, ops: []const packet.Op, result: u32) !usize {
        if (inputs > packet.max_values or ops.len > packet.max_ops) return error.GraphPlanSize;
        var count = inputs;
        for (ops) |op| {
            if (op.outputs == 0 or op.outputs > packet.max_arity or count + op.outputs > packet.max_values)
                return error.GraphPlanSize;
            if (std.mem.eql(u8, op.op, "bind")) {
                if (op.outputs != 1 or op.copies.len > 128 or op.args != null or op.b != null or op.precision != 0)
                    return error.GraphBinding;
                for (op.copies) |c| if (c.ref >= count or c.from > packet.max_digits or c.to > packet.max_digits or
                    c.digits > packet.max_digits - c.from or c.digits > packet.max_digits - c.to)
                    return error.GraphBinding;
                if (op.flags) |r| if (r >= count) return error.GraphReference;
            } else {
                if (std.mem.eql(u8, op.op, "factorize") or std.mem.eql(u8, op.op, "f64_repr")) return error.GraphAlgorithm;
                if (std.mem.startsWith(u8, op.op, "emit") or (std.mem.eql(u8, op.op, "format") or std.mem.eql(u8, op.op, "format_f64")) or op.copies.len != 0 or op.flags != null)
                    return error.GraphTemplate;
                const arity: u8 = if (std.meta.stringToEnum(packet.Algorithm, op.op)) |family| blk: {
                    try b.validateAlgorithm(family, op.width, op.precision);
                    if (op.outputs != packet.algorithmOutputs(family)) return error.GraphSlotArity;
                    break :blk packet.algorithmArity(family);
                } else blk: {
                    if (op.precision != 0) return error.GraphPrecision;
                    const t = b.find(op.op, op.width) orelse return error.GraphTemplate;
                    if (t.outputs != op.outputs) return error.GraphSlotArity;
                    break :blk t.inputs;
                };
                if (op.args) |args| {
                    if (args.len != arity) return error.GraphSlotArity;
                    for (args) |r| if (r >= count) return error.GraphReference;
                } else {
                    if (arity != 1 + @as(usize, @intFromBool(op.b != null)) or op.a >= count)
                        return error.GraphSlotArity;
                    if (op.b) |r| if (r >= count) return error.GraphReference;
                }
            }
            count += op.outputs;
        }
        if (result >= count) return error.GraphReference;
        return count;
    }

    pub fn validatePrints(b: *const Bank, prints: []const packet.Print, count: usize) !void {
        if (prints.len > packet.max_prints) return error.GraphPrint;
        if (prints.len == 0) return;
        try b.require("format", 62, 2, 2);
        try b.require("format", 128, 2, 2);
        for (prints) |p| {
            if (p.ref >= count or p.precision > 18 or (p.style != .fixed and p.precision != 0)) return error.GraphPrint;
            if (p.style == .byte) {
                if (p.precision != 0 or p.prefix or p.ending != .empty or p.second != null) return error.GraphPrint;
                try b.require("byte_check", 128, 1, 1);
                try b.require("emit_byte", 4, 1, 1);
                continue;
            }
            if (p.style == .prepared) {
                try b.require("format_f64", 62, 2, 2);
                if (p.prefix or p.precision != 0 or p.ending != .newline or p.second == null) return error.GraphPrint;
            }
            if ((p.style == .fraction or p.style == .tuple or p.style == .prepared) != (p.second != null)) return error.GraphPrint;
            if (p.second) |ref| if (ref >= count) return error.GraphReference;
            if (p.style == .fraction) try b.require("meta", 128, 2, 3);
            if (p.style == .tuple and (p.prefix or (p.ending != .newline and p.ending != .space and p.ending != .empty and p.ending != .comma_space)))
                return error.GraphPrint;
        }
    }

    fn require(b: *const Bank, name: []const u8, width: u16, inputs: u8, outputs: u8) !void {
        const t = b.find(name, width) orelse return error.GraphTemplate;
        if (t.inputs != inputs or t.outputs != outputs) return error.GraphSlotArity;
    }

    fn validateAlgorithm256(b: *const Bank, family: packet.Algorithm) !void {
        if (family != .divmod and family != .powmod) return error.GraphAlgorithm;
        try b.require("meta", 256, 2, 3);
        try b.require("divstep", 256, 3, 2);
        try b.require("divfinal", 256, 4, 2);
        if (family == .divmod) {
            try b.require("emit_first", 256, 1, 1);
            try b.require("emit_last", 256, 1, 1);
        } else {
            try b.require("half", 256, 1, 2);
            try b.require("mulmodstep", 256, 4, 1);
            try b.require("sub", 256, 2, 1);
            try b.require("emit", 256, 1, 1);
        }
    }

    fn validateAlgorithm(b: *const Bank, family: packet.Algorithm, width: u16, precision: u8) !void {
        if (!packet.algorithmWidthValid(family, width) or precision > 18 or (family != .fixed_div and precision != 0))
            return error.GraphAlgorithm;
        if (width == 256) return b.validateAlgorithm256(family);
        if (std.mem.startsWith(u8, @tagName(family), "f64_")) {
            if (width != 128) return error.GraphAlgorithm;
            try b.require("raw_meta", 128, 2, 3);
            try b.require("format_f64", 62, 2, 2);
            try b.require("decimal_mul", 128, 2, 1);
            try b.require("compare", 128, 2, 1);
            try b.require("round_div", 128, 4, 1);
            try b.require("add", 128, 2, 1);
            try b.require("sub", 128, 2, 1);
            try b.require("half", 128, 1, 2);
            try b.require("divstep", 128, 3, 2);
            try b.require("format", 62, 2, 2);
            return;
        }
        const factor = family == .smallest_factor or family == .factorize;
        const work_width = if (factor) @max(width, 8) else width;
        if (factor) {
            try b.require("add", work_width, 2, 1);
            try b.require("half", work_width, 1, 2);
            try b.require("divstep", work_width, 3, 2);
            try b.require("meta", work_width, 2, 3);
            try b.require("format", 62, 2, 2);
            try b.require("format", 128, 2, 2);
            return;
        }
        const kernels = [_]struct { []const u8, u8, u8 }{
            .{ "add", 2, 1 },        .{ "sub", 2, 1 },      .{ "mul", 2, 1 },  .{ "half", 1, 2 },
            .{ "divstep", 3, 2 },    .{ "divfinal", 4, 2 }, .{ "meta", 2, 3 }, .{ "digit_sum", 1, 1 },
            .{ "mulmodstep", 4, 1 },
        };
        for (kernels) |k| try b.require(k[0], if (std.mem.eql(u8, k[0], "meta")) @max(work_width, 8) else work_width, k[1], k[2]);
        if (family == .fixed_div) {
            try b.require("scale10", width, 1, 1);
        }
        if (family == .fixed_div or family == .divexact) try b.require("round_div", width, 4, 1);
        if (family == .fraction or family == .fixed_div or factor) {
            try b.require("format", 62, 2, 2);
            try b.require("format", 128, 2, 2);
        }
        if (family == .fraction) try b.require("meta", 128, 2, 3);
        if (family == .divmod) for ([_][]const u8{ "emit_first", "emit_last" }) |name|
            try b.require(name, @max(width, 62), 1, 1);
    }
};
