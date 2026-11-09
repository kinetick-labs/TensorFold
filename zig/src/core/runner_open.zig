//! The served in-model runner: knobs to a circuit file, its CPU backend and the frontend, for a family's engine.
const std = @import("std");
const runner = @import("runner.zig");
const circuit = @import("runner_circuit.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend_mod = @import("runner_frontend.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);

pub const Config = runner.Config;
pub const Runner = runner.Runner;
pub const Text = runner.Text;
pub const TraceAsk = runner.TraceAsk;
pub const validTraceFormat = @import("runner_trace.zig").validFormat;
pub const maxTraceRepeat = @import("runner_trace.zig").max_repeat;
pub const place = runner.place;
/// A family's knob prefix for the runner: generic TF_RUNNER* or its own alias (runner_binding.zig).
pub const bindings = @import("runner_binding.zig");
pub const Binding = bindings.Binding;

const Opened = struct {
    runner: Runner,
    circuit: *const circuit.Circuit, // own_circuit, or the circuit of the runner this one shares with
    own_circuit: circuit.Circuit = undefined,
    placer: runner.place.Placer,
    bank: *const graph.Bank = undefined, // likewise own_bank or a shared bank
    own_bank: graph.Bank = undefined,
    has_bank: bool = false,
    routed: route.Routed = undefined,
    quick: route.Quick = undefined, // cfg.sync_graph: routed's engine-thread part
    dispatch: route.Dispatch = undefined,
    threaded: std.Io.Threaded = undefined, // a configured frontend's engine-lifetime io
    configured_frontend: frontend_mod.Context = undefined,
    compiles: bool = true, // the frontend may build an ELF (clang); false under cfg.compile false
};

var loads: std.atomic.Value(u32) = .init(0);

/// Whether `r`'s frontend may build an ELF (clang), as its Config's compile asked.
pub fn compiles(r: *Runner) bool {
    const o: *Opened = @alignCast(@fieldParentPtr("runner", r));
    return o.compiles;
}

/// Circuit and graph-bank files this process has loaded (each open that shares nothing adds one or two).
pub fn loaded() u32 {
    return loads.load(.monotonic);
}

/// The runner for `cfg` (from Config.parse); lives as long as the engine. CUDA waits for its backend (DESIGN T4).
pub fn open(gpa: Allocator, cfg: Config) !*Runner {
    return openSharing(gpa, cfg, null);
}

/// open, reusing `with`'s loaded circuit (and bank) when `cfg` names the same files; backend and workers stay its own.
pub fn openSharing(gpa: Allocator, cfg: Config, with: ?*Runner) !*Runner {
    if (cfg.device != .cpu) {
        log.err("{s}_DEVICE=cuda: the CUDA circuit backend is not in this build", .{cfg.prefix});
        return error.BadKnob;
    }
    const w: ?*Opened = if (with) |r| @as(*Opened, @alignCast(@fieldParentPtr("runner", r))) else null;
    const o = try gpa.create(Opened);
    errdefer gpa.destroy(o);
    o.* = .{ .runner = undefined, .circuit = undefined, .placer = .{ .cfg = cfg.place } };
    if (w != null and std.mem.eql(u8, w.?.runner.cfg.circuit, cfg.circuit)) {
        o.circuit = w.?.circuit; // loaded and probed once, by the runner it shares with
    } else {
        o.own_circuit = circuit.Circuit.load(gpa, cfg.circuit) catch |e| {
            log.err("{s}_CIRCUIT={s}: {t}", .{ cfg.prefix, cfg.circuit, e });
            return e;
        };
        errdefer o.own_circuit.deinit(gpa);
        _ = loads.fetchAdd(1, .monotonic);
        o.circuit = &o.own_circuit;
        try probe(gpa, o.circuit);
    }
    errdefer if (o.circuit == &o.own_circuit) o.own_circuit.deinit(gpa);
    const back = try @import("runner_circuit_backend.zig").placed(gpa, o.circuit, cfg.workers, &o.placer);
    var frontend = frontend_mod.frontend();
    if (!cfg.compile or cfg.compile_steps != 10522) {
        // Empty scratch root prevents new compiler runs; S3 also refuses cached ELF when disabled.
        o.threaded = .init(gpa, .{});
        o.configured_frontend = .{ .io = o.threaded.io() };
        o.configured_frontend.compiler.step_budget = cfg.compile_steps;
        if (!cfg.compile) o.configured_frontend.compiler.scratch_root = "";
        frontend = o.configured_frontend.frontend();
        o.compiles = cfg.compile;
    }
    if (cfg.compile) {
        // the compiler kept started and the fixed runtime compiled now, so no call pays clang's start or the runtime
        var warm_io: std.Io.Threaded = .init(gpa, .{});
        defer warm_io.deinit();
        try @import("runner_frontend_split.zig").warm(gpa, warm_io.io(), .{});
    }
    var backend = back;
    // backend graph: T1's bank loads once here; pure-integer calc programs run on it, the rest on the circuit
    if (cfg.backend == .graph) {
        const t0 = nowNs();
        if (w != null and w.?.has_bank and std.mem.eql(u8, w.?.runner.cfg.graph_bank, cfg.graph_bank)) {
            o.bank = w.?.bank;
        } else {
            o.own_bank = graph.Bank.load(gpa, cfg.graph_bank) catch |e| {
                log.err("{s}_GRAPH_BANK={s}: {t}", .{ cfg.prefix, cfg.graph_bank, e });
                return e;
            };
            _ = loads.fetchAdd(1, .monotonic);
            o.bank = &o.own_bank;
        }
        o.has_bank = true;
        o.routed = .{ .inner = frontend, .bank = o.bank };
        // settling holds: graph jobs leave the engine thread's CPU as the circuit's do (unplaced, they share its pin)
        const placer: ?*runner.place.Placer = if (cfg.settle_us > 0) &o.placer else null;
        o.dispatch = .{ .gpa = gpa, .circuit = back, .graph = try graph.placed(gpa, o.bank, cfg.workers, placer), .bank = o.bank };
        frontend = o.routed.frontend();
        backend = o.dispatch.backend();
        log.info("in-model runner: graph bank {s} {s} in {d} ms", .{ cfg.graph_bank, if (o.bank == &o.own_bank) "loaded" else "shared", (nowNs() - t0) / std.time.ns_per_ms });
    }
    o.runner = .{ .gpa = gpa, .cfg = cfg, .frontend = frontend, .backend = backend, .placer = &o.placer };
    if (cfg.sync_graph) {
        o.quick = .{ .routed = &o.routed };
        o.runner.quick = o.quick.frontend();
    }
    log.info("in-model runner: circuit {s}{s}, cpu, {d} workers, {d} steps a run, backend {t}, settle {d} us{s}{s}", .{ cfg.circuit, if (o.circuit == &o.own_circuit) "" else " (shared)", cfg.workers, cfg.max_steps, cfg.backend, cfg.settle_us, if (cfg.compile) "" else ", no ELF", if (cfg.sync_graph) ", graph results settle at their trigger" else "" });
    return &o.runner;
}

/// The circuit route at boot (served compiler, this circuit, unplaced): a wrong loop value refuses the runner.
fn probe(gpa: Allocator, c: *const circuit.Circuit) !void {
    const t0 = nowNs();
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const compiler = try @import("runner_frontend_boot.zig").selfTest(gpa, threaded.io(), .{});
    const back = try @import("runner_circuit_backend.zig").backend(gpa, c, 1);
    defer @import("runner_circuit_backend.zig").destroy(back);
    var why: [256]u8 = undefined;
    const got = try @import("runner_probe.zig").run(gpa, @import("runner_frontend.zig").frontend(), back, compiler, &why);
    if (got.wrong) |w| {
        log.err("in-model runner probe: {s}; the runner refuses this circuit and compiler", .{w});
        return error.RunnerProbe;
    }
    if (!compiler) return log.info("in-model runner probe: no RV32 compiler, the circuit route abstains", .{});
    log.info("in-model runner probe: {d} probe cells print the right output on the circuit in {d} steps ({d} ms)", .{ got.ran, got.steps, (nowNs() - t0) / std.time.ns_per_ms });
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
