//! A family's binding of the in-model runner: the knob prefix it boots from, generic TF_RUNNER* or its own older names.
const std = @import("std");
const runner = @import("runner.zig");
const log = std.log.scoped(.runner);
const quiet_max = @import("runner_scan_quiet.zig").max;

/// What the runner must know of a family's chat markup, taken from its chat template and tokenizer.
pub const Family = struct {
    quiet: []const [2][]const u8 = &.{}, // opener and closer of text no cell is read in: tool calls, tool results
    never: []const []const u8 = &.{}, // markup an owned result may not spell (it would read as a call or a think end)
    trust: ?[]const u8 = null, // the compute default's guide, measured on this family; null: none yet, chats get none

    /// Whether `out` spells any of the family's markup: such a result is the model's to write, never owned.
    pub fn unsafe(f: Family, out: []const u8) bool {
        for (f.never) |m| if (std.mem.indexOf(u8, out, m) != null) return true;
        return false;
    }

    /// Marks `res` abstained (`markup`) when its output spells the family's markup; true when it did.
    pub fn screen(f: Family, res: *@import("runner_api.zig").Result) bool {
        if (!f.unsafe(res.out)) return false;
        res.status = .abstain;
        res.abstain = .markup;
        return true;
    }
};

pub const Binding = struct {
    prefix: []const u8 = "TF_RUNNER", // PREFIX=1 enables; PREFIX_CIRCUIT, PREFIX_FORMS, ... (runner.Config.parse)
    alias: ?[]const u8 = null, // the family's own older names for the same knobs: a boot switches one or the other on
    family: Family = .{},

    /// The runner's config under the prefix whose switch the boot set; both switches set is error.BadKnob.
    pub fn parse(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!?runner.Config {
        // by the switch alone: TF_RUNNER_CIRCUIT, _CLANG and _LLD also name tools for tests and image checks
        const old = b.alias orelse return b.of(try runner.Config.parse(get, b.prefix));
        if (!set(get(old))) return b.of(try runner.Config.parse(get, b.prefix));
        if (set(get(b.prefix))) {
            log.warn("turn the in-model runner on with {s} or {s}, not both", .{ b.prefix, old });
            return error.BadKnob;
        }
        return b.of(try runner.Config.parse(get, old));
    }

    fn of(comptime b: Binding, parsed: ?runner.Config) ?runner.Config {
        comptime for (b.family.quiet) |s| std.debug.assert(s[0].len > 0 and s[0].len <= quiet_max and s[1].len <= quiet_max);
        var c = parsed orelse return null;
        c.family = b.family;
        c.scan.quiet = b.family.quiet;
        return c;
    }

    /// The prefix a boot's knobs are read under: the alias when its switch is set, else the generic one.
    pub fn active(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) []const u8 {
        const old = b.alias orelse return b.prefix;
        return if (set(get(old))) old else b.prefix;
    }

    /// PREFIX_TRACE=1 (stderr) or a path: rounds around each runner fire (lanes round_trace); PREFIX_TRACE_ROUNDS: 12.
    pub fn openRoundTrace(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!void {
        if (b.alias) |old| if (set(get(old))) return roundTrace(old, get);
        return roundTrace(b.prefix, get);
    }
};

fn roundTrace(comptime p: []const u8, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!void {
    const path = get(p ++ "_TRACE") orelse return;
    if (path.len == 0 or std.mem.eql(u8, path, "0")) return;
    const rounds = if (get(p ++ "_TRACE_ROUNDS")) |r| std.fmt.parseInt(u32, r, 10) catch {
        log.warn("{s}_TRACE_ROUNDS={s}: not a count of rounds", .{ p, r });
        return error.BadKnob;
    } else 12;
    var z: [std.fs.max_path_bytes:0]u8 = undefined;
    const fd: c_int = if (std.mem.eql(u8, path, "1")) 2 else if (path.len < z.len) blk: {
        @memcpy(z[0..path.len], path);
        z[path.len] = 0;
        break :blk std.c.open(z[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    } else -1;
    if (fd < 0) return log.warn("{s}_TRACE: cannot open {s}", .{ p, path });
    @import("lanes").round_trace.open(fd, rounds, null);
    log.info("in-model runner: round trace, {d} rounds around each fire to {s}", .{ rounds, if (fd == 2) "stderr" else path });
}

fn set(v: ?[]const u8) bool {
    return if (v) |s| s.len > 0 else false;
}

/// DeepSeek-V4.1: the generic names, or TF_DSV41_RUNNER* as before; AB3's tool-full guide is its trust guide.
pub const dsv41: Binding = .{ .alias = "TF_DSV41_RUNNER", .family = .{ .trust = "Think and compute in small chunks. As soon as a sub-computation is needed, write its complete expression as `<expr> =` and continue from the returned value. Do not finish a manual trace before requesting it. For a short integer program, use a Python fenced cell with print, then `Result:`. Unsupported cells leave your continuation. Verify the setup, program semantics and non-computational reasoning. Engine values after `=` or `Result:` are computed exactly; use as given; do not re-derive or re-check by hand." } };

const t = std.testing;

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "runner binding: the generic names or the family's alias, never both switches" {
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_CIRCUIT", "/c.tfpc" }, .{ "TF_RUNNER_CIRCUIT", "/test.tfpc" } };
    const old = (try dsv41.parse(&Env.get)).?;
    try t.expectEqualStrings("/c.tfpc", old.circuit);
    try t.expectEqualStrings("TF_DSV41_RUNNER", dsv41.active(&Env.get));
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/g.tfpc" }, .{ "TF_RUNNER_FORMS", "inline" } };
    const generic = (try dsv41.parse(&Env.get)).?;
    try t.expectEqualStrings("/g.tfpc", generic.circuit);
    try t.expect(generic.scan.inline_math and !generic.scan.fence);
    try t.expectEqualStrings("TF_RUNNER", dsv41.active(&Env.get));
    const plain: Binding = .{ .family = .{ .quiet = &.{.{ "<a>", "</a>" }}, .never = &.{"</a>"} } };
    const fam = (try plain.parse(&Env.get)).?;
    try t.expectEqualStrings("/g.tfpc", fam.circuit);
    try t.expectEqualStrings("</a>", fam.scan.quiet[0][1]);
    try t.expect(fam.family.unsafe("1 </a>") and !fam.family.unsafe("42") and fam.family.trust == null);
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/g.tfpc" } };
    try t.expectError(error.BadKnob, dsv41.parse(&Env.get));
    Env.pairs = &.{.{ "TF_RUNNER_CIRCUIT", "/test.tfpc" }};
    try t.expect((try dsv41.parse(&Env.get)) == null);
    try t.expect((try plain.parse(&Env.get)) == null);
}

test "runner binding: the round trace stays off unless asked, and a bad round count is refused" {
    Env.pairs = &.{.{ "TF_DSV41_RUNNER", "1" }};
    try dsv41.openRoundTrace(&Env.get);
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE", "0" } };
    try dsv41.openRoundTrace(&Env.get);
    try t.expect(!@import("lanes").round_trace.on());
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE", "1" }, .{ "TF_DSV41_RUNNER_TRACE_ROUNDS", "x" } };
    try t.expectError(error.BadKnob, dsv41.openRoundTrace(&Env.get));
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE_ROUNDS", "x" } };
    try dsv41.openRoundTrace(&Env.get);
    try t.expect(!@import("lanes").round_trace.on());
}
