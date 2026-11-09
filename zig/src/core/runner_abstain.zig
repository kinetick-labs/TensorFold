//! Abstain records (PREFIX_ABSTAINS, with CELLS): each abstained cell's source, reason, stage and decision point.
const std = @import("std");
const receipt = @import("runner_receipt.zig");
const Allocator = std.mem.Allocator;

pub const schema = 1;

/// Why a cell abstained, as one stable name; its refusal stage follows from it (`stage`).
pub const Reason = enum {
    context, // the reply's own number context (exact inline math in a Python-float reply)
    language, // a `calc` / `exact` cell with exact cells off
    unsupported,
    syntax,
    name,
    too_long,
    too_big, // the frontend's own tags (runner_api.Abstain), as it gave them
    out_of_memory, // the backend start, or keeping its result
    backend_abstain,
    budget,
    fault, // the run: its own abstain (error_name: its tag), the step budget, a machine fault
    empty_output,
    multiline_output, // an inline value that is not one line
    markup, // a result that spells the family's chat markup (runner_binding.Family.never)
    mark_value_shape, // INLINE_MARK: a blank or multi-line value cannot carry the mark
    token_cap, // the owned text would pass the token cap
    trace_shape, // a trace run that did not print one value
};

pub fn stage(r: Reason) []const u8 {
    return switch (r) {
        .context => "context",
        .language, .unsupported, .syntax, .name, .too_long, .too_big => "frontend",
        .out_of_memory => "dispatch",
        .backend_abstain, .budget, .fault => "execute",
        .empty_output, .multiline_output, .mark_value_shape, .token_cap, .trace_shape, .markup => "own",
    };
}

/// One abstained cell at its decision; the log fills the reply's coordinates there, as an owned record's.
pub const Abstain = struct {
    cell: u32,
    reason: Reason,
    error_name: ?[]const u8 = null, // the original tag (static text)
    held_ms: u64 = 0,
    steps: u64 = 0,
    content: bool = false,
    at: u64 = 0,
    token: u32 = 0,
    round: u32 = 0,
};

/// Where cell `cell`'s source closed: its field and that field's bytes through the closing token.
pub const Close = struct { cell: u32, content: bool, end: u64 };

/// Why a settled cell abstained, read off its translation and result; `hint`: what only its caller knows.
pub fn reason(c: anytype, hint: ?Reason) struct { Reason, ?[]const u8 } {
    if (hint) |h| return .{ h, null };
    switch (c.translation) {
        .abstain => |t| switch (t) {
            inline else => |x| return .{ @field(Reason, @tagName(x)), @tagName(x) },
        },
        .program => {},
    }
    const res = c.result orelse return .{ .out_of_memory, "OutOfMemory" };
    return switch (res.status) {
        .abstain => if (res.abstain == .markup) .{ .markup, "markup" } else .{ .backend_abstain, if (res.abstain) |a| @tagName(a) else null },
        .budget => .{ .budget, "budget" },
        .fault => .{ .fault, "fault" },
        .ok => if (res.out.len == 0) .{ .empty_output, null } else if (std.mem.indexOfScalar(u8, std.mem.trimEnd(u8, res.out, "\n"), '\n') != null) .{ .multiline_output, null } else .{ .mark_value_shape, null },
    };
}

/// Cell `index` abstains at its decision: told, counted, and recorded when abstain records are on.
pub fn told(s: anytype, c: anytype, index: u32, held_ms: u64, hint: ?Reason) void {
    const why = reason(c, hint);
    c.abstained = true;
    c.told = .abstained;
    s.abstained_n += 1;
    s.rec.abstained(s.r.gpa, .{ .cell = index, .reason = why[0], .error_name = why[1], .held_ms = held_ms, .steps = if (c.result) |r| r.steps else 0 });
}

/// The receipt's abstain fields after its records: `abstains`, completeness, and the runner config they ran under.
pub fn write(o: *std.ArrayList(u8), gpa: Allocator, cells: anytype, l: anytype, cfg: anytype) Allocator.Error!void {
    try o.print(gpa, ",\"abstain_schema\":{d},\"abstains\":[", .{schema});
    for (l.abstains.items, 0..) |a, i| {
        if (i > 0) try o.append(gpa, ',');
        try one(o, gpa, cells, l, a);
    }
    // op, op_families, plan width and route attempts: the frontend gives the runner only its abstain tag
    try o.print(gpa, "],\"abstains_complete\":{},\"abstains_dropped\":{d},\"abstain_frontend_detail\":false", .{ l.dropped == 0 and !l.anchors_lost, l.dropped });
    if (@TypeOf(cfg) != @TypeOf(null)) try o.print(gpa, ",\"runner_config\":{{\"backend\":\"{t}\",\"exact\":{},\"sync_graph\":{},\"inline_mark\":{},\"settle_us\":{d},\"max_steps\":{d},\"max_tokens\":{d},\"compile_steps\":{d},\"compile\":{},\"inline_math\":{},\"trace\":{}}}", .{ cfg.backend, cfg.exact, cfg.sync_graph, cfg.inline_mark, cfg.settle_us, cfg.max_steps, cfg.max_tokens, cfg.compile_steps, cfg.compile, cfg.scan.inline_math, cfg.scan.trace });
}

fn one(o: *std.ArrayList(u8), gpa: Allocator, cells: anytype, l: anytype, a: Abstain) Allocator.Error!void {
    const c = cells[a.cell];
    const kind: []const u8 = if (c.translation == .program) @tagName(c.translation.program.kind) else "none";
    const route: []const u8 = if (c.translation != .program) "none" else if (c.translation.program.kind == .graph) "graph" else "circuit";
    try o.print(gpa, "{{\"cell\":{d},\"form\":\"{s}\",", .{ a.cell, receipt.form_names[if (c.stated != null) 3 else @backingInt(c.form)] });
    try receipt.text(o, gpa, "lang", c.lang);
    try receipt.text(o, gpa, "code", c.code);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(c.code, &digest, .{});
    try o.print(gpa, "\"span_sha256\":\"{s}\",\"span_len\":{d},\"prior\":[", .{ std.fmt.bytesToHex(digest, .lower), c.code.len });
    // the notebook its frontend was given: earlier code cells, never inline values or stated computations (open)
    var n: usize = 0;
    if (c.stated == null) for (cells[0..a.cell]) |p| if (p.form != .inline_expr and p.stated == null) {
        if (n > 0) try o.append(gpa, ',');
        try receipt.quoted(o, gpa, p.code);
        n += 1;
    };
    try o.appendSlice(gpa, "],");
    if (c.stated) |st| try receipt.text(o, gpa, "display", st.display) else try o.appendSlice(gpa, "\"display\":null,");
    const src = source(l, a.cell, if (c.stated) |st| st.display else c.code);
    if (src.close) |cl| {
        try o.print(gpa, "\"source_field\":\"{s}\",", .{field(cl.content)});
        if (src.at) |at| try o.print(gpa, "\"source_at\":{d},\"source_end\":{d},", .{ at, src.end }) else try o.appendSlice(gpa, "\"source_at\":null,\"source_end\":null,");
        try o.print(gpa, "\"source_close_at\":{d},", .{cl.end});
    } else try o.appendSlice(gpa, "\"source_field\":null,\"source_at\":null,\"source_end\":null,\"source_close_at\":null,");
    var next = l.tokens;
    for (l.records.items) |r| if (r.token > a.token and r.token < next) {
        next = r.token;
    };
    for (l.abstains.items) |b| if (b.token > a.token and b.token < next) {
        next = b.token;
    };
    try o.print(gpa, "\"program_kind\":\"{s}\",\"backend_route\":\"{s}\",\"stage\":\"{s}\",\"reason\":\"{t}\",", .{ kind, route, stage(a.reason), a.reason });
    if (a.error_name) |e| try o.print(gpa, "\"error_name\":\"{s}\",", .{e}) else try o.appendSlice(gpa, "\"error_name\":null,");
    try o.print(gpa, "\"resume_field\":\"{s}\",\"resume_at\":{d},\"resume_token\":{d},\"round\":{d},\"next_decision_token\":{d},\"reply_tokens\":{d},\"owned_tokens\":0,\"steps\":{d},\"held_ms\":{d}}}", .{ field(a.content), a.at, a.token, a.round, next, l.tokens, a.steps, a.held_ms });
}

fn field(content: bool) []const u8 {
    return if (content) "content" else "reasoning";
}

/// The source's bytes [at, end): its last exact copy before the cell closed (none: normalized, e.g. REPL prompts cut).
fn source(l: anytype, cell: u32, needle: []const u8) struct { close: ?Close, at: ?u64 = null, end: u64 = 0 } {
    const cl = for (l.closes.items) |x| {
        if (x.cell == cell) break x;
    } else return .{ .close = null };
    const text = l.text[@intFromBool(cl.content)].items;
    const hay = text[0..@min(text.len, cl.end)];
    for ([_][]const u8{ needle, std.mem.trim(u8, needle, " \t\r\n") }) |want| {
        if (want.len == 0) continue;
        if (std.mem.lastIndexOf(u8, hay, want)) |at| return .{ .close = cl, .at = at, .end = at + want.len };
    }
    return .{ .close = cl };
}
