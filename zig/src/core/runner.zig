//! The in-model runner: per-reply sessions run closed cells on the circuit and own the model's result positions.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("runner_api.zig");
const scan = @import("runner_scan.zig");
const trace = @import("runner_trace.zig");
const records = @import("runner_receipt.zig");
const abstain = @import("runner_abstain.zig");
pub const place = @import("runner_place.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);
const output = @import("runner_output.zig");
const answer = @import("runner_commit.zig");

pub const Config = struct {
    scan: scan.Config = .{},
    max_steps: u64 = 4_000_000, // per run (a step count: wall time never changes a result)
    compile_steps: usize = 10522, // PREFIX_COMPILE_STEPS: frontend cost limit; opt-in maximum 32768, not an instruction cap
    max_tokens: u32 = 512, // owned result tokens before the cut note
    workers: u8 = 2,
    device: enum { cpu, cuda } = .cpu,
    circuit: []const u8 = "", // calc and ELF programs run here (with the graph backend too)
    backend: enum { circuit, graph } = .circuit,
    graph_bank: []const u8 = "", // PREFIX_GRAPH_BANK: T1's bank manifest, loaded at boot (backend graph)
    exact: bool = false, // `calc` / `exact` cells (firmware exact rationals, not any language's print) run
    place: place.Config = .{}, // PREFIX_CPUS / PREFIX_NICE: where the runner's threads run (core/runner_place.zig)
    settle_us: u32 = 0, // PREFIX_SETTLE_US: a held tick waits this long for its cell; 0: one stage per tick (as before)
    guide: []const u8 = "", // PREFIX_GUIDE: "1" RUNNER-GUIDE.md, else a file, as chats' default guide (server/runner_guide.zig)
    guide_max_bytes: usize = 512, // PREFIX_GUIDE_MAX_BYTES: request guide cap, opt-in up to 2048 bytes
    compile: bool = true, // false: the frontend never builds an ELF (no clang); calc and graph programs only (S3)
    inline_mark: bool = false, // PREFIX_INLINE_MARK: tag owned inline values; off preserves their bytes
    digit_stop: lanes.runner_stop.Limits = .{}, // PREFIX_DIGIT_STOP / _MARK_STOP / _RUN_MAX / _HOLD_MAX: no runaway
    sync_graph: bool = false, // PREFIX_SYNC_GRAPH: a graph result settles at its trigger token, in that round (no hold)
    cells: bool = true, // PREFIX_CELLS: the receipt records each owned cell's span, route and landed value (runner_receipt.zig)
    abstains: bool = false, // PREFIX_ABSTAINS: with cells, each abstained cell's source, reason and stage (runner_abstain.zig)
    commit: bool = false, // PREFIX_COMMIT: an owned value of the task's own expression closes the think and is the answer
    prefix: []const u8 = "TF_RUNNER", // the knob prefix it was parsed under (runner_binding.zig), for messages
    family: @import("runner_binding.zig").Family = .{}, // the family's chat markup (its binding sets it)

    /// PREFIX "1" enables (unset, "" or "0": null); PREFIX_* knobs override defaults; a bad value is error.BadKnob.
    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!?Config {
        const on = get(prefix) orelse return null;
        if (on.len == 0 or std.mem.eql(u8, on, "0")) return null;
        if (!std.mem.eql(u8, on, "1")) return bad(prefix, on);
        var c: Config = .{ .prefix = prefix };
        if (get(prefix ++ "_BACKEND")) |b| if (b.len > 0) {
            c.backend = std.meta.stringToEnum(@TypeOf(c.backend), b) orelse return bad(prefix ++ "_BACKEND", b);
        };
        c.graph_bank = get(prefix ++ "_GRAPH_BANK") orelse "";
        if (c.backend == .graph and c.graph_bank.len == 0) return bad(prefix ++ "_GRAPH_BANK", "(unset)");
        c.circuit = get(prefix ++ "_CIRCUIT") orelse "";
        if (c.circuit.len == 0) return bad(prefix ++ "_CIRCUIT", "(unset)");
        if (get(prefix ++ "_DEVICE")) |d| if (d.len > 0) {
            c.device = std.meta.stringToEnum(@TypeOf(c.device), d) orelse return bad(prefix ++ "_DEVICE", d);
        };
        c.workers = try knob(u8, get, prefix ++ "_WORKERS", c.workers, 1, 64);
        c.max_steps = try knob(u64, get, prefix ++ "_MAX_STEPS", c.max_steps, 1, 1 << 40);
        c.compile_steps = try knob(usize, get, prefix ++ "_COMPILE_STEPS", c.compile_steps, 1, 32768);
        c.max_tokens = try knob(u32, get, prefix ++ "_MAX_TOKENS", c.max_tokens, 1, 1 << 20);
        c.exact = try knob(u1, get, prefix ++ "_EXACT", 0, 0, 1) == 1;
        c.settle_us = try knob(u32, get, prefix ++ "_SETTLE_US", c.settle_us, 0, 100_000);
        c.inline_mark = try knob(u1, get, prefix ++ "_INLINE_MARK", 0, 0, 1) == 1;
        c.digit_stop.after_value = try knob(u1, get, prefix ++ "_DIGIT_STOP", 1, 0, 1) == 1;
        c.digit_stop.after_mark = try knob(u1, get, prefix ++ "_MARK_STOP", 1, 0, 1) == 1;
        c.digit_stop.run_max = try knob(u32, get, prefix ++ "_DIGIT_RUN_MAX", c.digit_stop.run_max, 0, 1 << 20);
        c.digit_stop.hold_max = try knob(u32, get, prefix ++ "_DIGIT_HOLD_MAX", c.digit_stop.hold_max, 0, 1 << 20);
        c.sync_graph = try knob(u1, get, prefix ++ "_SYNC_GRAPH", 0, 0, 1) == 1;
        if (c.sync_graph and c.backend != .graph) return bad(prefix ++ "_SYNC_GRAPH", "1 (backend is not graph)");
        c.cells = try knob(u1, get, prefix ++ "_CELLS", 1, 0, 1) == 1;
        c.abstains = try knob(u1, get, prefix ++ "_ABSTAINS", 0, 0, 1) == 1;
        c.commit = try knob(u1, get, prefix ++ "_COMMIT", 0, 0, 1) == 1;
        c.guide = get(prefix ++ "_GUIDE") orelse "";
        if (std.mem.eql(u8, c.guide, "0")) c.guide = "";
        c.guide_max_bytes = try knob(usize, get, prefix ++ "_GUIDE_MAX_BYTES", c.guide_max_bytes, 512, 2048);
        c.place = try place.Config.parse(get, prefix);
        c.scan.lookahead = try knob(u32, get, prefix ++ "_LOOKAHEAD", c.scan.lookahead, 1, 4096);
        c.scan.float_window = try knob(u32, get, prefix ++ "_FLOAT_WINDOW", c.scan.float_window, 0, 65536);
        c.scan.inline_mod = try knob(u1, get, prefix ++ "_INLINE_MOD", 1, 0, 1) == 1;
        c.scan.inline_arith = try knob(u1, get, prefix ++ "_INLINE_ARITH", 1, 0, 1) == 1;
        c.scan.inline_chain = try knob(u1, get, prefix ++ "_INLINE_CHAIN", 0, 0, 1) == 1;
        c.scan.inline_quoted = try knob(u1, get, prefix ++ "_INLINE_QUOTED", 1, 0, 1) == 1;
        c.scan.trace_window = try knob(u32, get, prefix ++ "_TRACE_WINDOW", c.scan.trace_window, 1, 65536);
        c.scan.trace_whole = try knob(u1, get, prefix ++ "_STATED", 0, 0, 1) == 1;
        if (get(prefix ++ "_FORMS")) |f| if (f.len > 0) {
            c.scan.fence = false;
            c.scan.repl = false;
            c.scan.inline_math = false;
            c.scan.trace = false;
            var it = std.mem.splitScalar(u8, f, ',');
            while (it.next()) |form| {
                if (std.mem.eql(u8, form, "fence")) c.scan.fence = true else if (std.mem.eql(u8, form, "repl")) c.scan.repl = true else if (std.mem.eql(u8, form, "inline")) c.scan.inline_math = true else if (std.mem.eql(u8, form, "trace")) c.scan.trace = true else return bad(prefix ++ "_FORMS", f);
            }
        };
        // the knob widens form trace's statements: without the form it would do nothing
        if (c.scan.trace_whole and !c.scan.trace) return bad(prefix ++ "_STATED", "1 (form trace is off)");
        if (get(prefix ++ "_SPANS")) |sp| if (sp.len > 0) {
            if (std.mem.eql(u8, sp, "reasoning")) c.scan.reasoning_only = true else if (!std.mem.eql(u8, sp, "all")) return bad(prefix ++ "_SPANS", sp);
        };
        return c;
    }

    const knob = @import("runner_knob.zig").knob;
    const bad = @import("runner_knob.zig").bad;
};

/// The engine's tokenizer as the runner needs it: one token's bytes, and text to token ids.
pub const Text = output.Text;

/// A request's say over form trace (runner_trace.zig); defaults change nothing.
pub const TraceAsk = trace.Ask;

/// The server-wide runner: a frontend, a backend and the server's tokenizer (set at its start); sessions are per reply.
pub const Runner = struct {
    gpa: Allocator,
    cfg: Config,
    frontend: api.Frontend,
    backend: api.Backend,
    text: ?Text = null,
    next_id: u64 = 1,
    placer: ?*place.Placer = null, // the backend's job threads and the translations run placed
    quick: ?api.Frontend = null, // cfg.sync_graph: translates on the engine thread, graph programs or abstain only

    /// A reply's session; null until the server has given the runner its tokenizer.
    pub fn session(r: *Runner, ask: TraceAsk) Allocator.Error!?*Session {
        if (r.text == null) return null;
        const format = try r.gpa.dupe(u8, ask.format);
        errdefer r.gpa.free(format);
        const s = try r.gpa.create(Session);
        var cfg = r.cfg.scan;
        if (ask.off) cfg.trace = false;
        cfg.trace_repeat = ask.repeat;
        s.* = .{ .r = r, .id = r.next_id, .scan = .{ .cfg = cfg }, .format = format, .check = ask.check, .rec = .{ .on = r.cfg.cells, .abstains_on = r.cfg.abstains }, .digits = .{ .limits = r.cfg.digit_stop } };
        r.next_id += 1;
        return s;
    }
};

const Cell = struct {
    lang: []u8,
    code: []u8,
    form: api.Form,
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread = null,
    translated: std.atomic.Value(bool) = .init(false),
    translation: api.Translation = .{ .abstain = .unsupported },
    job: ?api.JobId = null,
    result: ?api.Result = null, // out copied into the arena
    abstained: bool = false,
    told: enum { none, owned, abstained } = .none, // what its result header got
    owned_tokens: u32 = 0,
    stated: ?scan.Stated = null, // form trace: the `[exact: …]` line's expression (text in the arena)
    quoted: bool = false, // its inline span sat inside an open quotation (scan.Closed.quoted): never owned
};

/// One reply's cells, runs and the trigger it waits on; every method runs on the engine thread.
pub const Session = struct {
    r: *Runner,
    id: u64,
    scan: scan.Scanner,
    cells: std.ArrayList(*Cell) = .empty,
    waiting: ?scan.Trigger = null,
    held_at: u64 = 0, // monotonic ns when the stream began to wait (the serving log's latency, never a result)
    owned_n: u32 = 0,
    abstained_n: u32 = 0,
    held_ms: u64 = 0, // summed over the reply's holds
    held_ticks: u32 = 0, // the current hold's ticks so far (the serving log's)
    /// An owned result's forced tokens still to commit.
    forcing: ?struct { at: u64, left: usize, cell: u32, inline_bytes_left: ?usize = null } = null,
    resume_ms: u64 = 0, // summed: an owned result's settling to the model's first token after it
    terminal_seen: bool = false,
    broken: bool = false, // out of memory once: no more cells, no more holds
    sync_ids: []u32 = &.{}, // cfg.sync_graph: the result a trigger settled with, forced by the stream (hook.settled)
    format: []u8, // TraceAsk.format, owned
    check: bool = false, // TraceAsk.check
    rec: records.Log = .{}, // the receipt's per-cell records
    commit: answer.State = .{}, // PREFIX_COMMIT: the request's ask (lane_stream), the think state, whether it fired
    digits: lanes.runner_stop.State = .{}, // the digit stop after owned inline values, in reasoning (lanes/runner_stop.zig)

    pub fn hook(s: *Session) lanes.stream.RunnerHook {
        return .{ .ptr = s, .token = onToken, .settled = if (s.r.cfg.sync_graph) settled else null, .eos = onEos, .stop = onStop, .watching = onWatching };
    }

    fn onStop(ptr: *anyopaque, tokens: []const u32, forced: usize) ?lanes.runner_stop.Stop {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (s.broken) return null;
        const tx = s.r.text.?;
        const hit = s.digits.stop(.{ .ctx = tx.ctx, .of = tx.bytes }, tokens, forced) orelse return null;
        s.digits.fired(hit.why);
        log.info("reply {d}: digit stop ({t}): a drawn token replaced by a newline; later runs cap at {d}", .{ s.id, hit.why, s.digits.cap() });
        return hit;
    }

    fn onWatching(ptr: *anyopaque) bool {
        const s: *Session = @ptrCast(@alignCast(ptr));
        return !s.broken and s.digits.watching();
    }

    /// The tokens a trigger's run settled with at the token itself (once; null: hold for them as before).
    fn settled(ptr: *anyopaque) ?[]const u32 {
        const s: *Session = @ptrCast(@alignCast(ptr));
        return if (s.sync_ids.len > 0) s.sync_ids else null;
    }

    fn onToken(ptr: *anyopaque, token: u32, think_open: bool, can_hold: bool, think_end: bool) bool {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (s.broken) return false;
        return s.feed(token, think_open, can_hold, think_end) catch {
            s.broken = true;
            return false;
        };
    }

    fn onEos(ptr: *anyopaque, think_open: bool, can_hold: bool) bool {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (s.broken or s.terminal_seen or s.commit.fired) return false;
        return s.finishEos(think_open, can_hold) catch {
            s.broken = true;
            return false;
        };
    }

    fn finishEos(s: *Session, think_open: bool, can_hold: bool) Allocator.Error!bool {
        const ev = try s.scan.finishEos(s.r.gpa, think_open);
        if (ev.closed) |c| try s.open(c);
        const trigger = ev.trigger orelse return false;
        if (!can_hold or trigger.index >= s.cells.items.len) return false;
        s.waiting = trigger;
        s.held_at = nowNs();
        s.held_ticks = 0;
        s.terminal_seen = true;
        return true;
    }

    fn feed(s: *Session, token: u32, think_open: bool, can_hold: bool, think_end: bool) Allocator.Error!bool {
        const tx = s.r.text.?;
        if (s.forcing) |*r| {
            // the forced tokens commit first; the next token is the model's own, one round after the result settled
            if (r.left > 0) r.left -= 1 else {
                const ms = (nowNs() -| r.at) / std.time.ns_per_ms;
                s.resume_ms += ms;
                log.info("reply {d} cell {d}: model's next token {d} ms after its result", .{ s.id, r.cell, ms });
                s.forcing = null;
            }
        }
        if (s.sync_ids.len > 0) {
            s.r.gpa.free(s.sync_ids);
            s.sync_ids = &.{};
        }
        const bytes = if (think_end) "" else tx.bytes(tx.ctx, token);
        s.digits.landed(bytes, think_open);
        if (s.digits.wants()) try s.digits.look(s.r.gpa, tx);
        try s.rec.token(s.r.gpa, token, bytes, think_open);
        if (answer.skip(&s.commit, think_open)) return false;
        const scanning = if (s.forcing) |*r| output.scanning(&r.inline_bytes_left, bytes) else bytes;
        if (scanning.len == 0 and bytes.len > 0) return false;
        const ev = try s.scan.feed(s.r.gpa, scanning, think_open);
        if (ev.closed) |c| try s.open(c);
        const t = ev.trigger orelse return false;
        if (!can_hold or t.index >= s.cells.items.len) return false;
        if (s.r.cfg.sync_graph and !t.trace) if (try s.now(t)) |ids| {
            // settled here: owned tokens go to the stream as forced ones (empty: abstained, the model's own text stays)
            if (ids.len == 0) {
                s.r.gpa.free(ids);
                return false;
            }
            s.sync_ids = ids;
            s.rec.synced();
            return true;
        };
        s.waiting = t;
        s.held_at = nowNs();
        s.held_ticks = 0;
        return true;
    }

    fn open(s: *Session, c: scan.Closed) Allocator.Error!void {
        const gpa = s.r.gpa;
        const cell = try gpa.create(Cell);
        errdefer gpa.destroy(cell);
        cell.* = .{ .lang = try gpa.dupe(u8, c.lang), .code = undefined, .form = c.form, .arena = .init(gpa) };
        errdefer gpa.free(cell.lang);
        cell.code = try gpa.dupe(u8, c.code);
        errdefer gpa.free(cell.code);
        // earlier code cells are the notebook; inline values and stated computations define nothing
        var n: usize = 0;
        for (s.cells.items) |p| n += @intFromBool(p.form != .inline_expr and p.stated == null);
        const prior = try cell.arena.allocator().alloc([]const u8, n);
        n = 0;
        for (s.cells.items) |p| if (p.form != .inline_expr and p.stated == null) {
            prior[n] = p.code;
            n += 1;
        };
        try s.cells.append(gpa, cell);
        s.rec.closed(gpa, c.index);
        // a stated computation is literals only: no notebook names to shadow
        if (c.trace) |st| {
            cell.stated = .{ .display = try cell.arena.allocator().dupe(u8, st.display), .divmod = st.divmod };
            const task: Translate = .{ .cell = cell, .fe = s.r.frontend, .prior = &.{}, .placer = s.r.placer };
            cell.thread = std.Thread.spawn(.{}, Translate.spawned, .{task}) catch blk: {
                task.run();
                break :blk null;
            };
            return;
        }
        // exact inline math would contradict a reply reasoning in Python floats; a quoted span is not a computation
        if (c.own_context or c.quoted) {
            cell.translation = .{ .abstain = .context };
            cell.quoted = c.quoted;
            cell.translated.store(true, .release);
            return;
        }
        if (!s.r.cfg.exact and (std.mem.eql(u8, cell.lang, "calc") or std.mem.eql(u8, cell.lang, "exact"))) {
            cell.translation = .{ .abstain = .language };
            cell.translated.store(true, .release);
            return;
        }
        const task: Translate = .{ .cell = cell, .fe = s.r.frontend, .prior = prior, .placer = s.r.placer };
        // sync_graph: what the quick frontend finds on the engine thread (graph steps; inline: all) needs no thread
        if (s.r.cfg.sync_graph) if (s.r.quick) |q| {
            const quick: api.Cell = .{ .form = cell.form, .lang = cell.lang, .code = cell.code, .prior = prior };
            const t = q.translate(q.ctx, cell.arena.allocator(), quick) catch @as(api.Translation, .{ .abstain = .too_big });
            // an inline value's quick translation is its whole one (no compile); other cells keep a graph program only
            if (cell.form == .inline_expr or (t == .program and t.program.kind == .graph)) {
                cell.translation = t;
                cell.translated.store(true, .release);
                return;
            }
        };
        // settling: an inline value (microseconds to translate) waits on no thread's start
        if (s.r.cfg.settle_us > 0 and c.form == .inline_expr) return task.run();
        cell.thread = std.Thread.spawn(.{}, Translate.spawned, .{task}) catch blk: {
            task.run();
            break :blk null;
        };
    }

    /// Advances translations and runs; the tokens to force when the awaited cell is settled (caller frees them).
    pub fn tick(s: *Session) Allocator.Error!?[]u32 {
        s.rec.rounds += 1;
        if (s.waiting) |w| if (s.r.cfg.settle_us > 0) s.settle(s.cells.items[w.index]);
        for (s.cells.items) |c| s.advance(c);
        const w = s.waiting orelse return null;
        const c = s.cells.items[w.index];
        s.held_ticks += 1;
        if (!c.abstained and c.result == null) return null;
        s.waiting = null;
        const held_ms = (nowNs() -| s.held_at) / std.time.ns_per_ms;
        s.held_ms += held_ms;
        return try s.conclude(w, c, held_ms);
    }

    /// cfg.sync_graph at a trigger: a graph program run here, its tokens forced this round (empty: abstain, null: hold)
    fn now(s: *Session, w: scan.Trigger) Allocator.Error!?[]u32 {
        const c = s.cells.items[w.index];
        if (!c.translated.load(.acquire) or c.job != null) return null;
        if (c.thread) |th| {
            th.join();
            c.thread = null;
        }
        if (c.result == null and !c.abstained) switch (c.translation) {
            .abstain => c.abstained = true,
            .program => |p| {
                if (p.kind != .graph) return null;
                const res = s.r.backend.now(p, s.r.cfg.max_steps, sync_steps) orelse return null;
                s.take(c, res);
            },
        };
        s.held_ticks = 0;
        return try s.conclude(w, c, 0);
    }

    /// A settled cell's tokens to force after its trigger (empty: abstained), counted and logged with the hold's ticks.
    fn conclude(s: *Session, w: scan.Trigger, c: *Cell, held_ms: u64) Allocator.Error![]u32 {
        if (w.trace) return try s.ownTrace(c, w.index, held_ms);
        // an inline value is owned only when the run halted with a one-line value
        if (!c.abstained and w.inline_space != null and (c.result.?.status != .ok or std.mem.indexOfScalar(u8, std.mem.trimEnd(u8, c.result.?.out, "\n"), '\n') != null)) c.abstained = true;
        if (!c.abstained and s.r.cfg.family.screen(&c.result.?)) c.abstained = true; // it would read as the family's markup
        if (!c.abstained and s.r.cfg.inline_mark and w.inline_space != null) {
            const value = output.value(c.result.?.out);
            if (std.mem.trim(u8, value, " \t\r").len == 0 or std.mem.indexOfAny(u8, value, "\r\n") != null)
                c.abstained = true;
        }
        if (c.abstained) {
            abstain.told(s, c, w.index, held_ms, if (c.quoted) .quoted else null);
            log.info("reply {d} cell {d} ({s} {t}): abstain ({s}), held {d} ms, ticks {d}", .{ s.id, w.index, c.lang, c.form, why(c), held_ms, s.held_ticks });
            return try s.r.gpa.alloc(u32, 0);
        }
        const res = c.result.?;
        const ids = try s.owned(res, w);
        if (s.r.cfg.inline_mark and w.inline_space != null and ids.len == 0) {
            abstain.told(s, c, w.index, held_ms, .token_cap);
            log.info("reply {d} cell {d}: marked inline value exceeds token cap", .{ s.id, w.index });
            return ids;
        }
        s.owned_n += 1;
        if (w.inline_space != null) try s.digits.arm(s.r.gpa, s.r.text.?, ids.len, if (s.r.cfg.inline_mark) output.suffix else output.value(res.out));
        c.told = .owned;
        c.owned_tokens = @intCast(ids.len);
        s.rec.owned(s.r.gpa, w.index, ids) catch {
            s.rec.on = false; // out of memory: this reply's receipt has no records
        };
        s.forcing = .{
            .at = nowNs(),
            .left = ids.len,
            .cell = w.index,
            .inline_bytes_left = if (s.r.cfg.inline_mark and w.inline_space != null)
                output.value(res.out).len + @intFromBool(w.inline_space.?)
            else
                null,
        };
        log.info("reply {d} cell {d} ({s} {t}): owned {d} tokens, {t} after {d} steps, run {d} ms, held {d} ms, ticks {d}", .{ s.id, w.index, c.lang, c.form, ids.len, res.status, res.steps, res.exec_ns / std.time.ns_per_ms, held_ms, s.held_ticks });
        return answer.after(s, c, w, w.index, ids);
    }

    /// Form trace: one `[exact: <expr> = <value>]` line, or nothing (abstain) unless the run printed one integer value.
    fn ownTrace(s: *Session, c: *Cell, index: u32, held_ms: u64) Allocator.Error![]u32 {
        const gpa = s.r.gpa;
        const tx = s.r.text.?;
        const line = if (!c.abstained and c.result.?.status == .ok) try trace.owned(gpa, s.format, c.stated.?.display, c.stated.?.divmod, c.result.?.out, s.check) else null;
        if (line) |text| {
            defer gpa.free(text);
            const ids = try tx.encode(tx.ctx, gpa, text);
            if (ids.len <= s.r.cfg.max_tokens) {
                s.owned_n += 1;
                c.told = .owned;
                c.owned_tokens = @intCast(ids.len);
                s.rec.owned(gpa, index, ids) catch {
                    s.rec.on = false;
                };
                s.forcing = .{ .at = nowNs(), .left = ids.len, .cell = index };
                const res = c.result.?;
                log.info("reply {d} cell {d} (trace): owned {d} tokens, {d} steps, run {d} ms, held {d} ms, ticks {d}", .{ s.id, index, ids.len, res.steps, res.exec_ns / std.time.ns_per_ms, held_ms, s.held_ticks });
                return ids;
            }
            gpa.free(ids);
        }
        abstain.told(s, c, index, held_ms, if (line != null) .token_cap else if (c.result != null and c.result.?.status == .ok and !c.abstained) .trace_shape else null);
        log.info("reply {d} cell {d} (trace): abstain ({s}), held {d} ms, ticks {d}", .{ s.id, index, if (c.result != null and c.result.?.status == .ok) "not one value" else why(c), held_ms, s.held_ticks });
        return try gpa.alloc(u32, 0);
    }

    fn why(c: *const Cell) []const u8 {
        if (c.translation == .abstain) return if (c.quoted) "quoted" else @tagName(c.translation.abstain);
        const res = c.result orelse return "backend start";
        if (res.status == .abstain) return if (res.abstain) |a| @tagName(a) else "backend";
        if (res.status != .ok) return @tagName(res.status); // inline: budget or fault
        return if (res.out.len == 0) "empty output" else "multi-line inline value";
    }

    /// The awaited cell's stages on this tick for up to settle_us: run here if that ends it, else its job waited on.
    fn settle(s: *Session, c: *Cell) void {
        const until = nowNs() + @as(u64, s.r.cfg.settle_us) * std.time.ns_per_us;
        var tried = false;
        while (true) {
            if (c.thread) |th| if (c.translated.load(.acquire)) {
                th.join();
                c.thread = null;
            };
            if (!tried and c.thread == null and c.job == null and c.result == null and !c.abstained and
                c.translated.load(.acquire) and c.translation == .program)
            {
                tried = true;
                const quick = @as(u64, s.r.cfg.settle_us) * quick_steps_per_us;
                if (s.r.backend.now(c.translation.program, s.r.cfg.max_steps, quick)) |res| return s.take(c, res);
            }
            s.advance(c);
            if (c.abstained or c.result != null or nowNs() >= until) return;
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    fn advance(s: *Session, c: *Cell) void {
        if (c.thread) |th| {
            if (!c.translated.load(.acquire)) return;
            th.join();
            c.thread = null;
        } else if (!c.translated.load(.acquire)) return;
        if (c.abstained or c.result != null) return;
        const prog = switch (c.translation) {
            .abstain => {
                c.abstained = true;
                return;
            },
            .program => |p| p,
        };
        const job = c.job orelse {
            c.job = s.r.backend.start(s.id, prog, s.r.cfg.max_steps) catch |e| switch (e) {
                error.Busy => return, // tried again next round: tokens never depend on it
                error.OutOfMemory => {
                    c.abstained = true;
                    return;
                },
            };
            // settling: a run that is done at once is taken on the tick that started it
            if (s.r.cfg.settle_us == 0) return;
            s.poll(c, c.job.?);
            return;
        };
        s.poll(c, job);
    }

    fn poll(s: *Session, c: *Cell, job: api.JobId) void {
        switch (s.r.backend.poll(job)) {
            .running => {},
            .done => |res| {
                c.job = null;
                s.take(c, res);
            },
        }
    }

    fn take(_: *Session, c: *Cell, res: api.Result) void {
        var kept = res;
        kept.out = c.arena.allocator().dupe(u8, res.out) catch {
            c.abstained = true;
            return;
        };
        c.result = kept;
        if (res.status == .abstain or (res.status == .ok and res.out.len == 0)) c.abstained = true;
    }

    /// The owned text for a settled run, as token ids within the token cap.
    fn owned(s: *Session, res: api.Result, w: scan.Trigger) Allocator.Error![]u32 {
        return output.owned(s.r.gpa, s.r.text.?, res, w, .{
            .max_tokens = s.r.cfg.max_tokens,
            .inline_mark = s.r.cfg.inline_mark,
        });
    }

    /// Whether the session waits on a run (its stream is held at a trigger).
    pub fn awaiting(s: *const Session) bool {
        return s.waiting != null;
    }

    /// The reply's runner receipt as JSON (runtime.runner, runner_receipt.zig); valid until the session is destroyed.
    pub fn receipt(s: *Session) []const u8 {
        return records.write(s);
    }

    /// Cancels runs, joins translations and frees the session; the backend drops the stream's machines.
    pub fn destroy(s: *Session) void {
        const gpa = s.r.gpa;
        gpa.free(s.sync_ids);
        if (s.cells.items.len > 0) log.info("reply {d}: {d} cells closed, {d} owned, {d} abstained at a header, {d} never claimed", .{ s.id, s.cells.items.len, s.owned_n, s.abstained_n, s.cells.items.len - s.owned_n - s.abstained_n });
        for (s.cells.items) |c| if (c.thread) |th| th.join(); // all first: a later cell's prior borrows earlier code
        for (s.cells.items) |c| {
            if (c.job) |j| s.r.backend.cancel(j);
            gpa.free(c.lang);
            gpa.free(c.code);
            c.arena.deinit();
            gpa.destroy(c);
        }
        s.r.backend.release(s.id);
        s.cells.deinit(gpa);
        s.scan.deinit(gpa);
        gpa.free(s.format);
        s.rec.deinit(gpa);
        gpa.destroy(s);
    }
};

/// Settling's cap on a run on the engine thread: steps per microsecond of settle_us (a graph step is about 20 ns).
const quick_steps_per_us = 20;

/// sync_graph's cap on a run at its trigger, on the engine thread: about 2 ms of graph steps; longer runs hold.
const sync_steps = 100_000;

/// Monotonic nanoseconds for the serving log only.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Translate = struct {
    cell: *Cell,
    fe: api.Frontend,
    prior: []const []const u8,
    placer: ?*place.Placer = null,

    /// On its own thread: placed first. The inline fallback (no thread) stays on the engine thread's CPU and niceness.
    fn spawned(t: Translate) void {
        if (t.placer) |p| p.enter();
        t.run();
    }

    fn run(t: Translate) void {
        const c = t.cell;
        const cell: api.Cell = .{ .form = c.form, .lang = c.lang, .code = c.code, .prior = t.prior };
        c.translation = t.fe.translate(t.fe.ctx, c.arena.allocator(), cell) catch .{ .abstain = .too_big };
        c.translated.store(true, .release);
    }
};

test {
    _ = @import("runner_tests.zig");
}
