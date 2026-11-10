//! One admitted reply: token ownership, streaming, completion and native-family parsing.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const reply_owned = @import("reply_owned.zig");
const projection = @import("runner_projection.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const tool_lfm2 = @import("tool_lfm2.zig");
const log = @import("log.zig");
const cached_tokens = @import("cached_tokens.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const Mailbox = @import("mailbox.zig").Mailbox;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Failure = chat.Failure;
const Reply = chat.Reply;
const Sink = chat.Sink;
const nowNs = clock.nowNs;
const seconds = clock.seconds;
const deltaOf = reply_text.deltaOf;

/// One chunk's logprobs: ``content`` always, as OpenAI clients read it, and ``key``'s entries.
fn rowsOf(a: Allocator, key: []const u8, entries: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "content", .{ .array = &.{} });
    try o.put(a, key, entries);
    return .{ .object = o };
}

pub const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    asked: ?*const std.atomic.Value(bool) = null, // POST /v1/requests/{id}/cancel's ask (cancel_ids.zig)
    cancel_sent: bool = false, // the ask went to the engine once; the loop then reads until the engine finishes
    seed: ?u64 = null,
    sink: ?Sink,
    thinking: bool,
    chat: bool = true,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    single: bool, // parallel_tool_calls: false: a cut call is no call, so its prose ends as the whole reply's does
    token_ids: bool, // "return_token_ids": the response's tensorfold block carries the reply's token ids
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    kimi: ?@import("kimi_reply.zig").Stream = null,
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: reply_text.Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: reply_text.Shown = .{},
    streaming_done: bool = false,
    wrote_ns: ?i96 = null,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    lp: ?@import("logprob_parts.zig").Live = null, // the request asked for logprobs: its rows, copied with each round
    loop_stop: bool = false, // --loop-guard-stop: the engine ends the reply where the loop guard fires
    deadline_ns: ?i96 = null, // "deadline_s": when the server ends the reply itself
    deadline_hit: bool = false, // it did: the engine was cancelled there; what it gave before its end still lands
    lane_own: bool = false, // Input.lane_own: a call max_tokens cut never shows as prose, as the reply's own call
    owned: ?reply_owned.Owned = null, // engine-owned bytes of this segment's content (reply_owned.of its cell writes)
    guide: ?[]const u8 = null, // the runner guide text the chat got ("" none); null: the runner is off
    cells_sent: usize = 0, // the runner records the stream sent last: a prefix the client has every landed byte of
    cells_sig: u64 = 0, // their Mailbox.covered sig
    base: usize = 0, // the reply's tokens before this segment's (compute_segments.zig): the runner records count them
    hold: bool = false, // the engine holds the reply at an EOS (Request.hold_eos): this segment ends there
    held: bool = false, // it did: the engine has not finished, and compute_turns.zig resumes or ends it
    skip: []const u32 = &.{}, // a resumed segment's first tokens: the inserted tool result, checked and not shown
    resumed: bool = false, // a segment after an inserted tool result (compute_segments.next)

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// Close a tool call left open at the model end token, except when another stop condition ended the reply.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
                g.held = g.hold;
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const live = g.liveRows();
        if (live) |l| try l.add(g.srv, g.a, fresh.items);
        const all = try g.visible.extend(g.a, g.srv.text, reply_text.decodeText, fresh.items);
        if (g.visible.pending()) return; // a character still split across tokens
        if (g.srv.text.vtable.format == .kimi) return @import("chat_kimi.zig").stream(g, sink);
        const raw_visible = g.stops.visible(all, true);
        const text = try @import("tool_minimax.zig").dsml(g.a, raw_visible, g.tools); // MiniMax-M3's calls as DSML
        var answer = text;
        var thought: []const u8 = "";
        if (g.thinking) {
            const original_close = try g.visible.closeOf(g.a, reply_text.decodeText, g.srv.text, g.srv.close_id);
            const close = try @import("tool_minimax.zig").closeAfterRewrite(g.a, raw_visible, g.tools, original_close);
            const split = try reply_text.splitThinkingAt(g.a, text, false, g.srv.markers, close);
            thought = g.streamed_reasoning.after(reply_text.pyStrip(split.reasoning), true); // the whole reply strips it
            if (thought.len > 0) try g.streamed_reasoning.set(g.a, reply_text.pyStrip(split.reasoning), thought);
            answer = split.answer;
        }
        answer = try tool_lfm2.dsml(g.a, answer, g.tools, false); // the answer only, so a call the reasoning mentions is not made
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false, true, reply_text.openersOf(g.srv.markers)) else answer;
        const vis = reply_text.heldBack(if (g.srv.info.capability(.reasoning)) reply_text.streamingVisible(shown) else shown, g.calls != null);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) try g.streamed.set(g.a, vis, delta);
        // each chunk carries the entries its text settles: reasoning ones before the answer's first text
        const rows = if (live) |l| try l.settle(g.srv, g.a, g.collected.items[0..l.spelled.ends.items.len], g.thinking, g.stops, g.streamed.text.items, g.streamed_reasoning.text.items) else null;
        const thought_rows = if (rows) |r| r.get("reasoning_content") else null;
        if (thought.len > 0 or thought_rows != null) {
            const d: Value = if (thought.len > 0) try deltaOf(g.a, "reasoning_content", thought) else .{ .object = try json.newObject(g.a) };
            if (thought_rows) |t| try d.object.put(g.a, "logprobs", try rowsOf(g.a, "reasoning_content", t));
            try g.emit(sink, d);
        }
        const prose_rows: []Value = if (rows) |r| r.get("content").?.array else &.{};
        if (delta.len > 0 or prose_rows.len > 0) {
            const d: Value = if (prose_rows.len > 0) try deltaOf(g.a, "content", delta) else .{ .string = delta };
            if (prose_rows.len > 0) try d.object.put(g.a, "logprobs", try rowsOf(g.a, "content", .{ .array = prose_rows }));
            try g.emit(sink, d);
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    /// Logprob state when chunks carry their own entries: without tools, no call can claim streamed text.
    fn liveRows(g: *Generation) ?*@import("logprob_parts.zig").Live {
        if (g.tools.len > 0) return null;
        return if (g.lp) |*l| l else null;
    }

    pub fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.wrote_ns = nowNs(g.srv.io);
    }

    /// End the request and wait for the engine; unfinished requests count as disconnects.
    pub fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    pub fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    pub fn loop(g: *Generation, gone: anytype) Failure!void {
        const Gone = switch (@typeInfo(@TypeOf(gone))) {
            .pointer => |p| p.child,
            else => @TypeOf(gone),
        };
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch |e| {
                    m.mutex.unlock(m.io);
                    return e;
                };
                if (g.lp) |*l| l.rows.appendSlice(g.a, m.rows.items[g.consumed..end]) catch |e| {
                    g.a.free(chunk.?);
                    m.mutex.unlock(m.io);
                    return e;
                };
                g.chunk_index += 1;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check()) {
                g.cancel();
                return error.Cancelled;
            }
            // the cancel route: the engine's cancel, as a disconnect sends it; the reply then ends "cancelled"
            if (g.asked) |x| if (!g.cancel_sent and g.reason == null and x.load(.acquire)) {
                g.srv.engine.cancel(g.id);
                g.cancel_sent = true;
            };
            if (g.deadline_ns) |d| if (!g.deadline_hit and g.reason == null and nowNs(g.srv.io) >= d) {
                g.deadline_hit = true;
                g.srv.engine.cancel(g.id); // the engine ends the stream: its tokens and usage stay whole
            };
            if (chunk == null) if (g.sink) |sink| try g.liveCells(sink);
            if (chunk == null and @hasDecl(Gone, "beat")) {
                // queueing and a long prefill send nothing: a client that times out a silent stream would leave
                const now = nowNs(g.srv.io);
                const since = g.wrote_ns orelse now;
                if (g.wrote_ns == null) g.wrote_ns = now;
                if (gone.beat(now - since) catch {
                    g.cancel();
                    return error.Cancelled;
                }) g.wrote_ns = now;
            }
            const tokens = try g.skipInserted(chunk orelse continue);
            if (g.reason == null) try g.commit(tokens);
            if (g.sink) |sink| try g.liveCells(sink);
            if (g.held) return;
        }
    }

    /// Sends the runner records whose bytes the client has (`tensorfold.runner.cells`), so a cut stream keeps them.
    pub fn liveCells(g: *Generation, sink: Sink) Failure!void {
        const m = g.box;
        var text: std.ArrayList(u8) = .empty;
        {
            m.mutex.lockUncancelable(m.io);
            defer m.mutex.unlock(m.io);
            const got = m.covered(.{ g.streamed_reasoning.text.items.len, g.streamed.text.items.len });
            // never fewer than sent: the text may have been replaced shorter (a stop string's held prefix)
            if (got.n < g.cells_sent or got.sig == g.cells_sig) return; // records off: n 0, sig 0, so never
            try text.appendSlice(g.a, "{\"runner\":{\"live\":true,\"cells\":[");
            for (m.cells.items[0..got.n], 0..) |versions, i| {
                if (i > 0) try text.append(g.a, ',');
                try text.appendSlice(g.a, versions.items[0].json);
            }
            try text.appendSlice(g.a, "]}}");
            g.cells_sent = got.n;
            g.cells_sig = got.sig;
        }
        const v = switch (try json.parse(g.a, text.items)) {
            .ok => |v| v,
            .err => return,
        };
        const delta = try json.newObject(g.a);
        try delta.put(g.a, "tensorfold", v); // openai.zig's chunk puts it beside the choices, as the final chunk's receipt
        try g.emit(sink, .{ .object = delta });
    }

    /// The engine-owned bytes of `content` (reply_owned.of), from the runner records the stream has had so far.
    fn ownedOf(g: *Generation, tokens: []const u32, raw: []const u8, content: []const u8) Allocator.Error!?reply_owned.Owned {
        if (g.srv.text.vtable.token_bytes == null) return null;
        var records: std.ArrayList([]const u8) = .empty;
        {
            const m = g.box;
            m.mutex.lockUncancelable(m.io);
            defer m.mutex.unlock(m.io);
            for (m.cells.items) |versions| if (versions.items.len > 0)
                try records.append(g.a, try g.a.dupe(u8, versions.items[versions.items.len - 1].json));
        }
        if (records.items.len == 0) return null;
        const pieces = try g.a.alloc([]const u8, tokens.len);
        for (tokens, pieces) |t, *p| p.* = (try g.srv.text.tokenBytes(g.a, t)) orelse return null;
        return reply_owned.of(g.a, records.items, g.base, pieces, raw, content, reply_text.openersOf(g.srv.markers));
    }

    /// Drops the inserted result that opens a resumed segment; a different token there fails the reply.
    fn skipInserted(g: *Generation, tokens: []u32) Failure![]u32 {
        const n = @min(tokens.len, g.skip.len);
        if (!std.mem.eql(u32, tokens[0..n], g.skip[0..n])) {
            log.line("error: {s}: the engine's tokens after a tool result are not the result it was given", .{g.reply_id});
            g.cancel();
            return error.Failed;
        }
        g.skip = g.skip[n..];
        return tokens[n..];
    }

    /// A reply's text as `finish` reads it: stops, MiniMax calls as DSML, a call closed at its EOS, the think split.
    pub const Read = struct { text: []const u8, close: []const u8, content: []const u8, reasoning: ?[]const u8 };

    /// `raw`, the decoded `content_tokens`, as the client reads it: every format but Kimi's native reader.
    pub fn readText(g: *const Generation, raw: []const u8, content_tokens: []const u32) Failure!Read {
        const a = g.a;
        const raw_visible = g.stops.visible(raw, false);
        const visible = try @import("tool_minimax.zig").dsml(a, raw_visible, g.tools); // MiniMax-M3's calls as DSML
        // closed before the think split, so a call ending an unclosed think block is the answer
        const close = try g.closeCall(visible);
        const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
        var r: Read = .{ .text = text, .close = close, .content = text, .reasoning = null };
        if (g.thinking) {
            const original_close = try reply_text.closeWhere(a, reply_text.decodeText, g.srv.text, content_tokens, g.srv.close_id);
            const close_at = try @import("tool_minimax.zig").closeAfterRewrite(a, raw_visible, g.tools, original_close);
            const split = try reply_text.splitThinkingAt(a, text, true, g.srv.markers, close_at);
            r.reasoning = nonEmpty(reply_text.pyStrip(split.reasoning));
            r.content = split.answer;
            // a resumed segment that ended at its EOS inside the think block the result turn reopened: its text answers
            const t = g.collected.items;
            if (g.resumed and close_at != .token and r.reasoning != null and reply_text.pyStrip(r.content).len == 0 and t.len > 0 and g.eos(t[t.len - 1])) {
                r.content = r.reasoning.?;
                r.reasoning = null;
            }
        } else if (g.srv.info.capability(.reasoning)) {
            const h = reply_text.parseHarmony(text);
            r.content = h.content;
            r.reasoning = h.reasoning;
        }
        r.content = try tool_lfm2.dsml(a, r.content, g.tools, true);
        return r;
    }

    pub fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, thinking: bool, reasons: bool, effort: ?[]const u8, exact: bool, drafts: bool) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done and !g.held) g.drain();
        const natural: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .silence => return cx.other("the text engine returned a speech-only finish reason"),
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
        };
        // the server ended the reply at its deadline: a late token's EOS, stop string or cap never hides that
        const ended = if (g.deadline_hit) "deadline" else natural;
        const stopped: ?u32 = if (g.loop_stop) m.stats.loop_period else null;
        const reason = if (stopped != null) @import("loop_stop.zig").finish_reason else ended;
        const finished_ns = nowNs(g.srv.io);
        const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
        const raw = try g.srv.text.decode(a, content_tokens);
        const native = if (g.srv.text.vtable.format == .kimi) try @import("chat_kimi.zig").read(g, content_tokens, true) else null;
        const read: Read = if (native) |n| .{ .text = n.content, .close = "", .content = n.content, .reasoning = nonEmpty(reply_text.pyStrip(n.reasoning)) } else try g.readText(raw, content_tokens);
        const text, const close, const content, const reasoning = .{ read.text, read.close, read.content, read.reasoning };
        g.owned = try g.ownedOf(content_tokens, raw, content); // the runner's cell writes (none: null)
        if (g.sink) |sink| {
            const thought = g.streamed_reasoning.text.items;
            if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
            const cut = g.lane_own and std.mem.eql(u8, reason, "length");
            const o = reply_text.openersOf(g.srv.markers);
            const shown = if (cut) try reply_owned.cutProse(a, content, o, g.owned) else if (g.calls != null) try tool_stream.endProseOwned(a, content, if (g.thinking) content else text, g.tools, reason, g.single, o, g.owned) else content;
            const sent = g.streamed.text.items;
            const rest = if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) shown[sent.len..] else "";
            if (rest.len > 0) try g.emit(sink, .{ .string = rest });
            if (g.calls) |*c| if (close.len > 0 or std.mem.eql(u8, reason, "length")) {
                // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                var out: std.ArrayList(Value) = .empty;
                try c.finishCut(if (g.thinking) content else text, &out); // a cut value sends its held bytes
                for (out.items) |d| try g.emit(sink, d);
            };
        }
        const prefilled = m.prefilled_ns;
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        // the effort the template heard: with thinking off, only one the request named
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = cx.named_effort orelse "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, decode_s) }); // the reply's decode seconds, as tokens_per_second divides them
        try @import("grammar_metrics.zig").put(a, runtime, m.stats.grammar);
        if (m.stats.gpu_seconds) |gpu| try runtime.put(a, "gpu_seconds", .{ .float = gpu }); // TF_Q27_PROFILE only
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = @import("tokens.zig").tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        const sha256 = @import("tokens.zig").tokenSha256(g.collected.items);
        try runtime.put(a, "token_sha256", .{ .string = try a.dupe(u8, &sha256) });
        // "return_token_ids": the reply's tokens, the same slice the shas above hash, so ids and sha cannot disagree
        if (g.token_ids) {
            const reply_ids = try a.alloc(Value, g.collected.items.len);
            for (g.collected.items, reply_ids) |t, *slot| slot.* = try json.intValue(a, t);
            try runtime.put(a, "token_ids", .{ .array = reply_ids });
        }
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            if (stopped != null) try loop_field.put(a, "stopped", .{ .bool = true });
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const cells = try a.dupe(u8, m.runner);
        var runner_raw: ?[2][]const u8 = null; // abstain anchors index these; openai.zig maps them to what it sends
        if (cells.len > 0) switch (try json.parse(a, cells)) {
            .ok => |v| {
                if (v.get("abstains") != null) runner_raw = try projection.rawFields(a, g.srv.text, content_tokens, if (g.thinking) g.srv.think_close_end else null);
                try runtime.put(a, "runner", v);
            },
            .err => {},
        };
        if (g.guide) |given| {
            const guide = try json.newObject(a);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(given, &digest, .{});
            try guide.put(a, "sha256", .{ .string = try a.dupe(u8, &std.fmt.bytesToHex(digest, .lower)) });
            try guide.put(a, "bytes", try json.intValue(a, given.len));
            try runtime.put(a, "runner_guide", .{ .object = guide });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        const counters = try a.dupe(u8, m.telemetry); // parsed strings point into it: it outlives the box
        if (counters.len > 0) switch (try json.parse(a, counters)) {
            .ok => |v| try spec.put(a, "proposer", v),
            .err => {},
        };
        const think_end: ?u32 = if (reasons) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .native_tool_format = native != null,
            .native_calls = if (native) |n| n.calls else null,
            .runner_raw = runner_raw,
            .owned = g.owned,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = cached_tokens.within(m.cached, prompt_len),
            .completion_tokens = g.collected.items.len,
            .reasoning_tokens = if (native) |n| n.reasoning_tokens else reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .thinking = thinking,
            .effort = effort,
            .probability_rows = if (g.lp) |l| .{ .tokens = content_tokens, .rows = l.rows.items, .thinking = g.thinking, .stops = g.stops, .closer = close, .sent = l.sent } else null,
            .prompt_logprobs = if (m.prompt_rows) |r| try @import("logprobs.zig").promptValue(g.srv, a, r) else null,
            .loop_stop = stopped,
        };
        if (std.mem.eql(u8, reason, "length") and reasons and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        const done_ns = nowNs(g.srv.io);
        const ttft_s = if (g.first_ns) |at| @max(0, seconds(at - received)) else 0;
        const decode_log_s = if (g.first_ns) |at| @max(0, seconds(done_ns - at)) else 0;
        var seed_text: [32]u8 = undefined;
        const seed = if (g.seed) |v| std.fmt.bufPrint(&seed_text, " seed={d}", .{v}) catch unreachable else "";
        const rate = if (decode_log_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_log_s else 0;
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d} ttft={d:.3} decode={d:.3} elapsed={d:.3} queued={d:.3} tok/s={d:.1}{s}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted, ttft_s, decode_log_s, @max(0, seconds(done_ns - received)), s.queued_seconds, rate, seed });
        return reply;
    }
};

fn nonEmpty(s: []const u8) ?[]const u8 {
    return if (s.len > 0) s else null;
}
