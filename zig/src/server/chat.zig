//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const recipient_reply = @import("recipient_reply.zig");
const tool_stream = @import("tool_stream.zig");
const prompt_mod = @import("prompt.zig");
const images = @import("images.zig");
const log = @import("log.zig");
const cached_tokens = @import("cached_tokens.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const wait = @import("wait_reason.zig");
const refusals = @import("refusals.zig");
const server_mod = @import("server.zig");
const Server = server_mod.Server;
const Mailbox = @import("mailbox.zig").Mailbox;
const StopHook = @import("stop_hook.zig").StopHook;
const engine_refusal = @import("engine_refusal.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };
/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    /// The client itself asked for thinking off (no server, template or route default counts): it must be honored.
    explicit_off: bool = false,
    trace: ?*@import("request_trace.zig").Trace = null, // the request's journal records; null when the journal is off
    /// The server's own request (a compaction summary, a lesson's wordings): it never takes --instruct-sampling's set.
    own: bool = false,
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// POST /v1/requests/{id}/cancel's ask for the client's request (cancel_ids.zig); a server-made request has none.
    cancel: ?*const std.atomic.Value(bool) = null,
    /// ``logprobs: true`` with this many ``top_logprobs``; null: not asked.
    logprobs: ?u8 = null,
    /// ``/v1/completions``' ``prompt_logprobs``: rows for the prompt with this many alternatives; null: not asked.
    prompt_logprobs: ?u8 = null,
    /// The client's own name for the grammar field (text.format, output_config.format); null on chat and completions.
    format: ?[]const u8 = null,
    /// The encoded image bytes the request may send; chat's limits unless a route (OCR) sets its own.
    image_bytes: images.ByteLimits = .{},
    /// With a known window, the reply may fill all the room the prompt leaves; max_tokens otherwise.
    fill_room: bool = false,
    /// --vision-helper: the request's own messages when they hold image or video parts (vision_helper_prompt.zig)
    media: ?Value = null,
    /// `tools` holds the compute default's own built-in tool (compute_turns.ownedByServer): its calls stay unseen.
    lane_own: bool = false,
};

/// A streamed piece: content text (a string) or a delta object (reasoning or tool calls).
pub const Sink = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, delta: Value) error{Closed}!void,
    /// Called only after admission, with cancellation and mailbox draining already guarded.
    begin: ?*const fn (ctx: *anyopaque, prompt_len: usize) error{Closed}!void = null,
    tokens: ?*const fn (ctx: *anyopaque, offset: usize, ids: []const u32) error{Closed}!void = null,
};

pub const Reply = struct {
    content: []const u8,
    stop_sequence: ?[]const u8,
    reasoning: ?[]const u8,
    tool_calls_streamed: bool,
    native_tool_format: bool = false,
    native_calls: ?[]Value = null,
    finish_reason: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    runtime: Value,
    speculative: Value,
    /// The template switches the prompt was rendered with (the next turn's prefill renders with them too).
    thinking: bool = false,
    effort: ?[]const u8 = null,
    /// ``choices[0].logprobs`` (prose, reasoning and call entries) for a request that asked; null otherwise.
    logprobs: ?Value = null,
    /// The tokens and target rows ``logprobs`` are projected from, once the route's tool policy is final.
    probability_rows: ?@import("logprob_parts.zig").Capture = null,
    /// For a token stream, each ``logprobs`` entry's raw token position, in the same shape; null otherwise.
    logprob_positions: ?Value = null,
    /// ``choices[0].prompt_logprobs`` (vLLM's form) for a request that asked; null otherwise.
    prompt_logprobs: ?Value = null,
    loop_stop: ?u32 = null, // the loop period when --loop-guard-stop ended the reply in its reasoning
    reasoning_recovered: bool = false, // ended at its end token in an unclosed think block: content repeats it
    runner_raw: ?[2][]const u8 = null, // with abstain records: the runner's raw field bytes (runner_projection.zig)
    compute_ran: usize = 0, // the lane's built-in calls the sandbox ran (not malformed or past the cap)
    owned: ?@import("reply_owned.zig").Owned = null, // engine-owned bytes of `content`, which no strip may cut
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

const nowNs = clock.nowNs;
const seconds = clock.seconds;

test "Server recovery: token staging allocation failure ends the loop without consuming tokens" {
    for (0..2) |fail_index| {
        var m: Mailbox = .{ .io = std.testing.io, .gpa = std.testing.allocator, .finished = true };
        defer m.deinit();
        try m.tokens.append(std.testing.allocator, 42);
        try m.chunks.append(std.testing.allocator, 1);
        try m.rows.append(std.testing.allocator, .{ .token = 42, .logprob = -1 });
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        var g: Generation = undefined;
        g.a = failing.allocator();
        g.box = &m;
        g.chunk_index = 0;
        g.consumed = 0;
        g.lp = .{};
        const Gone = struct {
            pub fn check(_: @This()) bool {
                return false;
            }
        };
        try std.testing.expectError(error.OutOfMemory, g.loop(Gone{}));
        try std.testing.expectEqual(@as(usize, 0), g.consumed);
        try std.testing.expectEqual(@as(usize, 0), g.chunk_index);
    }
}

/// Whether a reply gets a runner session (and the runner guide): with the spans `reasoning`, thinking chats only.
fn runsCells(srv: *Server, thinking: bool) bool {
    const rn = srv.info.runner orelse return false;
    return thinking or !rn.cfg.scan.reasoning_only;
}

/// A request rendered and checked but not submitted; it holds the preparing count until generate or release.
pub const Prepared = struct {
    input: Input,
    request: api.Request,
    prompt_len: usize,
    received: i96,
    thinking: bool,
    reasoning: bool, // thinking on a model with Info.reasoning: the reply splits off its reasoning block
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    token_ids: bool,
    guide: ?[]const u8 = null, // the runner guide the chat got ("" none); null with the runner off
    stops: fields_mod.Stops,
    preparing: bool,
    frames: []const ?api.Size = &.{}, // each image part's tower input size, in part order (prompt.Rendered.frames)
    deadline_ns: ?i96 = null, // "deadline_s": the server ends the reply there itself (finish_reason "deadline")
};

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.
pub fn prepare(srv: *Server, cx: *Cx, given: Input, gone: anytype) Failure!Prepared {
    const a = cx.a;
    var input = given;
    // the runner guide (runner_guide.zig) with the runner on: the request's, or the boot's under the compute default
    const thinks = input.prompt == null and (fields_mod.flag(input.fields, "enable_thinking") orelse srv.config.enable_thinking);
    const asked_guide = @import("compute_turns.zig").guideAsked(srv, input.fields, thinks);
    const guided = @import("runner_guide.zig").apply(a, if (input.prompt == null and runsCells(srv, thinks)) &srv.guides else null, asked_guide, input.messages) catch |e| switch (e) {
        error.BadGuide => return cx.refuse(try srv.guides.refusal(a)),
        error.OutOfMemory => return error.OutOfMemory,
    };
    input.messages = guided.messages;
    const io = srv.io;
    const received = nowNs(io);
    const f = input.fields;
    const stops_opt = try fields_mod.stopOptions(cx, f);
    var limit: i64 = @max(1, if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens);
    const priority = f.get("priority");
    const background = prompt_mod.isTitle(input.messages, input.tools.len > 0) or
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background")) or
        (priority == null and @import("server.zig").namedBackground(srv.config, f));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer srv.release(preparing, null);
    if (gone.check()) return error.Cancelled;
    const thinking = input.prompt == null and (fields_mod.flag(f, "enable_thinking") orelse srv.config.enable_thinking);
    const instruct = input.prompt == null and !thinking and !input.own; // a client's chat with thinking off may take --instruct-sampling's set
    const reasoning = thinking and srv.info.capability(.reasoning) and (srv.text.vtable.format != .kimi or input.prompt == null); // the prompt renders `thinking`; the reply reads `reasoning`
    const asked = if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null;
    const effort = srv.effortFor(asked);
    cx.named_effort = if (input.prompt == null) fields_mod.namedEffort(asked, srv.effort_levels) else null; // a raw prompt has no template
    cx.clear_thinking = fields_mod.flag(f, "clear_thinking");
    cx.template_fields = f;
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.capability(.context_fitted)) ", the most this server's memory budget fits" else "", n });
        if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        limit = if (input.fill_room) room else @min(limit, room);
    }
    const system_len: usize = if (input.prompt != null or rendered.media != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids, instruct);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .penalties = try srv.resolvePenalties(cx, f, instruct),
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .rewind_len = @intCast(rendered.rewind_len),
        .user_end = @intCast(rendered.user_end),
        .shared_prefixes = shared.items,
        .message_starts = rendered.message_starts,
        // a cut before the conversation's own text: fresh sessions resume their whole harness
        .chunks = try srv.chunks.request(a, rendered.ids, system_len),
        // from the tokens alone, for every request: cached states are keyed by tokens
        .decode_spans = try prompt_mod.replySpans(srv, cx, rendered.ids),
        .tools_json = if (input.tools.len > 0) try json.stringify(a, .{ .array = @constCast(input.tools) }, .{ .ascii = false }) else "",
        .decide = if (input.prompt == null and !background) prompt_mod.decideOf(srv, rendered.ids) else null,
        .images = rendered.images,
        .media = rendered.media,
        .cache_key = rendered.cache_key,
    };
    try srv.checkFeatures(cx, f, input.tools.len > 0, reasoning, rendered.ids, input.tools, &request);
    server_mod.nameStructure(input.format, &request);
    errdefer srv.release(false, request);
    if (reasoning) {
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
        if (budget > 0 and !srv.info.capability(.thinking_budget))
            return cx.refuse("thinking_budget is not supported by this native engine");
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        request.loop_stop = request.loop_guard and srv.config.loop_stop;
        if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
        }
        // The answer quota is separate only on engines that enforce it.
        if (srv.info.capability(.answer_limit)) @import("thinking_reserve.zig").apply(&request, srv.config.thinking_reserve, if (window > 0) window - n else std.math.maxInt(u32), srv.think_close_end);
    }
    request.logprobs = input.logprobs;
    request.prompt_logprobs = input.prompt_logprobs;
    // the compute lane: a reply offered the built-in tool holds at each EOS for its results (not with stop strings)
    request.hold_eos = stops_opt.strings.len == 0 and @import("compute_turns.zig").arms(srv, input, &request);
    // the in-model runner (runner_binding.zig): code cells this reply writes run, their results owned in place
    request.run_cells = runsCells(srv, thinking);
    if (request.run_cells and thinking and srv.think_close.len > 0) request.think_end = srv.think_close_end;
    if (request.run_cells) request.run_trace = @import("runner_guide.zig").traceAsk(f) catch
        return cx.refuse("runner_trace must be a boolean, runner_trace_format one line of at most 128 bytes with {} once, runner_trace_check a boolean, runner_trace_repeat 0 to 3");
    if (thinking) request.run_commit = try @import("runner_commit_ask.zig").of(a, srv.info.runner, srv.think_close, input.messages, input.tools, &request);
    if (try engine_refusal.of(srv.engine, cx, &request)) |message| return cx.refuse(message);
    if (input.trace) |trace| trace.prepared(&request, thinking, effort);
    const deadline: ?i96 = if (f.get("deadline_s")) |v| blk: {
        const s = (if (v == .bool) null else v.float64()) orelse return cx.refuse("deadline_s must be a number of seconds");
        if (!(s > 0 and s <= 1e6)) return cx.refuse("deadline_s must be more than 0 and at most 1000000 seconds");
        break :blk received + @as(i96, @intFromFloat(s * std.time.ns_per_s));
    } else null;
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .reasoning = reasoning, .effort = effort, .sampled = sampling != null, .drafts = drafts, .token_ids = fields_mod.flag(f, "return_token_ids") orelse false, .stops = stops_opt, .preparing = preparing, .frames = rendered.frames, .guide = guided.guide, .deadline_ns = deadline };
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer srv.release(preparing, request); // after the engine's finished: the errdefer below drains first
    if (request.background) {
        while (true) {
            const now = nowNs(io);
            const waiting = now < received + 150 * std.time.ns_per_ms or (srv.preparing.load(.acquire) > 0 and now < received + 2 * std.time.ns_per_s);
            if (!waiting) break;
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
    }
    if (gone.check()) return error.Cancelled;
    if (!srv.info.capability(.text_generation)) return cx.refuse("this model does not support text generation");
    if (srv.info.capability(.decides)) return cx.refuse("this model makes decisions and does not write text: use POST /v1/systemone");
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    if (stops_opt.strings.len > 0) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => refusals.count(srv, .engine_busy, cx.wait(wait.engine_busy, "the engine is busy; retry shortly", .{})),
        error.Closed => refusals.count(srv, .engine_unavailable, cx.wait(wait.engine_unavailable, "the engine is unavailable; retry after it restarts", .{})),
        error.InvalidSpans => refusals.count(srv, .invalid_request, cx.fail(.request, "the prompt span layout is invalid", .{})),
        error.InvalidRequest => refusals.count(srv, .invalid_request, cx.fail(.request, "the request does not match the loaded model", .{})),
        error.Stale => refusals.failure(srv, .engine_stale, cx.fail(.server, "{s}", .{api.stale_message})),
        error.MemoryPressure => refusals.count(srv, .memory_pressure, cx.wait(wait.memory_pressure, "{s}", .{api.pressure_message})),
    };
    srv.release(preparing, null); // a background request waits only while a foreground one prepares
    preparing = false;
    var filter: @import("compute_turns.zig").Filter = .{ .a = a, .inner = if (sink) |x| x else undefined, .tool = srv.info.compute_tool.name };
    const shown: ?Sink = if (request.hold_eos and sink != null) filter.sink() else sink; // built-in calls stay hidden
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .asked = input.cancel, .reply_id = input.id, .seed = if (request.sampling) |v| v.seed else null, .sink = shown, .hold = request.hold_eos, .thinking = prepared.reasoning or (srv.text.vtable.format != .kimi and srv.info.capability(.reasoning) and reply_text.isChannel(srv.markers)), .chat = input.prompt == null, .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .token_ids = prepared.token_ids, .lp = if (request.logprobs != null) .{ .positions = if (sink) |s| s.tokens != null else false } else null, .loop_stop = request.loop_stop, .single = fields_mod.flag(input.fields, "tool_call_single") orelse false, .recover = input.prompt == null and request.structure == null, .deadline_ns = prepared.deadline_ns, .lane_own = input.lane_own, .guide = prepared.guide };
    defer srv.noteRequest(prepared.prompt_len, cached_tokens.within(box.cached, prepared.prompt_len), gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds, if (box.stats.prefill_seconds != null) box.stats.queued_seconds else null);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    if (srv.recipients and input.prompt == null) gen.recipient = try recipient_reply.Reading.of(a, srv.text, request.prompt, input.tools, prepared.reasoning);
    const result: Failure!Reply = blk: {
        if (sink) |s| if (s.begin) |begin| begin(s.ctx, prepared.prompt_len) catch break :blk error.Cancelled;
        if (sink != null and input.tools.len > 0 and srv.text.vtable.format != .kimi) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        if (gen.held) break :blk @import("compute_segments.zig").rest(&gen, cx, prepared, &filter, gone);
        break :blk gen.finish(cx, prepared.prompt_len, received, prepared.thinking, prepared.reasoning, effort, prepared.sampled, prepared.drafts);
    };
    if (result) |r| {
        if (@import("compute_turns.zig").forced(srv)) try @import("compute_turns.zig").countExecuted(a, r.runtime, r.compute_ran);
    } else |e| logEnded(input.id, e, cx.message, prepared.prompt_len, gen.collected.items.len, seconds(nowNs(io) - received));
    return result;
}

/// The ``ended`` line of a submitted reply that ends without one: its client left, or it failed.
fn logEnded(id: []const u8, e: Failure, message: []const u8, prompt: usize, tokens: usize, after: f64) void {
    const why: ?[]const u8 = switch (e) {
        error.Cancelled => null,
        error.Refused => message, // once submitted, only the engine's own failure refuses
        else => @errorName(e),
    };
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, why, prompt, tokens, after)});
}

/// The reply to ``input``; ``sink`` hears the stream (null: not streamed). ``gone`` says the client left.
pub fn run(srv: *Server, cx: *Cx, input: Input, sink: ?Sink, gone: anytype) Failure!Reply {
    if (srv.config.compact_at == null) return generate(srv, cx, try prepare(srv, cx, input, gone), sink, gone);
    return @import("compact.zig").run(srv, cx, input, sink, gone);
}

/// The token loop and the text it streams.
const Generation = @import("chat_generation.zig").Generation;

pub const deltaOf = reply_text.deltaOf;
