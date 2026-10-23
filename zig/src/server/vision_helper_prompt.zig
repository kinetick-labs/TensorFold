//! --vision-helper's prompts: media parts split out (Python's split_images), template rendered, helper's ids and media.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const messages = @import("messages.zig");
const model_text = @import("model_text.zig");
const log = @import("log.zig");
const helper_mod = @import("vision_helper.zig");
const Server = @import("server.zig").Server;
const Rendered = @import("prompt.zig").Rendered;
const Source = helper_mod.Source;
const Value = json.Value;
const Cx = errors.Cx;

/// chat.Input.media through the helper: normalized, split, rendered, then the helper's token ids and media.
pub fn prepare(srv: *Server, cx: *Cx, raw: Value, tools: []const Value, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    const helper = srv.vision_helper orelse return cx.refuse("image input requires --vision-helper");
    const kept = try messages.normalizeWith(cx, raw, srv.late_system, srv.needs_user_after_tool, .{ .images = true, .videos = true });
    var sources: std.ArrayList(Source) = .empty;
    const template = try messages.toolArguments(cx, try splitMedia(cx, kept, srv.media_limits, &sources));
    if (!helper.videos) for (sources.items) |src| if (src.kind == .video) return cx.refuse("content parts must be text or image_url; this helper serves no video");
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = true,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
    };
    const text = srv.text.render(cx.a, template, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template rejected the request: {s}", .{problem}),
    };
    const window = srv.info.context_window;
    const result = helper.prepare(cx.a, text.text, sources.items, if (window > 0) window else null) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.fail(.capacity, "image processing failed ({s}); retry shortly", .{@errorName(e)}),
    };
    switch (result) {
        .refused => |r| {
            if (r.context) return cx.fail(.context_length, "{s}", .{r.message});
            return cx.fail(if (r.status == 503) .capacity else .request, "{s}", .{r.message});
        },
        .ok => |p| {
            log.line("vision: {d} image(s), {d} video(s): {d} visual tokens of {d} prompt tokens (prepare {d:.3}s, tower {d:.3}s, helper GPU peak {d:.2} GiB, features {s})", .{ p.images, p.videos, p.visual_tokens, p.tokens.len, p.prepare_s, p.encode_s, @as(f64, @floatFromInt(p.peak)) / (1 << 30), p.features_sha });
            dump(srv, cx.a, p.media);
            return .{ .ids = p.tokens, .media = p.media };
        },
    }
}

var dumped = std.atomic.Value(u32).init(0);

/// TENSORFOLD_VISION_DUMP: `helper-<n>.{features,rows,pos}` raw, as native/cuda_mia_vision.zig writes the bridge's.
fn dump(srv: *Server, a: std.mem.Allocator, m: *const @import("engine_api").Media) void {
    const dir = std.mem.span(std.c.getenv("TENSORFOLD_VISION_DUMP") orelse return);
    const n = dumped.fetchAdd(1, .monotonic) + 1;
    const parts = [_]struct { []const u8, []const u8 }{
        .{ "features", m.features },
        .{ "rows", std.mem.sliceAsBytes(m.rows) },
        .{ "pos", std.mem.sliceAsBytes(m.positions) },
    };
    for (parts) |part| {
        const path = std.fmt.allocPrint(a, "{s}/helper-{d}.{s}", .{ dir, n, part[0] }) catch return;
        std.Io.Dir.cwd().writeFile(srv.io, .{ .sub_path = path, .data = part[1] }) catch |e| log.line("vision dump: {s}", .{@errorName(e)});
    }
}

/// ``has_images``: a message whose content holds an image_url or video_url part.
pub fn hasVisual(list_in: ?Value) bool {
    const list = list_in orelse return false;
    if (list != .array) return false;
    for (list.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| if (part.typeIs("image_url") or part.typeIs("video_url")) return true;
    }
    return false;
}

const media_keys = [_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" };

fn mediaBesides(part: Value, keep: []const u8) bool {
    for (media_keys) |k| if (!std.mem.eql(u8, k, keep) and json.truthyField(part, k)) return true;
    return false;
}

/// Media limits ``split_images`` checks: Flash Next's (the single-Spark recipe's patches 0008 and 0009).
pub const MediaLimits = struct {
    max_images: u32 = 50,
    max_videos: u32 = 4,
    image_bytes: usize = 10 * 1024 * 1024,
    video_bytes: usize = 64 * 1024 * 1024,
    max_url_chars: usize = 4096,
    allow_urls: bool = false,
};

/// split_images: image_url parts become {"type": "image", "detail"}, video_url {"type": "video"}; media in order.
pub fn splitMedia(cx: *Cx, list_in: ?Value, limits: MediaLimits, sources: *std.ArrayList(Source)) errors.Refused!Value {
    const list = list_in orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a non-empty list");
    var out: std.ArrayList(Value) = .empty;
    var images: u32 = 0;
    var videos: u32 = 0;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!messages.isRole(role)) return cx.refuse("invalid message role");
        for (media_keys) |k| if (json.truthyField(message, k)) return cx.refuse("images must be image_url parts in user or tool message content");
        const content = message.get("content");
        if (content == null or content.? == .null or content.? == .string) {
            try out.append(cx.a, message);
            continue;
        }
        if (content.? != .array) return cx.refuse("message content must be text or an array of content parts");
        var parts: std.ArrayList(Value) = .empty;
        for (content.?.array) |part| {
            if (part != .object) return cx.refuse("each content part must be an object");
            if (part.typeIs("text")) {
                const t = part.get("text");
                if (t == null or t.? != .string or mediaBesides(part, "")) return cx.refuse("text parts must contain a text string without media");
                try parts.append(cx.a, part);
            } else if (part.typeIs("video_url")) {
                if (!std.mem.eql(u8, role.?.string, "user")) return cx.refuse("video_url parts are supported only in user messages");
                if (mediaBesides(part, "video_url")) return cx.refuse("video_url parts cannot contain other media");
                if (videos >= limits.max_videos) return cx.fail(.request, "a request supports at most {d} videos; send fewer, or restart the server with --vision-max-videos N to raise the count limit", .{limits.max_videos});
                const v = part.get("video_url");
                const url = if (v) |x| x.strField("url") else null;
                if (url == null or url.?.len == 0) return cx.refuse("video_url must contain a non-empty url string");
                if (std.mem.startsWith(u8, url.?, "data:")) {
                    if (url.?.len > limits.video_bytes * 3 / 2 + 256) return cx.refuse("video data URL exceeds the encoded byte limit");
                } else if (!std.mem.startsWith(u8, url.?, "https://")) {
                    return cx.refuse("videos require data URLs or public HTTPS URLs");
                } else if (!limits.allow_urls) {
                    return cx.refuse("video URLs are off on this server; send the video as a data URL, or start the server with --vision-urls");
                } else if (url.?.len > limits.max_url_chars) return cx.refuse("video URL is too long");
                try sources.append(cx.a, .{ .kind = .video, .url = url.? });
                videos += 1;
                const o = try json.newObject(cx.a);
                try o.put(cx.a, "type", .{ .string = "video" });
                try parts.append(cx.a, .{ .object = o });
            } else if (part.typeIs("image_url")) {
                const r = role.?.string;
                if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) return cx.refuse("image_url parts are supported only in user and tool messages");
                if (mediaBesides(part, "image_url")) return cx.refuse("image_url parts cannot contain other media");
                if (images >= limits.max_images) return cx.fail(.request, "a request supports at most {d} images across the full message history, including prior turns; remove older image content, start a new conversation, or restart the server with --vision-max-images N to raise the count limit (other image limits still apply)", .{limits.max_images});
                const v = part.get("image_url");
                const url = if (v) |x| x.strField("url") else null;
                if (v == null or v.? != .object or url == null or url.?.len == 0) return cx.refuse("image_url must contain a non-empty url string");
                var detail: []const u8 = "auto";
                if (v.?.get("detail")) |d| {
                    const ok = d == .string and (std.mem.eql(u8, d.string, "auto") or std.mem.eql(u8, d.string, "low") or std.mem.eql(u8, d.string, "high"));
                    if (!ok) return cx.refuse("image detail must be auto, low or high");
                    detail = d.string;
                }
                if (std.mem.startsWith(u8, url.?, "data:")) {
                    if (url.?.len > limits.image_bytes * 3 + 256) return cx.refuse("image data URL exceeds the encoded byte limit");
                } else if (!std.mem.startsWith(u8, url.?, "https://")) {
                    return cx.refuse("images require data URLs or public HTTPS URLs");
                } else if (!limits.allow_urls) {
                    return cx.refuse("image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls");
                } else if (url.?.len > limits.max_url_chars) return cx.refuse("image URL is too long");
                try sources.append(cx.a, .{ .kind = .image, .url = url.?, .detail = detail });
                images += 1;
                const o = try json.newObject(cx.a);
                try o.put(cx.a, "type", .{ .string = "image" });
                try o.put(cx.a, "detail", .{ .string = detail });
                try parts.append(cx.a, .{ .object = o });
            } else return cx.refuse("content parts must be text, image_url or video_url; audio is unsupported");
        }
        try out.append(cx.a, .{ .object = try messages.withField(cx, message.object, "content", .{ .array = parts.items }) });
    }
    return .{ .array = out.items };
}

test "split_images: image and video parts become the template's parts, the media in order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const list = (try json.parse(cx.a,
        \\[{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBBB"}}]}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "t", "content": [{"type": "text", "text": "shot:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CCCC"}}]}]
    )).ok;
    try std.testing.expect(hasVisual(list));
    var sources: std.ArrayList(Source) = .empty;
    const kept = try messages.normalizeWith(&cx, list, "system", false, .{ .images = true, .videos = true });
    const template = try splitMedia(&cx, kept, .{}, &sources);
    try std.testing.expectEqual(@as(usize, 3), sources.items.len);
    try std.testing.expect(sources.items[0].kind == .image and std.mem.eql(u8, sources.items[0].detail, "low"));
    try std.testing.expect(sources.items[1].kind == .video);
    try std.testing.expect(sources.items[2].kind == .image and std.mem.eql(u8, sources.items[2].detail, "auto"));
    try std.testing.expectEqualStrings(
        \\[{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "detail": "low"}, {"type": "video"}]}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "t", "content": [{"type": "text", "text": "shot:"}, {"type": "image", "detail": "auto"}]}]
    , try json.stringify(cx.a, template, .{ .ascii = false }));
}

test "split_images refuses what Python refuses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cases = [_][2][]const u8{
        .{ "[{\"role\": \"assistant\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,AA\"}}]}]", "image_url parts are supported only in user and tool messages" },
        .{ "[{\"role\": \"tool\", \"content\": [{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64,AA\"}}]}]", "video_url parts are supported only in user messages" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"http://x/y.png\"}}]}]", "images require data URLs or public HTTPS URLs" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://x/y.png\"}}]}]", "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,x\", \"detail\": \"max\"}}]}]", "image detail must be auto, low or high" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"input_audio\", \"input_audio\": {}}]}]", "content parts must be text, image_url or video_url; audio is unsupported" },
    };
    for (cases) |c| {
        var cx: Cx = .{ .a = arena.allocator() };
        var sources: std.ArrayList(Source) = .empty;
        const list = (try json.parse(cx.a, c[0])).ok;
        try std.testing.expectError(error.Refused, splitMedia(&cx, list, .{}, &sources));
        try std.testing.expectEqualStrings(c[1], cx.message);
    }
    // more than the request's images
    var cx: Cx = .{ .a = arena.allocator() };
    var sources: std.ArrayList(Source) = .empty;
    const two = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,a\"}}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,b\"}}]}]")).ok;
    try std.testing.expectError(error.Refused, splitMedia(&cx, two, .{ .max_images = 1 }, &sources));
    try std.testing.expect(std.mem.startsWith(u8, cx.message, "a request supports at most 1 images"));
}
