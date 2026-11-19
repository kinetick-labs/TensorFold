//! Hy4-flash's chat template through this engine against jinja2 in transformers' environment, on demand:
//! tools/hy4/template_parity.py writes DIR/{chat_template.jinja,cases.json,ref.json}, then
//! HY4_TEMPLATE_DIR=DIR zig test -lc --dep sys -Mroot=zig/src/core/template/hy4_parity_test.zig -Msys=zig/src/platform/sys.zig
//! The template is Tencent's (tencent/Hy4-flash, tencent-hy-license) and is not kept in the tree.
const std = @import("std");
const template = @import("template.zig");

fn slurp(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24));
}

test "hy4 template parity with jinja2" {
    const a = std.testing.allocator;
    const dir = std.testing.environ.getPosix("HY4_TEMPLATE_DIR") orelse return error.SkipZigTest;
    const src = try slurp(a, dir, "chat_template.jinja");
    defer a.free(src);
    const cases_json = try slurp(a, dir, "cases.json");
    defer a.free(cases_json);
    const ref_json = try slurp(a, dir, "ref.json");
    defer a.free(ref_json);
    const cases = try std.json.parseFromSlice(std.json.Value, a, cases_json, .{});
    defer cases.deinit();
    const refs = try std.json.parseFromSlice(std.json.Value, a, ref_json, .{});
    defer refs.deinit();
    var bad: usize = 0;
    for (cases.value.array.items, refs.value.array.items, 0..) |c, r, i| {
        var diag = template.Diag{};
        defer if (diag.msg.len > 0) a.free(diag.msg);
        const obj = c.object;
        const gen = if (obj.get("add_generation_prompt")) |g| g.bool else true;
        const out = template.renderChat(a, src, obj.get("messages") orelse .null, obj.get("tools") orelse .null, c, gen, &diag) catch |err| {
            if (r.object.get("error")) |e| {
                std.debug.print("case {d}: both fail: ours={s} ({s}) ref={s}\n", .{ i, diag.msg, @errorName(err), e.string });
            } else {
                std.debug.print("case {d}: ours FAILED {s}: {s}\n", .{ i, @errorName(err), diag.msg });
                bad += 1;
            }
            continue;
        };
        defer a.free(out);
        if (r.object.get("text")) |t| {
            if (std.mem.eql(u8, t.string, out)) {
                std.debug.print("case {d}: identical ({d} bytes)\n", .{ i, out.len });
            } else {
                bad += 1;
                const n = std.mem.indexOfDiff(u8, t.string, out) orelse 0;
                std.debug.print("case {d}: DIFF at {d}\nref: {s}\nour: {s}\n", .{ i, n, t.string[n -| 40..@min(t.string.len, n + 80)], out[n -| 40..@min(out.len, n + 80)] });
            }
        } else {
            bad += 1;
            std.debug.print("case {d}: ours rendered but ref failed\n", .{i});
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}
