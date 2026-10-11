//! The Flash Next Triton spec (zig/tests/cuda/flashnext/kernels.json) against itself, and against a capture when one is given.
//!
//!   check-kernels SPEC.json
//!   check-kernels SPEC.json CAPTURE
//!
//! A captured `_ple_rows` (`name`, `fn`, or a `function` ending in `._ple_rows`) must carry a 64-digit `hash` and
//! `cubin_sha256`. Rows with neither field are derived spec rows and are left unchecked. CAPTURE is one set
//! (`aot.json` + `cubins/<hash>.cubin`) or a root of `sm<N>/` sets. Each listed cubin must exist and hash to its
//! `cubin_sha256`, and that sha must agree with the spec when the spec names the same hash. A set missing `_ple_rows`
//! fails here, which is the load error MissingTritonVariant.

const std = @import("std");

const hex_len = 64;

const Ctx = struct {
    arena: std.mem.Allocator,
    problems: std.ArrayList([]const u8) = .empty,
    /// hash -> cubin_sha256 for entries whose two fields agreed.
    hashes: std.StringHashMapUnmanaged([]const u8) = .empty,

    fn add(self: *Ctx, comptime fmt: []const u8, args: anytype) !void {
        try self.problems.append(self.arena, try std.fmt.allocPrint(self.arena, fmt, args));
    }

    fn finish(self: *Ctx) ![]const []const u8 {
        return self.problems.toOwnedSlice(self.arena);
    }
};

fn isHex64(text: []const u8) bool {
    if (text.len != hex_len) return false;
    for (text) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

fn isPle(obj: std.json.ObjectMap) bool {
    if (obj.get("name")) |v| if (v == .string and std.mem.eql(u8, v.string, "_ple_rows")) return true;
    if (obj.get("fn")) |v| if (v == .string and std.mem.eql(u8, v.string, "_ple_rows")) return true;
    if (obj.get("function")) |v| if (v == .string and std.mem.endsWith(u8, v.string, "._ple_rows")) return true;
    return false;
}

fn label(obj: std.json.ObjectMap) []const u8 {
    if (obj.get("function")) |v| if (v == .string) return v.string;
    if (obj.get("name")) |v| if (v == .string) return v.string;
    if (obj.get("fn")) |v| if (v == .string) return v.string;
    return "kernel";
}

/// A string field: null when absent, error.NotString when present as another JSON type.
fn stringField(obj: std.json.ObjectMap, key: []const u8) error{NotString}!?[]const u8 {
    const v = obj.get(key) orelse return null;
    if (v != .string) return error.NotString;
    return v.string;
}

/// Records one kernel. `true` when the entry is a valid `_ple_rows`, `false` when it is another valid captured row,
/// null when the row was skipped or rejected.
fn note(ctx: *Ctx, obj: std.json.ObjectMap, allow_derived: bool) !?bool {
    const ple = isPle(obj);
    const hash = stringField(obj, "hash") catch {
        try ctx.add("{s}: hash and cubin_sha256 must both be 64 hex digits", .{label(obj)});
        return null;
    };
    const sha = stringField(obj, "cubin_sha256") catch {
        try ctx.add("{s}: hash and cubin_sha256 must both be 64 hex digits", .{label(obj)});
        return null;
    };
    if (hash == null and sha == null) {
        if (ple) {
            try ctx.add("{s}: _ple_rows has no hash and cubin_sha256", .{label(obj)});
        } else if (!allow_derived) {
            try ctx.add("{s}: captured kernel has no hash and cubin_sha256", .{label(obj)});
        }
        return null;
    }
    if (hash == null or sha == null or !isHex64(hash.?) or !isHex64(sha.?)) {
        try ctx.add("{s}: hash and cubin_sha256 must both be 64 hex digits", .{label(obj)});
        return null;
    }
    if (ctx.hashes.get(hash.?)) |prev| {
        if (!std.ascii.eqlIgnoreCase(prev, sha.?)) try ctx.add("hash {s}: cubin_sha256 {s} and {s} disagree", .{ hash.?, prev, sha.? });
    } else try ctx.hashes.put(ctx.arena, hash.?, sha.?);
    return ple;
}

fn kernelItems(ctx: *Ctx, text: []const u8, what: []const u8) !?[]std.json.Value {
    const parsed = std.json.parseFromSlice(std.json.Value, ctx.arena, text, .{}) catch |e| {
        try ctx.add("{s}: {t}", .{ what, e });
        return null;
    };
    if (parsed.value != .object) {
        try ctx.add("{s}: no kernels array", .{what});
        return null;
    }
    const kernels = parsed.value.object.get("kernels") orelse {
        try ctx.add("{s}: no kernels array", .{what});
        return null;
    };
    if (kernels != .array) {
        try ctx.add("{s}: no kernels array", .{what});
        return null;
    }
    return kernels.array.items;
}

fn auditSpec(ctx: *Ctx, text: []const u8) !void {
    const items = (try kernelItems(ctx, text, "kernels.json")) orelse return;
    var saw_ple = false;
    var captured_ple = false;
    for (items) |item| {
        if (item != .object) {
            try ctx.add("kernels.json: a kernel entry is not an object", .{});
            continue;
        }
        if (isPle(item.object)) saw_ple = true;
        if (try note(ctx, item.object, true)) |ple| if (ple) {
            captured_ple = true;
        };
    }
    if (!captured_ple and !saw_ple) try ctx.add("kernels.json: no captured _ple_rows variant", .{});
}

fn isSm(name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, "sm") or name.len < 3) return false;
    for (name[2..]) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn auditSet(ctx: *Ctx, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(ctx.arena, &.{ dir, "aot.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, ctx.arena, .limited(1 << 26)) catch |e| {
        try ctx.add("{s}: {t}", .{ path, e });
        return;
    };
    const items = (try kernelItems(ctx, text, path)) orelse return;
    var saw_ple = false;
    var captured_ple = false;
    for (items) |item| {
        if (item != .object) {
            try ctx.add("{s}: a kernel entry is not an object", .{path});
            continue;
        }
        if (isPle(item.object)) saw_ple = true;
        const ple = try note(ctx, item.object, false) orelse continue;
        if (ple) captured_ple = true;
        const hash = (try stringField(item.object, "hash")).?;
        const sha = (try stringField(item.object, "cubin_sha256")).?;
        const cubin = try std.fmt.allocPrint(ctx.arena, "{s}/cubins/{s}.cubin", .{ dir, hash });
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, cubin, gpa, .limited(1 << 26)) catch |e| {
            try ctx.add("{s}: {t}", .{ cubin, e });
            continue;
        };
        defer gpa.free(bytes);
        var sum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
        const got = std.fmt.bytesToHex(sum, .lower);
        if (!std.ascii.eqlIgnoreCase(sha, &got)) try ctx.add("{s}: sha256 {s} != cubin_sha256 {s}", .{ cubin, &got, sha });
    }
    if (!captured_ple and !saw_ple) try ctx.add("{s}: no _ple_rows variant", .{path});
}

fn auditCapture(ctx: *Ctx, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch |e| {
        try ctx.add("capture {s}: {t}", .{ dir, e });
        return;
    };
    defer d.close(io);
    var found = false;
    if (d.access(io, "aot.json", .{})) |_| {
        found = true;
        try auditSet(ctx, gpa, io, dir);
    } else |_| {}
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory or !isSm(entry.name)) continue;
        found = true;
        const sub = try std.fs.path.join(ctx.arena, &.{ dir, entry.name });
        try auditSet(ctx, gpa, io, sub);
    }
    if (!found) try ctx.add("capture {s}: no aot.json and no sm<N> directory", .{dir});
}

/// Problems in a kernels.json document. Empty means the spec names a captured `_ple_rows` and its hashes agree.
pub fn specProblems(arena: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var ctx: Ctx = .{ .arena = arena };
    try auditSpec(&ctx, text);
    return ctx.finish();
}

/// Spec problems plus one capture directory's cubins. `gpa` backs cubin reads, which are freed per file.
pub fn captureProblems(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, spec_text: []const u8, dir: []const u8) ![]const []const u8 {
    var ctx: Ctx = .{ .arena = arena };
    try auditSpec(&ctx, spec_text);
    try auditCapture(&ctx, gpa, io, dir);
    return ctx.finish();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or args.len > 3) return error.Usage;
    const text = std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(1 << 26)) catch |e| {
        std.debug.print("unreadable kernel spec: {s} ({t})\n", .{ args[1], e });
        std.process.exit(1);
    };
    defer gpa.free(text);
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const problems = if (args.len == 3)
        try captureProblems(arena, gpa, io, text, args[2])
    else
        try specProblems(arena, text);
    for (problems) |p| std.debug.print("{s}\n", .{p});
    if (problems.len != 0) std.process.exit(1);
}

const ple_fn = "tensorfold.families.qwen4_exp.cuda.exl3_mm._ple_rows";
const hash_a = "ab3dcb9a54773a89e24842def8dace7aacfe9b21dc3570ad7316d529cab84f2b";
const sha_a = "e488725feb8abe3c09a13f60192b38a987bcffe7ae282c51f53b4c5e5e04cbab";

const Held = struct {
    arena: std.heap.ArenaAllocator,
    problems: []const []const u8 = &.{},

    fn spec(text: []const u8) !Held {
        var self: Held = .{ .arena = std.heap.ArenaAllocator.init(std.testing.allocator) };
        errdefer self.arena.deinit();
        self.problems = try specProblems(self.arena.allocator(), text);
        return self;
    }

    fn capture(spec_text: []const u8, dir: []const u8) !Held {
        var self: Held = .{ .arena = std.heap.ArenaAllocator.init(std.testing.allocator) };
        errdefer self.arena.deinit();
        self.problems = try captureProblems(self.arena.allocator(), std.testing.allocator, std.testing.io, spec_text, dir);
        return self;
    }

    fn deinit(self: *Held) void {
        self.arena.deinit();
    }
};

fn expectOk(problems: []const []const u8) !void {
    if (problems.len == 0) return;
    for (problems) |p| std.debug.print("unexpected: {s}\n", .{p});
    return error.TestUnexpectedResult;
}

fn expectHit(problems: []const []const u8, needle: []const u8) !void {
    for (problems) |p| if (std.mem.indexOf(u8, p, needle) != null) return;
    std.debug.print("missing \"{s}\" in:\n", .{needle});
    for (problems) |p| std.debug.print("  {s}\n", .{p});
    return error.TestUnexpectedResult;
}

test "a spec without _ple_rows is refused" {
    var held = try Held.spec(
        \\{"kernels":[{"function":"tensorfold.cuda.moe._router","name":"_router"}]}
    );
    defer held.deinit();
    try expectHit(held.problems, "no captured _ple_rows");
}

test "a _ple_rows row without a hash is refused" {
    var held = try Held.spec(
        \\{"kernels":[{"function":"tensorfold.families.qwen4_exp.cuda.exl3_mm._ple_rows","name":"_ple_rows"}]}
    );
    defer held.deinit();
    try expectHit(held.problems, "_ple_rows has no hash");
}

test "a captured _ple_rows passes with no cubin directory, and a derived row is skipped" {
    var held = try Held.spec(
        \\{"kernels":[
        \\  {"function":"tensorfold.cuda.moe._router","name":"_router","rule":"rows bucket"},
        \\  {"function":"tensorfold.families.qwen4_exp.cuda.exl3_mm._ple_rows","name":"_ple_rows","hash":"ab3dcb9a54773a89e24842def8dace7aacfe9b21dc3570ad7316d529cab84f2b","cubin_sha256":"e488725feb8abe3c09a13f60192b38a987bcffe7ae282c51f53b4c5e5e04cbab"}
        \\]}
    );
    defer held.deinit();
    try expectOk(held.problems);
}

test "the same hash with two cubin shas is refused" {
    var held = try Held.spec(
        \\{"kernels":[
        \\  {"function":"tensorfold.families.qwen4_exp.cuda.exl3_mm._ple_rows","name":"_ple_rows","hash":"ab3dcb9a54773a89e24842def8dace7aacfe9b21dc3570ad7316d529cab84f2b","cubin_sha256":"e488725feb8abe3c09a13f60192b38a987bcffe7ae282c51f53b4c5e5e04cbab"},
        \\  {"function":"tensorfold.cuda.moe._router","name":"_router","hash":"ab3dcb9a54773a89e24842def8dace7aacfe9b21dc3570ad7316d529cab84f2b","cubin_sha256":"8164187fa502c254a5d237ef02fd09372e75d06e52a45c831a63a058314e4aff"}
        \\]}
    );
    defer held.deinit();
    try expectHit(held.problems, "disagree");
}

fn shaOf(bytes: []const u8) [64]u8 {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
    return std.fmt.bytesToHex(sum, .lower);
}

fn pleSpec(a: std.mem.Allocator, sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"kernels":[{{"function":"{s}","name":"_ple_rows","hash":"{s}","cubin_sha256":"{s}"}}]}}
    , .{ ple_fn, hash_a, sha });
}

fn pleAot(a: std.mem.Allocator, sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"kernels":[{{"fn":"_ple_rows","name":"_ple_rows","function":"{s}","hash":"{s}","cubin_sha256":"{s}"}}]}}
    , .{ ple_fn, hash_a, sha });
}

test "a cubin whose bytes disagree with cubin_sha256 is refused" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "cubins", .default_dir);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const payload = "not-the-captured-cubin";
    const aot = try pleAot(a, sha_a);
    defer a.free(aot);
    const spec = try pleSpec(a, sha_a);
    defer a.free(spec);
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data = aot });
    try tmp.dir.writeFile(io, .{ .sub_path = "cubins/" ++ hash_a ++ ".cubin", .data = payload });
    var held = try Held.capture(spec, dir);
    defer held.deinit();
    try expectHit(held.problems, "sha256");
    try std.testing.expect(!std.mem.eql(u8, &shaOf(payload), sha_a));
}

test "matching cubin bytes pass, including an sm<N> capture root" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "sm121/cubins");
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const payload = "ple-rows-cubin";
    const sha = shaOf(payload);
    const aot = try pleAot(a, &sha);
    defer a.free(aot);
    const spec = try pleSpec(a, &sha);
    defer a.free(spec);
    try tmp.dir.writeFile(io, .{ .sub_path = "sm121/aot.json", .data = aot });
    try tmp.dir.writeFile(io, .{ .sub_path = "sm121/cubins/" ++ hash_a ++ ".cubin", .data = payload });
    var held = try Held.capture(spec, dir);
    defer held.deinit();
    try expectOk(held.problems);
}

test "a capture set with no _ple_rows cubin is refused" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const spec = try pleSpec(a, sha_a);
    defer a.free(spec);
    try tmp.dir.writeFile(io, .{ .sub_path = "aot.json", .data =
        \\{"kernels":[{"fn":"_router","name":"_router","function":"tensorfold.cuda.moe._router","hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","cubin_sha256":"e488725feb8abe3c09a13f60192b38a987bcffe7ae282c51f53b4c5e5e04cbab"}]}
    });
    var held = try Held.capture(spec, dir);
    defer held.deinit();
    try expectHit(held.problems, "no _ple_rows");
}
