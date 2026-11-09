//! Opt-in inline chain (TF_DSV41_RUNNER_INLINE_CHAIN): the last `<expr> =` after an earlier `=` statement.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const scan = @import("runner_scan.zig");
const t = std.testing;

test {
    _ = @import("runner_inline_stacked_test.zig"); // STATED's stacked column, from here: runner.zig stays at 600 lines
}

const on: scan.Config = .{ .inline_math = true, .inline_chain = true };

/// The line up to the model's `=`, the cell code, and CPython's value of it (`^` read as `**`).
const Golden = struct { []const u8, []const u8, []const u8 };

/// bn-011 smoke r0-583d0cd875f6: the ten steps the no-chain rule refused, each where the model then wrote this value.
pub const goldens = [_]Golden{
    .{ "Check modulo 2^18= 262144. Last? R mod 262144. Since 10^18 divisible by 262144? 10^18 =", "10^18", "1000000000000000000" },
    .{ "1000^3 = 1000000000 mod 262144. 262144*3814=", "262144*3814", "999817216" },
    .{ "1000^3 = 1000000000 mod 262144. 262144*3814=262144*(3800+14)=996,147,200+3,670,016=999,817,216; remainder 182,784. Or 213568*1000 mod. 213,568,000 /262144. 262144*814=", "262144*814", "213385216" },
    .{ "1000^4 = 1000000000000 mod? 182,784*1000=", "182784*1000", "182784000" },
    .{ "3^16= 43046721 mod 262144. 262144*164=", "262144*164", "42991616" },
    .{ "3^64 = 3433683820292512484657849089281 mod 262144. Square 147,073 mod. 147,073^2. (147,000+73)^2=", "(147000+73)^2", "21630467329" },
    .{ "3^128 = 11790184577738583171520872861412518665678211592275841109096961 mod. Square 179,457 mod 262144. 179,457^2. (179,000+457)^2=", "(179000+457)^2", "32204814849" },
    .{ "3^256 = 139008452377144732764939786789661303114218850808529137991604824430036072629766435941001769154109609521811665540548899435521 mod. Square 162,305 mod 262144. 162,305^2. (162,000+305)^2=", "(162000+305)^2", "26342913025" },
    .{ "Check modulo 5^18 maybe. R mod 5^18 should equal 3^400 mod 5^18. Let's verify maybe. 5^18=", "5^18", "3814697265625" },
    .{ "3^32 = 1853020188851841 mod M5. Need reduce. M5*? 3,814,697,265,625 * 485? Let's compute. 3^32 =", "3^32", "1853020188851841" },
    // other live replies, then the clause breaks it reads
    .{ "Term3: 30,292 * 65,536. Since 65,536 = 2^16. 30,292*65,536 = ", "30292*65536", "1985216512" },
    .{ "12 * 4 = 48; 48 * 7 =", "48 * 7", "336" },
    .{ "So 2^10 = 1024: 1024*3 =", "1024*3", "3072" },
    // a stop glued to the earlier value (`= 5.`), then a space
    .{ "2+3 = 5. 3*4 =", "3*4", "12" },
    .{ "So 7 * 8 = 56.  7*8 =", "7*8", "56" },
};

/// Lines no cell may open with the chain knob on.
const refused = [_][]const u8{
    // the right side of the earlier statement continues (T5's no-chain cases) or no clause ends
    "a = 3*4 =",
    "15*16 = 240*17 =",
    "x = 5, y = 3+4 =",
    "2ab = 2*512,484,657*849,089,281 =",
    // a running multiplier, a part of something, or an operand word
    "69,663^2 = 4,852,933,569. Times 10^6 =",
    "849^2=720,801; times 10^12 =",
    "This is 3+4*9=39 digits? Last 18 digits of 3^100 =",
    "High sum = 1,119,392,629,836,636. Multiply by 10^15 =",
    // congruence: a residue chain or a `≡`
    "10^3=1000 mod97=30. 10^6=",
    "10^3 = 1000 mod 97 = 30. 10^6 =",
    "2^10 = 1024 % 1000 = 24. 2^20 =",
    "r = 9*10^11 + 90. Since 10^11 ≡ -10 mod r? Because 9*10^11 + 90 =",
    // names beside an earlier `=`
    "We have a6=7^64=23,859,541. Need 7^128 =",
    "Need 7^128 = a6^2 mod M. Then 7^123 =",
    "r16 =746,814,714. Check: 977,326,736 * 3 =",
    // digit counts and concatenation
    "So 9 + 6*3 = 27. It has 7 groups (20 digits? 70 +6*3=",
    "So 3+4*9 = 39. Concatenate: 79 + 228162514 =",
    // CPython's value would differ or fail: floats, zero-led literals, power towers, negative exponents
    "Total 19783 s. 19783 = 5*3600 + 1783. Convert: 19783 /3600=",
    "i=8: 985+014+0=",
    "3^20=3,486,784,401. But 3^20^20 =",
    "spacing = 2^-53. So 1 - 2^-53 =",
    "Then 37*19.95 = 738.15? 19.95*30=",
    "11/2 = 5.5 * 2 =",
    "2+3 = 5.5*2 =",
    // the modulus of a mod step, either case
    "3^20=3,486,784,401. Mod 5^18 =",
    "So Mod 10^9 =",
    // comparisons anywhere on the line, and `==`
    "4e9 <3.8e12. Then 3*4 =",
    "x == 3. 4*5 =",
};

fn cellOf(a: std.mem.Allocator, line: []const u8, cfg: scan.Config) !?[]u8 {
    var s: scan.Scanner = .{ .cfg = cfg };
    defer s.deinit(a);
    const e = try s.feed(a, line, true);
    const c = e.closed orelse return null;
    if (c.form != .inline_expr or e.trigger == null or e.trigger.?.inline_space == null) return null;
    return try a.dupe(u8, c.code);
}

test "runner scan: the inline chain knob opens the last expression after a complete earlier statement" {
    for (goldens) |g| {
        const got = try cellOf(t.allocator, g[0], on) orelse {
            std.debug.print("no cell: {s}\n", .{g[0]});
            return error.TestExpectedInline;
        };
        defer t.allocator.free(got);
        try t.expectEqualStrings(g[1], got);
    }
    for (refused) |line| if (try cellOf(t.allocator, line, on)) |got| {
        defer t.allocator.free(got);
        std.debug.print("opened, must not: {s} -> {s}\n", .{ line, got });
        return error.TestUnexpectedInline;
    };
}

test "runner scan: the inline chain is off by default; T5's no-chain rule stands" {
    for (goldens[0..10]) |g| {
        // the period-glued `5^18=` has no earlier `=`, but needs the chain knob's full stop too
        try t.expectEqual(@as(?[]u8, null), try cellOf(t.allocator, g[0], .{ .inline_math = true }));
    }
    try t.expect(!(scan.Config{}).inline_chain);
}

fn bankPath() ![]const u8 {
    const p = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return std.mem.span(p);
}

test "runner proof: inline chain lines scan to cells that run on the graph bank and print CPython's value (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-inline-chain-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for (goldens) |g| {
        const code = try cellOf(t.allocator, g[0], on) orelse return error.TestExpectedInline;
        defer t.allocator.free(code);
        const result = try f.translate(f.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = code });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        if (result != .program or result.program.kind != .graph) {
            std.debug.print("not on the graph: {s}: {any}\n", .{ code, result });
            return error.TestExpectedGraph;
        }
        var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, &bank, result.program.bytes);
        defer machine.deinit();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        const end = try machine.run(4_000_000, &out, null);
        try t.expect(end.end == .halted);
        std.testing.expectEqualStrings(g[2], std.mem.trimEnd(u8, out.items, "\n")) catch |e| {
            std.debug.print("inline chain value: {s}\n", .{code});
            return e;
        };
    }
}
