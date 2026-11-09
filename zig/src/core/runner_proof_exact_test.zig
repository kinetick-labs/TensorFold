//! T1's exact inline fractions and decimals (8989951ab) through the real frontend, graph-v7 bank and Dispatch.
const std = @import("std");
const pt = @import("runner_proof_test.zig");
const frontend = @import("runner_frontend.zig");
const rapi = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const program = @import("runner_graph_program.zig");
const Count = pt.Count;
const Both = pt.Both;
const finish = pt.finish;

const gpa = std.testing.allocator;

test "runner proof: retained TRACE divmod and modulo requests on graph-v7 Dispatch" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var fallback: Count = .{ .inner = both.calc };
    var routed: route.Routed = .{ .inner = frontend.frontend(), .bank = &both.bank };
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = fallback.backend(), .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const front = routed.frontend();
    const run = dispatch.backend();
    const C = struct { code: []const u8, want: []const u8, display: []const u8, tuple: bool };
    const cases = [_]C{
        .{ .code = "(10**30-1)%97", .want = "84\n", .display = "(10^30 - 1) mod 97", .tuple = false },
        .{ .code = "(1234567890123456789)%(90000000009)", .want = "0\n", .display = "1234567890123456789 mod 90000000009", .tuple = false },
        .{ .code = "(12345678901234567890)%(10**10+1)", .want = "0\n", .display = "12345678901234567890 mod (10^10 + 1)", .tuple = false },
        .{ .code = "(12345678901234567890)%(900000000090)", .want = "0\n", .display = "12345678901234567890 mod 900000000090", .tuple = false },
        .{ .code = "(12345678901234567890)%101", .want = "0\n", .display = "12345678901234567890 mod 101", .tuple = false },
        .{ .code = "(192443283334088001)%(3814697265625)", .want = "3250375103626\n", .display = "192443283334088001 mod 3814697265625", .tuple = false },
        .{ .code = "(192443283334088001)%262144", .want = "116033\n", .display = "192443283334088001 mod 262144", .tuple = false },
        .{ .code = "(192443283334088001)%3814697265625", .want = "3250375103626\n", .display = "192443283334088001 mod 3814697265625", .tuple = false },
        .{ .code = "(192443283334088001)%5**18", .want = "3250375103626\n", .display = "192443283334088001 mod 5^18", .tuple = false },
        .{ .code = "(23859541*285189666)%1000000007", .want = "481071848\n", .display = "(23859541*285189666) mod 1000000007", .tuple = false },
        .{ .code = "(23859541*714810341)%1000000007", .want = "518928159\n", .display = "(23859541*714810341) mod 1000000007", .tuple = false },
        .{ .code = "(293681684639187364)%1000000007", .want = "583415590\n", .display = "293681684639187364 mod 1000000007", .tuple = false },
        .{ .code = "(315256100364237796)%1000000007", .want = "157445110\n", .display = "315256100364237796 mod 1000000007", .tuple = false },
        .{ .code = "(4852937748780900)%1000000007", .want = "714810341\n", .display = "4852937748780900 mod 1000000007", .tuple = false },
        .{ .code = "(510953823600536281)%1000000007", .want = "23859541\n", .display = "510953823600536281 mod 1000000007", .tuple = false },
        .{ .code = "(81333145593191556)%1000000007", .want = "23859541\n", .display = "81333145593191556 mod 1000000007", .tuple = false },
        .{ .code = "(98765432109876543210)%(12345678901234567890)", .want = "900000000090\n", .display = "98765432109876543210 mod 12345678901234567890", .tuple = false },
        .{ .code = "(9988776655443322)%(1122334455667788)", .want = "1010101010101018\n", .display = "9988776655443322 mod 1122334455667788", .tuple = false },
        .{ .code = "109739369%13717421", .want = "1\n", .display = "109739369 mod 13717421", .tuple = false },
        .{ .code = "112233445566770%88", .want = "66\n", .display = "112233445566770 mod 88", .tuple = false },
        .{ .code = "133600968182994384%1000000000", .want = "182994384\n", .display = "133600968182994384 mod 1000000000", .tuple = false },
        .{ .code = "192443283334088001%3814697265625", .want = "3250375103626\n", .display = "192443283334088001 mod 3814697265625", .tuple = false },
        .{ .code = "23859541*285189666%(1000000007)", .want = "481071848\n", .display = "23859541*285189666 mod 1000000007", .tuple = false },
        .{ .code = "32685677580%4826", .want = "0\n", .display = "32685677580 mod 4826", .tuple = false },
        .{ .code = "33232930569601%1000000007", .want = "930336977\n", .display = "33232930569601 mod 1000000007", .tuple = false },
        .{ .code = "45949729863572161%1000000007", .want = "541924058\n", .display = "45949729863572161 mod 1000000007", .tuple = false },
        .{ .code = "4852937748780900%1000000007", .want = "714810341\n", .display = "4852937748780900 mod 1000000007", .tuple = false },
        .{ .code = "58483618068999%998244353", .want = "474404141\n", .display = "58483618068999 mod 998244353", .tuple = false },
        .{ .code = "59865*200218%262144", .want = "40458\n", .display = "59865*200218 mod 262144", .tuple = false },
        .{ .code = "710656%15625", .want = "7531\n", .display = "710656 mod 15625", .tuple = false },
        .{ .code = "8913585%19665", .want = "5340\n", .display = "8913585 mod 19665", .tuple = false },
        .{ .code = "932748470*43046721%1000000000", .want = "151266870\n", .display = "932748470*43046721 mod 1000000000", .tuple = false },
        .{ .code = "divmod(10**24, 97)", .want = "(10309278350515463917525, 75)\n", .display = "(10^24) ÷ 97", .tuple = true },
        .{ .code = "divmod(10**30-1, 97)", .want = "(10309278350515463917525773195, 84)\n", .display = "(10^30 - 1) ÷ 97", .tuple = true },
        .{ .code = "divmod(1010101010101018, 112233445566770)", .want = "(9, 88)\n", .display = "1010101010101018 ÷ 112233445566770", .tuple = true },
        .{ .code = "divmod(123456789012345, 8913585)", .want = "(13850408, 19665)\n", .display = "123456789012345 ÷ 8913585", .tuple = true },
        .{ .code = "divmod(23777868243178464, 41832496)", .want = "(568406634, 0)\n", .display = "23777868243178464 ÷ 41832496", .tuple = true },
        .{ .code = "divmod(314159265358979323846264, 271828183)", .want = "(1155727349135756, 150034916)\n", .display = "314159265358979323846264 ÷ 271828183", .tuple = true },
        .{ .code = "divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)", .want = "(1155727349790921717910215407028, 23663814570294126502)\n", .display = "314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536", .tuple = true },
        .{ .code = "divmod(32685677580, -4826)", .want = "(-6772830, 0)\n", .display = "32685677580 ÷ -4826", .tuple = true },
        .{ .code = "divmod(33311225865831424, 998244353)", .want = "(33369811, 474404141)\n", .display = "33311225865831424 ÷ 998244353", .tuple = true },
        .{ .code = "divmod(40092776, 233)", .want = "(172072, 0)\n", .display = "40092776 ÷ 233", .tuple = true },
        .{ .code = "divmod(45949729863572161, 1000000007)", .want = "(45949729, 541924058)\n", .display = "45949729863572161 ÷ 1000000007", .tuple = true },
        .{ .code = "divmod(4852937748780900, 1000000007)", .want = "(4852937, 714810341)\n", .display = "4852937748780900 ÷ 1000000007", .tuple = true },
        .{ .code = "divmod(987654321098765432109876, 1000003)", .want = "(987651358144690998, 36882)\n", .display = "987654321098765432109876 ÷ 1000003", .tuple = true },
    };
    for (cases) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const tr = try front.translate(front.ctx, arena.allocator(), .{ .form = .repl, .lang = "python", .code = c.code });
        try std.testing.expect(tr == .program);
        try std.testing.expectEqual(@TypeOf(tr.program.kind).graph, tr.program.kind);
        const py_source = try std.fmt.allocPrint(gpa, "print({s})", .{c.code});
        defer gpa.free(py_source);
        const py = try std.process.run(gpa, std.testing.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", py_source }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
        defer gpa.free(py.stdout);
        defer gpa.free(py.stderr);
        try std.testing.expect(py.term.success());
        try std.testing.expectEqualStrings(c.want, py.stdout);
        const result = finish(run, try run.start(1, tr.program, 4_000_000));
        defer run.release(1);
        try std.testing.expectEqual(rapi.Status.ok, result.status);
        try std.testing.expectEqualStrings(c.want, result.out);
        const owned = (try @import("runner_trace.zig").owned(gpa, "[exact: {}]", c.display, c.tuple, result.out, false)) orelse return error.TestExpectedOwned;
        defer gpa.free(owned);
        std.debug.print("TRACE Dispatch exact {s}: {s}", .{ c.code, result.out });
    }
    try std.testing.expectEqual(@as(u32, 0), fallback.started);
}

test "runner proof: exact inline rational and decimal corpus and operand mutations use Dispatch" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    var fallback: Count = .{ .inner = both.calc };
    var routed: route.Routed = .{ .inner = frontend.frontend(), .bank = &both.bank };
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = fallback.backend(), .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const front = routed.frontend();
    const run = dispatch.backend();
    const C = struct { code: []const u8, want: []const u8, stale: bool };
    const cases = [_]C{
        .{ .code = "8613/9", .want = "957\n", .stale = false },
        .{ .code = "8613/(8)", .want = "1076.625\n", .stale = false },
        .{ .code = "8613/(8)", .want = "957\n", .stale = true },
        .{ .code = "8613/(10)", .want = "861.3\n", .stale = false },
        .{ .code = "8613/(10)", .want = "957\n", .stale = true },
        .{ .code = "112 /(28)", .want = "4\n", .stale = false },
        .{ .code = "(-23777868243178464)/(-41832496)", .want = "568406634\n", .stale = false },
        .{ .code = "392764052 ÷ 116", .want = "3385897\n", .stale = false },
        .{ .code = "8.3168*5.4828", .want = "45.59935104\n", .stale = false },
        .{ .code = "(4573/625)*5.4828", .want = "40.11655104\n", .stale = false },
        .{ .code = "(4573/625)*5.4828", .want = "45.59935104\n", .stale = true },
        .{ .code = "(5823/625)*5.4828", .want = "51.08215104\n", .stale = false },
        .{ .code = "(5823/625)*5.4828", .want = "45.59935104\n", .stale = true },
        .{ .code = "8.3168*(11207/2500)", .want = "37.28255104\n", .stale = false },
        .{ .code = "8.3168*(11207/2500)", .want = "45.59935104\n", .stale = true },
        .{ .code = "8.3168*(16207/2500)", .want = "53.91615104\n", .stale = false },
        .{ .code = "8.3168*(16207/2500)", .want = "45.59935104\n", .stale = true },
        .{ .code = "0.752 × 22.17", .want = "16.67184\n", .stale = false },
        .{ .code = "(-31/125) * 22.17", .want = "-5.49816\n", .stale = false },
        .{ .code = "(-31/125) * 22.17", .want = "16.67184\n", .stale = true },
        .{ .code = "(219/125) * 22.17", .want = "38.84184\n", .stale = false },
        .{ .code = "(219/125) * 22.17", .want = "16.67184\n", .stale = true },
        .{ .code = "0.752 * (2117/100)", .want = "15.91984\n", .stale = false },
        .{ .code = "0.752 * (2117/100)", .want = "16.67184\n", .stale = true },
        .{ .code = "0.752 * (2317/100)", .want = "17.42384\n", .stale = false },
        .{ .code = "0.752 * (2317/100)", .want = "16.67184\n", .stale = true },
        .{ .code = "(6)/12", .want = "0.5\n", .stale = false },
        .{ .code = "5/24 *3/(2)", .want = "0.3125\n", .stale = false },
        .{ .code = "5/24 *3/(4)", .want = "0.15625\n", .stale = false },
        .{ .code = "(6)/24 *3/3", .want = "0.25\n", .stale = false },
        .{ .code = "5/(25) *3/3", .want = "0.2\n", .stale = false },
        .{ .code = "(1 -1/2)", .want = "0.5\n", .stale = false },
        .{ .code = "((0) -1/2)", .want = "-0.5\n", .stale = false },
        .{ .code = "((0) -1/2)", .want = "0.5\n", .stale = true },
        .{ .code = "((2) -1/2)", .want = "1.5\n", .stale = false },
        .{ .code = "((2) -1/2)", .want = "0.5\n", .stale = true },
        .{ .code = "(1 -(0)/2)", .want = "1\n", .stale = false },
        .{ .code = "(1 -(0)/2)", .want = "0.5\n", .stale = true },
        .{ .code = "(1 -(2)/2)", .want = "0\n", .stale = false },
        .{ .code = "(1 -(2)/2)", .want = "0.5\n", .stale = true },
        .{ .code = "(1 -1/(1))", .want = "0\n", .stale = false },
        .{ .code = "(1 -1/(1))", .want = "0.5\n", .stale = true },
        .{ .code = "-1/4", .want = "-0.25\n", .stale = false },
        .{ .code = "-1/(5)", .want = "-0.2\n", .stale = false },
        .{ .code = "-1/(5)", .want = "-0.25\n", .stale = true },
        .{ .code = "-(0)/4", .want = "0\n", .stale = false },
        .{ .code = "-(0)/4", .want = "-0.25\n", .stale = true },
        .{ .code = "-(2)/4", .want = "-0.5\n", .stale = false },
        .{ .code = "-(2)/4", .want = "-0.25\n", .stale = true },
        .{ .code = "-1/(5)", .want = "-0.2\n", .stale = false },
        .{ .code = "-(0)/6", .want = "0\n", .stale = false },
        .{ .code = "-1/8", .want = "-0.125\n", .stale = false },
        .{ .code = "-(0)/8", .want = "0\n", .stale = false },
        .{ .code = "-(0)/8", .want = "-0.125\n", .stale = true },
        .{ .code = "-(2)/8", .want = "-0.25\n", .stale = false },
        .{ .code = "-(2)/8", .want = "-0.125\n", .stale = true },
        .{ .code = "37*19.95", .want = "738.15\n", .stale = false },
        .{ .code = "(36)*19.95", .want = "718.2\n", .stale = false },
        .{ .code = "(36)*19.95", .want = "738.15\n", .stale = true },
        .{ .code = "(38)*19.95", .want = "758.1\n", .stale = false },
        .{ .code = "(38)*19.95", .want = "738.15\n", .stale = true },
        .{ .code = "37*(379/20)", .want = "701.15\n", .stale = false },
        .{ .code = "37*(379/20)", .want = "738.15\n", .stale = true },
        .{ .code = "37*(419/20)", .want = "775.15\n", .stale = false },
        .{ .code = "37*(419/20)", .want = "738.15\n", .stale = true },
        .{ .code = "18*8.40", .want = "151.2\n", .stale = false },
        .{ .code = "(17)*8.40", .want = "142.8\n", .stale = false },
        .{ .code = "(17)*8.40", .want = "151.2\n", .stale = true },
        .{ .code = "(19)*8.40", .want = "159.6\n", .stale = false },
        .{ .code = "(19)*8.40", .want = "151.2\n", .stale = true },
        .{ .code = "18*(37/5)", .want = "133.2\n", .stale = false },
        .{ .code = "18*(37/5)", .want = "151.2\n", .stale = true },
        .{ .code = "18*(47/5)", .want = "169.2\n", .stale = false },
        .{ .code = "18*(47/5)", .want = "151.2\n", .stale = true },
        .{ .code = "0/3", .want = "0\n", .stale = false },
        .{ .code = "0/(2)", .want = "0\n", .stale = false },
        .{ .code = "0/(4)", .want = "0\n", .stale = false },
        .{ .code = "-0.00*2.0", .want = "0\n", .stale = false },
        .{ .code = "-0.00*(1)", .want = "0\n", .stale = false },
        .{ .code = "-0.00*(3)", .want = "0\n", .stale = false },
        .{ .code = "-(-1)*2.0", .want = "2\n", .stale = false },
        .{ .code = "-(-1)*2.0", .want = "0\n", .stale = true },
        .{ .code = "-(1)*2.0", .want = "-2\n", .stale = false },
        .{ .code = "-(1)*2.0", .want = "0\n", .stale = true },
        .{ .code = "-2**2/(2)", .want = "-2\n", .stale = false },
        .{ .code = "-2**2/(4)", .want = "-1\n", .stale = false },
        .{ .code = "-(3)**2/3", .want = "-3\n", .stale = false },
        .{ .code = "(-2)**2/(2)", .want = "2\n", .stale = false },
        .{ .code = "(-2)**2/(4)", .want = "1\n", .stale = false },
        .{ .code = "(-(3))**2/3", .want = "3\n", .stale = false },
        .{ .code = "(0)/-3", .want = "0\n", .stale = false },
        .{ .code = "1/-(2)", .want = "-0.5\n", .stale = false },
        .{ .code = "1/-(4)", .want = "-0.25\n", .stale = false },
        .{ .code = "-(0)/-3", .want = "0\n", .stale = false },
        .{ .code = "-1/-(2)", .want = "0.5\n", .stale = false },
        .{ .code = "-1/-(4)", .want = "0.25\n", .stale = false },
        .{ .code = "0.1+0.2", .want = "0.3\n", .stale = false },
        .{ .code = "(-9/10)+0.2", .want = "-0.7\n", .stale = false },
        .{ .code = "(-9/10)+0.2", .want = "0.3\n", .stale = true },
        .{ .code = "(11/10)+0.2", .want = "1.3\n", .stale = false },
        .{ .code = "(11/10)+0.2", .want = "0.3\n", .stale = true },
        .{ .code = "0.1+(-4/5)", .want = "-0.7\n", .stale = false },
        .{ .code = "0.1+(-4/5)", .want = "0.3\n", .stale = true },
        .{ .code = "0.1+(6/5)", .want = "1.3\n", .stale = false },
        .{ .code = "0.1+(6/5)", .want = "0.3\n", .stale = true },
        .{ .code = "999.99+0.01", .want = "1000\n", .stale = false },
        .{ .code = "(99899/100)+0.01", .want = "999\n", .stale = false },
        .{ .code = "(99899/100)+0.01", .want = "1000\n", .stale = true },
        .{ .code = "(100099/100)+0.01", .want = "1001\n", .stale = false },
        .{ .code = "(100099/100)+0.01", .want = "1000\n", .stale = true },
        .{ .code = "999.99+(-99/100)", .want = "999\n", .stale = false },
        .{ .code = "999.99+(-99/100)", .want = "1000\n", .stale = true },
        .{ .code = "999.99+(101/100)", .want = "1001\n", .stale = false },
        .{ .code = "999.99+(101/100)", .want = "1000\n", .stale = true },
        .{ .code = "1.0/4", .want = "0.25\n", .stale = false },
        .{ .code = "(0)/4", .want = "0\n", .stale = false },
        .{ .code = "(0)/4", .want = "0.25\n", .stale = true },
        .{ .code = "(2)/4", .want = "0.5\n", .stale = false },
        .{ .code = "(2)/4", .want = "0.25\n", .stale = true },
        .{ .code = "1.0/(5)", .want = "0.2\n", .stale = false },
        .{ .code = "1.0/(5)", .want = "0.25\n", .stale = true },
        .{ .code = "0.000000000000000001*1", .want = "0.000000000000000001\n", .stale = false },
        .{ .code = "(-999999999999999999/1000000000000000000)*1", .want = "-0.999999999999999999\n", .stale = false },
        .{ .code = "(-999999999999999999/1000000000000000000)*1", .want = "0.000000000000000001\n", .stale = true },
        .{ .code = "(1000000000000000001/1000000000000000000)*1", .want = "1.000000000000000001\n", .stale = false },
        .{ .code = "(1000000000000000001/1000000000000000000)*1", .want = "0.000000000000000001\n", .stale = true },
        .{ .code = "0.000000000000000001*(0)", .want = "0\n", .stale = false },
        .{ .code = "0.000000000000000001*(0)", .want = "0.000000000000000001\n", .stale = true },
        .{ .code = "0.000000000000000001*(2)", .want = "0.000000000000000002\n", .stale = false },
        .{ .code = "0.000000000000000001*(2)", .want = "0.000000000000000001\n", .stale = true },
        .{ .code = "1.5-2.75", .want = "-1.25\n", .stale = false },
        .{ .code = "(1/2)-2.75", .want = "-2.25\n", .stale = false },
        .{ .code = "(1/2)-2.75", .want = "-1.25\n", .stale = true },
        .{ .code = "(5/2)-2.75", .want = "-0.25\n", .stale = false },
        .{ .code = "(5/2)-2.75", .want = "-1.25\n", .stale = true },
        .{ .code = "1.5-(7/4)", .want = "-0.25\n", .stale = false },
        .{ .code = "1.5-(7/4)", .want = "-1.25\n", .stale = true },
        .{ .code = "1.5-(15/4)", .want = "-2.25\n", .stale = false },
        .{ .code = "1.5-(15/4)", .want = "-1.25\n", .stale = true },
        .{ .code = "(1/2)+((0)/3)", .want = "0.5\n", .stale = false },
        .{ .code = "(1/2)+(1/(2))", .want = "1\n", .stale = false },
        .{ .code = "(1/2)+(1/(4))", .want = "0.75\n", .stale = false },
        .{ .code = "((0)/2)/(3/4)", .want = "0\n", .stale = false },
        .{ .code = "(1/2)/((2)/4)", .want = "1\n", .stale = false },
        .{ .code = "(1/2)/((4)/4)", .want = "0.5\n", .stale = false },
        .{ .code = "(1/2)/(3/(3))", .want = "0.5\n", .stale = false },
        .{ .code = "((3)/3)**3", .want = "1\n", .stale = false },
        .{ .code = "(2/(2))**3", .want = "1\n", .stale = false },
        .{ .code = "(2/(4))**3", .want = "0.125\n", .stale = false },
        .{ .code = "(0.1/2)+(0.3/2)", .want = "0.2\n", .stale = false },
        .{ .code = "((-9/10)/2)+(0.3/2)", .want = "-0.3\n", .stale = false },
        .{ .code = "((-9/10)/2)+(0.3/2)", .want = "0.2\n", .stale = true },
        .{ .code = "((11/10)/2)+(0.3/2)", .want = "0.7\n", .stale = false },
        .{ .code = "((11/10)/2)+(0.3/2)", .want = "0.2\n", .stale = true },
        .{ .code = "(0.1/(1))+(0.3/2)", .want = "0.25\n", .stale = false },
        .{ .code = "(0.1/(1))+(0.3/2)", .want = "0.2\n", .stale = true },
        .{ .code = "(0.1/2)+((-7/10)/2)", .want = "-0.3\n", .stale = false },
        .{ .code = "(0.1/2)+((-7/10)/2)", .want = "0.2\n", .stale = true },
        .{ .code = "(0.1/2)+((13/10)/2)", .want = "0.7\n", .stale = false },
        .{ .code = "(0.1/2)+((13/10)/2)", .want = "0.2\n", .stale = true },
        .{ .code = "(0.1/2)+(0.3/(1))", .want = "0.35\n", .stale = false },
        .{ .code = "(0.1/2)+(0.3/(1))", .want = "0.2\n", .stale = true },
        .{ .code = "(0.1/2)+(0.3/(3))", .want = "0.15\n", .stale = false },
        .{ .code = "(0.1/2)+(0.3/(3))", .want = "0.2\n", .stale = true },
        .{ .code = "889.35*0.075", .want = "66.70125\n", .stale = false },
        .{ .code = "(17767/20)*0.075", .want = "66.62625\n", .stale = false },
        .{ .code = "(17767/20)*0.075", .want = "66.70125\n", .stale = true },
        .{ .code = "(17807/20)*0.075", .want = "66.77625\n", .stale = false },
        .{ .code = "(17807/20)*0.075", .want = "66.70125\n", .stale = true },
        .{ .code = "889.35*(-37/40)", .want = "-822.64875\n", .stale = false },
        .{ .code = "889.35*(-37/40)", .want = "66.70125\n", .stale = true },
        .{ .code = "889.35*(43/40)", .want = "956.05125\n", .stale = false },
        .{ .code = "889.35*(43/40)", .want = "66.70125\n", .stale = true },
        .{ .code = "889.35*0.925", .want = "822.64875\n", .stale = false },
        .{ .code = "(17767/20)*0.925", .want = "821.72375\n", .stale = false },
        .{ .code = "(17767/20)*0.925", .want = "822.64875\n", .stale = true },
        .{ .code = "(17807/20)*0.925", .want = "823.57375\n", .stale = false },
        .{ .code = "(17807/20)*0.925", .want = "822.64875\n", .stale = true },
        .{ .code = "889.35*(-3/40)", .want = "-66.70125\n", .stale = false },
        .{ .code = "889.35*(-3/40)", .want = "822.64875\n", .stale = true },
        .{ .code = "889.35*(77/40)", .want = "1711.99875\n", .stale = false },
        .{ .code = "889.35*(77/40)", .want = "822.64875\n", .stale = true },
        .{ .code = "822.64875*1.06", .want = "872.007675\n", .stale = false },
        .{ .code = "(657319/800)*1.06", .want = "870.947675\n", .stale = false },
        .{ .code = "(657319/800)*1.06", .want = "872.007675\n", .stale = true },
        .{ .code = "(658919/800)*1.06", .want = "873.067675\n", .stale = false },
        .{ .code = "(658919/800)*1.06", .want = "872.007675\n", .stale = true },
        .{ .code = "822.64875*(3/50)", .want = "49.358925\n", .stale = false },
        .{ .code = "822.64875*(3/50)", .want = "872.007675\n", .stale = true },
        .{ .code = "822.64875*(103/50)", .want = "1694.656425\n", .stale = false },
        .{ .code = "822.64875*(103/50)", .want = "872.007675\n", .stale = true },
        .{ .code = "80. /2", .want = "40\n", .stale = false },
        .{ .code = "(79) /2", .want = "39.5\n", .stale = false },
        .{ .code = "(79) /2", .want = "40\n", .stale = true },
        .{ .code = "(81) /2", .want = "40.5\n", .stale = false },
        .{ .code = "(81) /2", .want = "40\n", .stale = true },
        .{ .code = "80. /(1)", .want = "80\n", .stale = false },
        .{ .code = "80. /(1)", .want = "40\n", .stale = true },
        .{ .code = "134709035556. /1000", .want = "134709035.556\n", .stale = false },
        .{ .code = "(134709035555) /1000", .want = "134709035.555\n", .stale = false },
        .{ .code = "(134709035555) /1000", .want = "134709035.556\n", .stale = true },
        .{ .code = "(134709035557) /1000", .want = "134709035.557\n", .stale = false },
        .{ .code = "(134709035557) /1000", .want = "134709035.556\n", .stale = true },
        .{ .code = ".5*2", .want = "1\n", .stale = false },
        .{ .code = "(-1/2)*2", .want = "-1\n", .stale = false },
        .{ .code = "(-1/2)*2", .want = "1\n", .stale = true },
        .{ .code = "(3/2)*2", .want = "3\n", .stale = false },
        .{ .code = "(3/2)*2", .want = "1\n", .stale = true },
        .{ .code = ".5*(1)", .want = "0.5\n", .stale = false },
        .{ .code = ".5*(1)", .want = "1\n", .stale = true },
        .{ .code = ".5*(3)", .want = "1.5\n", .stale = false },
        .{ .code = ".5*(3)", .want = "1\n", .stale = true },
        .{ .code = "5.*2", .want = "10\n", .stale = false },
        .{ .code = "(4)*2", .want = "8\n", .stale = false },
        .{ .code = "(4)*2", .want = "10\n", .stale = true },
        .{ .code = "(6)*2", .want = "12\n", .stale = false },
        .{ .code = "(6)*2", .want = "10\n", .stale = true },
        .{ .code = "5.*(1)", .want = "5\n", .stale = false },
        .{ .code = "5.*(1)", .want = "10\n", .stale = true },
        .{ .code = "5.*(3)", .want = "15\n", .stale = false },
        .{ .code = "5.*(3)", .want = "10\n", .stale = true },
        .{ .code = ".75/.25", .want = "3\n", .stale = false },
        .{ .code = "(-1/4)/.25", .want = "-1\n", .stale = false },
        .{ .code = "(-1/4)/.25", .want = "3\n", .stale = true },
        .{ .code = "(7/4)/.25", .want = "7\n", .stale = false },
        .{ .code = "(7/4)/.25", .want = "3\n", .stale = true },
        .{ .code = ".75/(-3/4)", .want = "-1\n", .stale = false },
        .{ .code = ".75/(-3/4)", .want = "3\n", .stale = true },
        .{ .code = ".75/(5/4)", .want = "0.6\n", .stale = false },
        .{ .code = ".75/(5/4)", .want = "3\n", .stale = true },
        .{ .code = "-0.*2.0", .want = "0\n", .stale = false },
        .{ .code = "-0.*(1)", .want = "0\n", .stale = false },
        .{ .code = "-0.*(3)", .want = "0\n", .stale = false },
        .{ .code = "-(-1)*2.0", .want = "2\n", .stale = false },
        .{ .code = "-(-1)*2.0", .want = "0\n", .stale = true },
        .{ .code = "-(1)*2.0", .want = "-2\n", .stale = false },
        .{ .code = "-(1)*2.0", .want = "0\n", .stale = true },
        .{ .code = "1.+2.", .want = "3\n", .stale = false },
        .{ .code = "(0)+2.", .want = "2\n", .stale = false },
        .{ .code = "(0)+2.", .want = "3\n", .stale = true },
        .{ .code = "(2)+2.", .want = "4\n", .stale = false },
        .{ .code = "(2)+2.", .want = "3\n", .stale = true },
        .{ .code = "1.+(1)", .want = "2\n", .stale = false },
        .{ .code = "1.+(1)", .want = "3\n", .stale = true },
        .{ .code = "1.+(3)", .want = "4\n", .stale = false },
        .{ .code = "1.+(3)", .want = "3\n", .stale = true },
        .{ .code = "5./4", .want = "1.25\n", .stale = false },
        .{ .code = "(4)/4", .want = "1\n", .stale = false },
        .{ .code = "(4)/4", .want = "1.25\n", .stale = true },
        .{ .code = "(6)/4", .want = "1.5\n", .stale = false },
        .{ .code = "(6)/4", .want = "1.25\n", .stale = true },
        .{ .code = "5./(5)", .want = "1\n", .stale = false },
        .{ .code = "5./(5)", .want = "1.25\n", .stale = true },
        .{ .code = "-.125*8", .want = "-1\n", .stale = false },
        .{ .code = "-.125*(7)", .want = "-0.875\n", .stale = false },
        .{ .code = "-.125*(7)", .want = "-1\n", .stale = true },
        .{ .code = "-.125*(9)", .want = "-1.125\n", .stale = false },
        .{ .code = "-.125*(9)", .want = "-1\n", .stale = true },
        .{ .code = "-(-7/8)*8", .want = "7\n", .stale = false },
        .{ .code = "-(-7/8)*8", .want = "-1\n", .stale = true },
        .{ .code = "-(9/8)*8", .want = "-9\n", .stale = false },
        .{ .code = "-(9/8)*8", .want = "-1\n", .stale = true },
        .{ .code = "(.5+.25)*4", .want = "3\n", .stale = false },
        .{ .code = "(.5+.25)*(3)", .want = "2.25\n", .stale = false },
        .{ .code = "(.5+.25)*(3)", .want = "3\n", .stale = true },
        .{ .code = "(.5+.25)*(5)", .want = "3.75\n", .stale = false },
        .{ .code = "(.5+.25)*(5)", .want = "3\n", .stale = true },
        .{ .code = "((-1/2)+.25)*4", .want = "-1\n", .stale = false },
        .{ .code = "((-1/2)+.25)*4", .want = "3\n", .stale = true },
        .{ .code = "((3/2)+.25)*4", .want = "7\n", .stale = false },
        .{ .code = "((3/2)+.25)*4", .want = "3\n", .stale = true },
        .{ .code = "(.5+(-3/4))*4", .want = "-1\n", .stale = false },
        .{ .code = "(.5+(-3/4))*4", .want = "3\n", .stale = true },
        .{ .code = "(.5+(5/4))*4", .want = "7\n", .stale = false },
        .{ .code = "(.5+(5/4))*4", .want = "3\n", .stale = true },
    };
    for (cases) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const tr = try front.translate(front.ctx, arena.allocator(), .{ .form = .inline_expr, .lang = "python", .code = c.code });
        try std.testing.expect(tr == .program);
        try std.testing.expectEqual(@TypeOf(tr.program.kind).graph, tr.program.kind);
        const result = finish(run, try run.start(1, tr.program, 4_000_000));
        defer run.release(1);
        std.debug.print("exact inline Dispatch {s}: {s}", .{ c.code, result.out });
        try std.testing.expectEqual(rapi.Status.ok, result.status);
        if (c.stale) try std.testing.expect(!std.mem.eql(u8, c.want, result.out)) else try std.testing.expectEqualStrings(c.want, result.out);
    }
    try std.testing.expectEqual(@as(u32, 0), fallback.started);
}

test "runner proof: exact inline domain width refusal and Python form separation" {
    var both: Both = undefined;
    try both.open();
    defer both.close();
    const direct = graph.directFrontend(&both.bank);
    const bind = @import("runner_graph_bind.zig");
    const unsupported = [_][]const u8{
        "1e-5",
        ".*2",
        "5..*2",
        "x/2",
        "2**2**2/3",
        "2**-2/3",
        "999999999999999999999999999999999999999999999999999999999999999/2",
        "0.0000000000000000001",
    };
    for (unsupported) |code| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const tr = try direct.translate(direct.ctx, arena.allocator(), .{ .form = .inline_expr, .lang = "python", .code = code });
        try std.testing.expect(tr == .abstain);
    }
    // Only this fallback is form-specific; other existing Python float routes retain their contract.
    for ([_]rapi.Form{ .repl, .fence }) |form| {
        const packet_bytes = try bind.cell(gpa, &both.bank, rapi.Cell{ .form = form, .lang = "python", .code = "0.1+0.2" });
        if (packet_bytes) |bytes| {
            gpa.free(bytes);
            return error.TestExpectedNoExactInlineBinding;
        }
    }
    const invalid = [_][]const u8{ "1/0", "1/(1/0)", "(1/0)^0", "0.1/3", "1.0/3" };
    var fallback: Count = .{ .inner = both.calc };
    var routed: route.Routed = .{ .inner = frontend.frontend(), .bank = &both.bank };
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = fallback.backend(), .graph = both.graph, .bank = &both.bank };
    defer dispatch.deinit();
    const front = routed.frontend();
    const run = dispatch.backend();
    for (invalid) |code| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const tr = try front.translate(front.ctx, arena.allocator(), .{ .form = .inline_expr, .lang = "python", .code = code });
        try std.testing.expect(tr == .program);
        try std.testing.expectEqual(@TypeOf(tr.program.kind).graph, tr.program.kind);
        const result = finish(run, try run.start(1, tr.program, 4_000_000));
        defer run.release(1);
        try std.testing.expectEqual(rapi.Status.abstain, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.out.len);
    }
    try std.testing.expectEqual(@as(u32, 0), fallback.started);
}
