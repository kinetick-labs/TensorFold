//! Form trace: real census reasoning lines own `[exact: …]`, ambiguous ones abstain; scanner and lane-host replies.
const std = @import("std");
const trace = @import("runner_trace.zig");
const scan = @import("runner_scan.zig");
const rapi = @import("runner_api.zig");
const rt = @import("runner_test.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const runner = @import("runner.zig");

const gpa = std.testing.allocator;

test {
    _ = @import("runner_trace_bind_test.zig"); // binding recovery, from here: runner.zig stays at 600 lines
}

/// The Python source a line states (after `lines` defined its names), or null.
fn stated(lines: []const []const u8) !?[]u8 {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    var last: ?[]u8 = null;
    for (lines) |l| {
        if (last) |x| gpa.free(x);
        last = null;
        if (try t.line(l)) |f| last = try gpa.dupe(u8, f.source);
    }
    return last;
}

fn expectStated(lines: []const []const u8, want: []const u8) !void {
    const got = try stated(lines) orelse {
        std.debug.print("abstained: {s}\n", .{lines[lines.len - 1]});
        return error.TestExpectedStatement;
    };
    defer gpa.free(got);
    try std.testing.expectEqualStrings(want, got);
}

fn expectNone(lines: []const []const u8) !void {
    if (try stated(lines)) |got| {
        defer gpa.free(got);
        std.debug.print("stated {s} from: {s}\n", .{ got, lines[lines.len - 1] });
        return error.TestUnexpectedStatement;
    }
}

// Verbatim T6 census reasoning lines (t6-census/raw).
const bn019 = "We need answer division. Need compute quotient remainder for huge integer. Need be accurate. We can use big integer arithmetic manually. Need divide A=314159265358979323846264338327950288419716939937510 by B=271828182845904523536. Need quotient and remainder.";
const bn001 = "We need answer exact multiplication. Need compute. Need be careful. Need multiply A=12345678901234567890 (20 digits?) B=9876543210987654321 (19 digits). Need exact product.";
const bn008 = "We need answer compute 7^123 mod 1000000007. Need calculate accurately. Need provide maybe concise. Let's compute carefully. Mod p=1,000,000,007 prime. Need exponentiation.";
const bn015 = "Let A = 0xFEDCBA9876543210, B=0x0123456789ABCDEF. N = A*2^64 + B.";
const bn017 = "We need answer exact product 41*42*...*60. Need compute. Need be careful. Could express as 60!/40! maybe exact integer. User asks exact product. Need provide number. Let's compute accurately.";
const bn020 = "We need answer. Need compute exact. Need be careful. Expression (10^80 + 123456789)^2 - 10^160. Expand: 2*123456789*10^80 + 123456789^2. Need exact decimal. Need compute 123456789^2. Let's calculate. 123,456,789^2. Known? 123456789^2 = 15241578750190521? Let's verify. 123456789*123456789.";
const rc009 = "We need answer. Need compute exact value 2^100 - 3^50. Need run code? As AI no execution environment? We can compute manually with big ints, verify. Need exact. Let's calculate carefully. Need maybe use Python mentally? Let's derive.";
const ar008 = "We need answer compute. Need be careful. 2^37 - 3^19. Let's calculate.";
const bn002 = "Need divide 10^30 -1 by 97. Let's calculate exact quotient. N = 999,999,999,999,999,999,999,999,999,999 (30 nines). Divide by 97. Need quotient. We can do long division or use formula. Need exact.";
const bn006 = "We need answer arithmetic. Need be careful. Need subtract A=876543210987654321098765432109 from B=1000000000000000000000000000000 (1 followed? Let's count). Need compute B - A.";
const bn010 = [_][]const u8{
    "Let's denote A=12,345,678,901,234,567,890 (20 digits? Let's count). B=98,765,432,109,876,543,210 (20 digits? Actually 98 quintillion = 20 digits). Need gcd.",
    "So gcd(A,B)=gcd(A, 900,000,000,090). Let C=900,000,000,090.",
    "Need gcd(A,C). A=12,345,678,901,234,567,890. C ~9e11. Need compute A mod C. Could use factors.",
};

test "runner trace: census statements extract exactly what the line states" {
    try expectStated(&.{bn019}, "divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)");
    try expectStated(&.{bn001}, "(12345678901234567890) * (9876543210987654321)");
    try expectStated(&.{bn008}, "7**123%1000000007");
    try expectStated(&.{bn015}, "(0xFEDCBA9876543210)*2**64+(0x0123456789ABCDEF)");
    try expectStated(&.{bn017}, "41*42*43*44*45*46*47*48*49*50*51*52*53*54*55*56*57*58*59*60");
    try expectStated(&.{rc009}, "2**100-3**50");
    try expectStated(&.{ar008}, "2**37-3**19");
    try expectStated(&.{bn002}, "divmod(10**30-1, 97)");
    try expectStated(&bn010, "(12345678901234567890)%(900000000090)");
}

test "runner trace: a statement too wide for the frontend yields to the line's next one (big-number-020)" {
    try expectStated(&.{bn020}, "123456789**2");
}

test "runner trace: the display line and the owned text" {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    const f = (try t.line(bn019)).?;
    try std.testing.expect(f.divmod);
    try std.testing.expectEqualStrings("314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536", f.display);
    const line = (try trace.owned(gpa, "", f.display, true, "(1155727349790921717910215407028, 23663814570294126502)\n", false)).?;
    defer gpa.free(line);
    try std.testing.expectEqualStrings("[exact, computed by the engine: 314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536 = 1155727349790921717910215407028 remainder 23663814570294126502]\n", line);
    const g = (try t.line(bn008)).?;
    try std.testing.expectEqualStrings("7^123 mod 1000000007", g.display);
    var fresh = trace.Tracer.init(gpa); // bn019 above defined A and B; bn015's other A poisons it
    defer fresh.deinit();
    const h = (try fresh.line(bn015)).?;
    try std.testing.expectEqualStrings("0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF", h.display);
    for ([_][]const u8{ "3.75\n", "1\n2\n", "", "(1, 2)\n" }) |out| try std.testing.expectEqual(null, try trace.owned(gpa, "", "x", false, out, false));
    for ([_][]const u8{ "(1, 2, 3)\n", "7\n", "(1,2)\n" }) |out| try std.testing.expectEqual(null, try trace.owned(gpa, "", "x", true, out, false));
}

test "runner trace: doubts, lists, hypotheticals and two readings abstain" {
    try expectNone(&.{bn006}); // "(1 followed? Let's count)" sits in B's sentence
    try expectNone(&.{ "Let D = 271828182845904523536.", "Let's compute D*155, D*156 etc." });
    try expectNone(&.{ "Let M = 1234567890.", "If P = M*(10^10+1) then fine.", "Need compute P*2." });
    try expectNone(&.{ "Let A = 5. Let M = 7.", "Need compute A + 3 mod M." });
    try expectNone(&.{"Need compute 2^3^2."});
    try expectNone(&.{ "Let M = 7.", "Need compute (5 mod M)*2." });
    try expectNone(&.{"Need compute A ~2.718e20 * 3."});
    try expectNone(&.{"Need compute 3^400 mod 10^18? Not sure."});
    try expectNone(&.{"Need compute 2^64*3, right?"}); // the doubt sits after the phrase, in its sentence
    try expectNone(&.{ "Let A = 12345678901 (≈1.2e10).", "Need compute A*1000003." }); // an approximate definition binds nothing
    try expectNone(&.{"Need compute 1.5*2."});
    try expectNone(&.{"Need compute 012*3."});
    try expectNone(&.{"Need compute [314,159,265]*2."});
    try expectNone(&.{"Need compute positions 12-20."});
    try expectNone(&.{"Need compute 12,34 + 5."});
    try expectNone(&.{"Need compute 2^N."});
    try expectNone(&.{"Need compute 41*43*...*60."});
}

test "runner trace: a name after an operator is no definition; a redefinition poisons the name" {
    try expectStated(&.{ "Let B = 5.", "A * B = 37,276,043,023,296,000 * 273,589,847,231,500,800.", "Need compute B*1000003." }, "(5)*1000003");
    try expectNone(&.{ "Let A = 5.", "Let A = 6.", "Need compute A*1000003." });
    try expectStated(&.{ "Let A = 12,345.", "A = 12345.", "Need compute A*1000003." }, "(12345)*1000003");
}

test "runner trace: any reassignment poisons a name, so a stale value never states (live big-number-015)" {
    // `Step k N=`: a label before the name (trace3 run2 s3 claimed 280223976814164*65536, 12,816 off the model's N)
    try expectNone(&.{ "Let N=280,223,976,814,164.", "Step3 N=18,364,758,544,493,064,720.", "Compute N*65536." });
    // `N = f(N)`: a loop rule moves N every step (trace4a s2 claimed 0*256 + 118)
    try expectNone(&.{ "N=0.", "N = N*256 + b.", "b5=118. Need N*256+118." });
    // `N =` and `Let N =` with another value
    try expectNone(&.{ "N = 280223976814164.", "N = 18364758544493064720.", "Need compute N*65536." });
    try expectNone(&.{ "N = 5.", "Let N = 6.", "Need compute N*1000003." });
    // a note or a hypothesis binds nothing, a later plain value does (R1); a value after a label still blocks
    try expectStated(&.{ "Let A = 12345678901 (≈1.2e10).", "A = 12345678901.", "Need compute A*1000003." }, "(12345678901)*1000003");
    try expectStated(&.{ "If A = 6 then fine.", "Let A = 5.", "Need compute A*1000003." }, "(5)*1000003");
    try expectNone(&.{ "Step3 A = 5.", "A = 5.", "Need compute A*1000003." });
    // kept: the same value again, with a note or a label, and a name after an operator assigns nothing
    try expectStated(&.{ "Let N=280,223,976,814,164.", "N=280,223,976,814,164 (~2.802e14).", "Compute N*65536." }, "(280223976814164)*65536");
    try expectStated(&.{ "Let N=280,223,976,814,164.", "Step3 N=280,223,976,814,164.", "Compute N*65536." }, "(280223976814164)*65536");
    try expectStated(&.{ "Let B = 5.", "2 * B = 10.", "Need compute B*1000003." }, "(5)*1000003");
    // the same line's own computation never reads the old value: `N = N*256 + 118` states nothing
    try expectNone(&.{ "N=0.", "N = N*256 + 118." });
    try expectNone(&.{ "N = 280,223,976,814,164.", "N = N*65536 + 12,816." });
    // as before, only a name's first definition states its computation; a reassignment's never does
    try expectNone(&.{ "Let N = 5.", "N = 280,223,976,814,164 * 3." });
    // kept: a reassigned name's line still states its own literal computation (it binds nothing)
    try expectStated(&.{ "L = b1*b2.", "Compute L = 899,435,521 * 109,096,961." }, "899435521*109096961");
    // notation the parser refuses poisons, never normalizes: `1|2` may be OR, `12|345` may be OR, 1eN is not read
    try expectNone(&.{ "Let D=12.", "D=1|2.", "Need compute D*1000003." });
    try expectNone(&.{ "Let D = 12345.", "D=12|345.", "Need compute D*1000003." });
    try expectNone(&.{ "Let M = 1,000,000,007.", "M = 1e9+7.", "Need compute M*1000003." });
    // kept: the same value in another form, doubted (live R4 product-b)
    try expectStated(&.{ "Let B = 12,345,678,901.", "B = 12,000,000,000 + 345,678,901? Or use pattern.", "Need compute B*1000003." }, "(12345678901)*1000003");
}

test "runner trace: an owned line restated through its names never fires again (live big-number-019)" {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    const f = (try t.line(bn019)).?;
    try t.fire(f.display);
    const again = "Need divide A by B. A has 6 base1e9 limbs, B 3 limbs. Quotient ~ A/B ~ (3.141e50)/(2.718e20)=1.155e30.";
    try std.testing.expectEqual(null, try t.line(again));
}

test "runner trace: a restated owned line fires again only as often as the request's repeat allows" {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.repeat = 1;
    const f = try gpa.dupe(u8, (try t.line(bn019)).?.display);
    defer gpa.free(f);
    try t.fire(f);
    const again = "Need divide A by B. A has 6 base1e9 limbs, B 3 limbs. Quotient ~ A/B ~ (3.141e50)/(2.718e20)=1.155e30.";
    const g = (try t.line(again)).?;
    try std.testing.expectEqualStrings(f, g.display);
    try t.fire(g.display);
    try std.testing.expectEqual(null, try t.line(again));
}

/// A line's statement: Python source (checks, dedup), the frontend's source and the displayed expression.
fn expectRun(lines: []const []const u8, source: []const u8, run: []const u8, display: []const u8) !void {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    var last: ?trace.Found = null;
    for (lines) |l| last = try t.line(l);
    const f = last orelse return error.TestExpectedStatement;
    try std.testing.expectEqualStrings(source, f.source);
    try std.testing.expectEqualStrings(run, f.run);
    try std.testing.expectEqualStrings(display, f.display);
}

test "runner trace: a lone power mod m runs as pow(a, b, m), hex runs in decimal; the shown line keeps the model's text" {
    try expectRun(&.{bn008}, "7**123%1000000007", "print(pow(7, 123, 1000000007))", "7^123 mod 1000000007");
    try expectRun(&.{"Need compute 3^400 mod 10^18."}, "3**400%10**18", "print(pow(3, 400, 1000000000000000000))", "3^400 mod 10^18");
    try expectRun(&.{"Need compute (12+5)^40 mod 1000000007."}, "(12+5)**40%1000000007", "print(pow((12+5), 40, 1000000007))", "(12 + 5)^40 mod 1000000007");
    try expectRun(&.{"Need multiply 7^123 mod 1000000007 by 3."}, "(7**123%1000000007) * (3)", "(pow(7, 123, 1000000007)) * (3)", "(7^123 mod 1000000007) * 3");
    try expectRun(&.{bn015}, "(0xFEDCBA9876543210)*2**64+(0x0123456789ABCDEF)", "(18364758544493064720)*2**64+(81985529216486895)", "0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF");
    // not a lone power: Python's -a**b % m and a*b**c % m are not pow(a, b, m)
    try expectRun(&.{"Need compute -7^123 mod 1000000007."}, "-7**123%1000000007", "-7**123%1000000007", "-7^123 mod 1000000007");
    try expectRun(&.{"Need compute 2*7^123 mod 1000000007."}, "2*7**123%1000000007", "2*7**123%1000000007", "2*7^123 mod 1000000007");
    try expectRun(&.{"Need divide 0x10 by 3."}, "divmod(0x10, 3)", "divmod(16, 3)", "0x10 ÷ 3");
}

test "runner trace: census pow and hex statements run exactly on the real frontend and graph bank (TF_RUNNER_GRAPH_BANK)" {
    const path = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    var bank = try graph.Bank.load(gpa, std.mem.span(path));
    defer bank.deinit();
    const g = try graph.backend(gpa, &bank, 2);
    defer graph.destroy(g);
    var dispatch: route.Dispatch = .{ .gpa = gpa, .circuit = g, .graph = g, .bank = &bank };
    defer dispatch.deinit();
    var routed: route.Routed = .{ .inner = @import("runner_frontend.zig").frontend(), .bank = &bank };
    const front = routed.frontend();
    const cases = [_][3][]const u8{
        .{ bn008, "7^123 mod 1000000007", "937329259\n" },
        .{ "Need compute 3^400 mod 10^18.", "3^400 mod 10^18", "192443283334088001\n" },
        .{ bn015, "0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF", "338770000845734292516042252062085074415\n" },
        .{ "Let B=0x0123456789ABCDEF. Need compute (B+1)*2^64.", "(0x0123456789ABCDEF + 1)*2^64", "1512366075204170947414340898899623936\n" },
    };
    for (cases) |c| {
        var t = trace.Tracer.init(gpa);
        defer t.deinit();
        const f = (try t.line(c[0])).?;
        try std.testing.expectEqualStrings(c[1], f.display);
        var cell = std.heap.ArenaAllocator.init(gpa); // as the runner translates: in the cell's arena
        defer cell.deinit();
        const tr = try front.translate(front.ctx, cell.allocator(), .{ .form = .repl, .lang = "python", .code = f.run });
        std.debug.print("trace run {s}: {s}\n", .{ f.run, if (tr == .program) @tagName(tr.program.kind) else @tagName(tr.abstain) });
        try std.testing.expect(tr == .program and tr.program.kind == .graph);
        const d = dispatch.backend();
        const job = try d.start(1, tr.program, 1 << 32);
        const res = while (true) switch (d.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        try std.testing.expectEqual(rapi.Status.ok, res.status);
        try std.testing.expectEqualStrings(c[2], res.out);
        d.release(1);
    }
}

test "runner trace: negative controls: prose, code and small inline sums state nothing" {
    try expectNone(&.{"We need answer. Need be careful. Let's think about the algorithm."});
    try expectNone(&.{"for i in range(10): total = total * 3"});
    try expectStated(&.{"Need compute 2+3."}, "2+3"); // a small sum after a goal head is still a statement
}

test "runner trace: hand-trace cues" {
    for ([_][]const u8{ "Step i=0: rem=314.", "We can perform long division or use modular arithmetic.", "314,159 * 271,828.", "Let's do base 1000.", "Then carry 3 into the next group", "Row 0: [0,1,2]", "So 41*42=1722." }) |l|
        try std.testing.expect(trace.cue(l));
    for ([_][]const u8{ "So that is all.", "Need quotient and remainder.", "We need compute. Let's denote B." }) |l|
        try std.testing.expect(!trace.cue(l));
}

fn feedAll(s: *scan.Scanner, tokens: []const []const u8, out: *std.ArrayList(scan.Events)) !void {
    for (tokens) |t| try out.append(gpa, try s.feed(gpa, t, true));
}

test "runner trace: the scanner closes a trace cell at its line and triggers at the cue line's end" {
    var s: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer s.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&s, &.{ bn019, "\n", "\n", "We need compute.", " Nothing yet.\n", "We can perform long division or use modular arithmetic.", "\n", "Step i=0: rem=314.\n" }, &ev);
    const c = ev.items[1].closed.?;
    try std.testing.expectEqual(scan.Stated{ .display = c.trace.?.display, .divmod = true }, c.trace.?);
    try std.testing.expectEqualStrings("divmod(314159265358979323846264338327950288419716939937510, 271828182845904523536)", c.code);
    try std.testing.expect(c.form == .repl);
    for (ev.items[0..6]) |e| try std.testing.expectEqual(@as(?scan.Trigger, null), e.trigger);
    try std.testing.expectEqual(scan.Trigger{ .index = 0, .trace = true }, ev.items[6].trigger.?);
    try std.testing.expectEqual(@as(?scan.Trigger, null), ev.items[7].trigger); // owned once
}

test "runner trace: off by default, out of window, spoiled tokens wait for the next cue" {
    var off: scan.Scanner = .{ .cfg = .{} };
    defer off.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&off, &.{ bn019 ++ "\n", "We can perform long division.\n" }, &ev);
    for (ev.items) |e| try std.testing.expect(e.closed == null and e.trigger == null);
    ev.clearRetainingCapacity();
    var far: scan.Scanner = .{ .cfg = .{ .trace = true, .trace_window = 2 } };
    defer far.deinit(gpa);
    try feedAll(&far, &.{ bn019 ++ "\n", "a", "b", "c", "We can perform long division.\n" }, &ev);
    try std.testing.expect(ev.items[0].closed != null and ev.items[4].trigger == null);
    ev.clearRetainingCapacity();
    var spoil: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer spoil.deinit(gpa);
    try feedAll(&spoil, &.{ bn019 ++ "\n", "We can perform long division.\nStep", " i=0: rem=314.\n" }, &ev);
    try std.testing.expect(ev.items[1].trigger == null);
    try std.testing.expectEqual(scan.Trigger{ .index = 0, .trace = true }, ev.items[2].trigger.?);
}

test "runner trace: a fence's code lines state nothing, and a trace cell is no notebook prior" {
    var s: scan.Scanner = .{ .cfg = .{ .trace = true } };
    defer s.deinit(gpa);
    var ev: std.ArrayList(scan.Events) = .empty;
    defer ev.deinit(gpa);
    try feedAll(&s, &.{ "```python\n", "Need compute 2^100 - 3^50.\n", "Step 1\n", "```\n" }, &ev);
    for (ev.items[0..3]) |e| try std.testing.expect(e.closed == null and e.trigger == null);
}

/// The circuit's printed values for the trace programs (the fake frontend passes the Python source through).
const TraceBack = struct {
    delay: u32 = 0,
    fault: bool = false,
    jobs: std.ArrayList(struct { prog: []u8, polls: u32 }) = .empty,

    const answers = [_][2][]const u8{
        .{ "(987654) * (137)", "135308598\n" },
        .{ "pow(7, 123, 1000000007)", "937329259\n" }, // the fake frontend drops print( )
    };

    fn backend(b: *TraceBack) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const b: *TraceBack = @ptrCast(@alignCast(ctx));
        try b.jobs.append(gpa, .{ .prog = try gpa.dupe(u8, p.bytes), .polls = 0 });
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *TraceBack = @ptrCast(@alignCast(ctx));
        const j = &b.jobs.items[job];
        j.polls += 1;
        if (j.polls <= b.delay) return .running;
        if (b.fault) return .{ .done = .{ .status = .fault, .out = "" } };
        for (answers) |x| if (std.mem.eql(u8, x[0], j.prog)) return .{ .done = .{ .status = .ok, .out = x[1], .steps = 100 } };
        return .{ .done = .{ .status = .fault, .out = "" } };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *TraceBack) void {
        for (b.jobs.items) |j| gpa.free(j.prog);
        b.jobs.deinit(gpa);
    }
};

/// One reply; `guess` is the model's own line after `lead` (a digit line is itself a cue).
fn traced(lead: []const u8, guess: []const u8, on: bool, delay: u32, fault: bool) ![]u8 {
    return tracedAsk(lead, guess, on, delay, fault, .{});
}

fn tracedAsk(lead: []const u8, guess: []const u8, on: bool, delay: u32, fault: bool, ask: runner.TraceAsk) ![]u8 {
    var back: TraceBack = .{ .delay = delay, .fault = fault };
    defer back.deinit();
    var unused: rt.Back = .{};
    defer unused.deinit();
    var m = [_]rt.Model{.{ .lead = lead, .guess = guess }};
    const texts = try rt.replies(&m, .{ .trace = on, .backend = back.backend(), .trace_ask = ask }, &unused);
    defer gpa.free(texts);
    return texts[0];
}

const mul_lead = "Need multiply A=987654 B=137. Need exact product.\nLet's do long multiplication.\n";

test "runner trace: the reply owns one [exact: …] line after the cue, then the model goes on" {
    const got = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_lead ++ "[exact, computed by the engine: 987654 * 137 = 135308598]\nDone.", got);
    const pow = "We need answer compute 7^123 mod 1000000007. Let's compute carefully.\nWe can do it by hand.\n";
    const same = try traced(pow, "ok", true, 0, false);
    defer gpa.free(same);
    try std.testing.expectEqualStrings(pow ++ "[exact, computed by the engine: 7^123 mod 1000000007 = 937329259]\nDone.", same);
}

test "runner trace: late runs give the same reply; faults, the form off and no cue keep the model's text" {
    const early = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(early);
    const late = try traced(mul_lead, "ok", true, 6, false);
    defer gpa.free(late);
    try std.testing.expectEqualStrings(early, late);
    const fault = try traced(mul_lead, "ok", true, 0, true);
    defer gpa.free(fault);
    try std.testing.expectEqualStrings(mul_lead ++ "ok\nDone.", fault);
    const off = try traced(mul_lead, "ok", false, 0, false);
    defer gpa.free(off);
    try std.testing.expectEqualStrings(mul_lead ++ "ok\nDone.", off);
    const quiet = "Need multiply A=987654 B=137. Need exact product.\nThat is the task.\n";
    const nocue = try traced(quiet, "ok", true, 0, false);
    defer gpa.free(nocue);
    try std.testing.expectEqualStrings(quiet ++ "ok\nDone.", nocue);
    const digits = try traced(quiet, "135308598", true, 0, false); // the model's own digit line is a hand-work cue
    defer gpa.free(digits);
    try std.testing.expectEqualStrings(quiet ++ "135308598\n[exact, computed by the engine: 987654 * 137 = 135308598]\n", digits);
}

test "runner trace: a request's repeat reaches the reply's tracer" {
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test", .scan = .{ .trace = true } }, .frontend = undefined, .backend = undefined, .text = .{ .bytes = undefined, .encode = undefined } };
    const s = (try r.session(.{ .repeat = 2 })).?;
    defer gpa.destroy(s);
    defer gpa.free(s.format);
    defer s.scan.deinit(gpa);
    try std.testing.expectEqual(@as(u8, 2), s.scan.cfg.trace_repeat);
}

test "runner trace: a request's wording frames the owned line; runner_trace false keeps the model's text" {
    const worded = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .format = "[exact, computed by the engine: {}]" });
    defer gpa.free(worded);
    try std.testing.expectEqualStrings(mul_lead ++ "[exact, computed by the engine: 987654 * 137 = 135308598]\nDone.", worded);
    const noted = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .format = "[exact: {}] (computed exactly by the engine)" });
    defer gpa.free(noted);
    try std.testing.expectEqualStrings(mul_lead ++ "[exact: 987654 * 137 = 135308598] (computed exactly by the engine)\nDone.", noted);
    const old = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .format = "[exact: {}]" }); // the pre-round-3 default
    defer gpa.free(old);
    try std.testing.expectEqualStrings(mul_lead ++ "[exact: 987654 * 137 = 135308598]\nDone.", old);
    const default = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .format = trace.default_format });
    defer gpa.free(default);
    const plain = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(plain);
    try std.testing.expectEqualStrings(plain, default);
    const off = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .off = true, .format = "[engine: {}]" });
    defer gpa.free(off);
    try std.testing.expectEqualStrings(mul_lead ++ "ok\nDone.", off);
}

test "runner trace: a request's check follows the owned line; without it the reply is unchanged" {
    const checked = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .check = true });
    defer gpa.free(checked);
    // the lane host's vocab is printable ASCII, so `≡` has no token here (runner_trace_check_test.zig pins the bytes)
    try std.testing.expectEqualStrings(mul_lead ++ "[exact, computed by the engine: 987654 * 137 = 135308598] check: 3 * 2  6 (mod 9), 654 * 137  598 (mod 1000)\nDone.", checked);
    const off = try tracedAsk(mul_lead, "ok", true, 0, false, .{ .check = false });
    defer gpa.free(off);
    const plain = try traced(mul_lead, "ok", true, 0, false);
    defer gpa.free(plain);
    try std.testing.expectEqualStrings(plain, off);
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test", .scan = .{ .trace = true } }, .frontend = undefined, .backend = undefined, .text = .{ .bytes = undefined, .encode = undefined } };
    const s = (try r.session(.{ .check = true })).?;
    defer gpa.destroy(s);
    defer gpa.free(s.format);
    defer s.scan.deinit(gpa);
    try std.testing.expect(s.check);
}

test "runner trace: wording around divmod's quotient and remainder; a request's wording must be one line with {} once" {
    const line = (try trace.owned(gpa, "[exact, computed by the engine: {}]", "(10^30 - 1) ÷ 97", true, "(10309278350515463917525773195, 84)\n", false)).?;
    defer gpa.free(line);
    try std.testing.expectEqualStrings("[exact, computed by the engine: (10^30 - 1) ÷ 97 = 10309278350515463917525773195 remainder 84]\n", line);
    const lead = (try trace.owned(gpa, "{} (exact)", "2^10", false, "1024\n", false)).?;
    defer gpa.free(lead);
    try std.testing.expectEqualStrings("2^10 = 1024 (exact)\n", lead);
    try std.testing.expect(trace.validFormat("[exact: {}]"));
    try std.testing.expect(trace.validFormat("{}"));
    var long: [trace.max_format + 1]u8 = undefined;
    @memset(&long, 'x');
    long[0..2].* = "{}".*;
    try std.testing.expect(trace.validFormat(long[0..trace.max_format]));
    try std.testing.expect(!trace.validFormat(&long));
    for ([_][]const u8{ "", "[exact]", "[{} = {}]", "[exact: {}]\nmore", "[exact: {}]\r" }) |bad| try std.testing.expect(!trace.validFormat(bad));
}

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

fn readAll(path: []const u8) ![]u8 {
    var threaded: std.Io.Threaded = .init_single_threaded;
    return std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, gpa, .limited(1 << 30));
}

test "runner trace: census replay equals the offline study (TF_TRACE_CENSUS, TF_TRACE_EXPECT)" {
    const path = env("TF_TRACE_CENSUS") orelse {
        std.debug.print("skip trace census replay: TF_TRACE_CENSUS unset\n", .{});
        return;
    };
    const data = try readAll(path);
    defer gpa.free(data);
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(gpa);
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |row| {
        if (row.len == 0) continue;
        var parsed = try std.json.parseFromSlice(struct { id: []const u8, text: []const u8 }, gpa, row, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        var s: scan.Scanner = .{ .cfg = .{ .trace = true, .fence = false, .repl = false } };
        defer s.deinit(gpa);
        var codes: std.ArrayList([]u8) = .empty;
        defer {
            for (codes.items) |c| gpa.free(c);
            codes.deinit(gpa);
        }
        var parts = std.mem.splitScalar(u8, parsed.value.text, '\n');
        while (parts.next()) |part| {
            const ev = try s.feed(gpa, part, true);
            if (ev.closed) |c| try codes.append(gpa, try gpa.dupe(u8, c.code));
            const nl = try s.feed(gpa, "\n", true);
            if (nl.closed) |c| try codes.append(gpa, try gpa.dupe(u8, c.code));
            if (nl.trigger) |t| if (t.trace) try got.print(gpa, "{s}\t{s}\n", .{ parsed.value.id, codes.items[t.index] });
        }
    }
    std.debug.print("trace census replay: {d} owned lines\n{s}", .{ std.mem.count(u8, got.items, "\n"), got.items });
    const expect = env("TF_TRACE_EXPECT") orelse return;
    const want = try readAll(expect);
    defer gpa.free(want);
    try std.testing.expectEqualStrings(want, got.items);
}

test "runner trace: the reply's runtime block counts a trace cell under trace, not repl" {
    var back: TraceBack = .{};
    defer back.deinit();
    var unused: rt.Back = .{};
    defer unused.deinit();
    var m = [_]rt.Model{.{ .lead = mul_lead, .guess = "ok" }};
    var sums: [1][]u8 = undefined;
    rt.freeAll(try rt.replies(&m, .{ .trace = true, .backend = back.backend(), .summaries = &sums }, &unused));
    defer gpa.free(sums[0]);
    try std.testing.expect(std.mem.indexOf(u8, sums[0], "\"repl\":{\"closed\":0,\"owned\":0,") != null);
    try std.testing.expect(std.mem.indexOf(u8, sums[0], "\"trace\":{\"closed\":1,\"owned\":1,\"abstained\":0,") != null);
}

test {
    _ = @import("runner_trace_value.zig");
}
