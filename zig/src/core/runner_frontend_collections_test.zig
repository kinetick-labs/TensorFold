const std = @import("std");
const collections = @import("runner_frontend_collections.zig");
const frontend = @import("runner_frontend.zig");
const reference = @import("runner_frontend_reference_test.zig");
const t = std.testing;
pub const cases = [_][2][]const u8{
    .{ "print(\"abc\"[::2147483647]);print(sorted({True,1,2}));print(sorted({1,True,2}))", "a\n[True, 2]\n[1, 2]\n" },
    .{ "xs=[];it=enumerate(xs);sorted([it],key=sum);xs.append(1);print(list(it))", "[]\n" },
    .{ "x=1; print(x is None, None is x, x is not None, None is not x)", "False False True True\n" },
    .{ "a=[1,-2,-3,4]; print([x for x in a if x>=0])", "[1, 4]\n" },
    .{ "row=[0]*3; a=[row]*3; a[0][0]=7; print(a); print(a[1] is row)", "[[7, 0, 0], [7, 0, 0], [7, 0, 0]]\nTrue\n" },
    .{ "a=[[0]*2 for i in range(2)]; a[0][0]=7; print(a)", "[[7, 0], [0, 0]]\n" },
    .{ "d={True:'a',2:'b'}; d[1]='c'; print(d); print(d.get(7)); print(list(d.items()))", "{True: 'c', 2: 'b'}\nNone\n[(True, 'c'), (2, 'b')]\n" },
    .{ "s=set([3,1,3,2]); s.discard(2); s.add(4); print(sorted(s)); print(3 in s, 2 not in s)", "[1, 3, 4]\nTrue True\n" },
    .{ "a=[1]; b=a; c=a[:]; b.append(2); a+=[3]; print(a,b,c)", "[1, 2, 3] [1, 2, 3] [1]\n" },
    .{ "def f(a):\n    a.append(2)\n    return a\na=[1]; print(a,f(a))", "[1, 2] [1, 2]\n" },
    .{ "d={'a':1}; view=d.values(); d['a']=2; print(list(view)); e=enumerate([4,5],2); print(list(e)); print(list(e))", "[2]\n[(2, 4), (3, 5)]\n[]\n" },
    .{ "print(''.join(map(str,sorted([1,2,3],reverse=True)))); print(sorted([[1,2],[0],[3]],key=sum))", "321\n[[0], [1, 2], [3]]\n" },
    .{ "s='112112'; c=0; subs=[]\nfor i in range(len(s)):\n    sm=0\n    for j in range(i,len(s)):\n        sm+=int(s[j])\n        if sm==j-i+1:\n            c+=1; subs.append(s[i:j+1])\nprint(c,subs)", "6 ['1', '11', '1', '1', '11', '1']\n" },
    .{ "a=[]\ndef f(x):\n    a.append(x)\n    return x\nprint(any(f(x) for x in [1,2,3]),a); a=[]; print(any([f(x) for x in [1,2,3]]),a)", "True [1]\nTrue [1, 2, 3]\n" },
    .{ "print(['a\\\'b','a\"b','\\\\','\\n','\\x00']); print(('x',)); print('abc'[::-1])", "[\"a'b\", 'a\"b', '\\\\', '\\n', '\\x00']\n('x',)\ncba\n" },
    .{ "g=[[1,2],[2],[]]; q=[0]; seen={0}\nfor node in q:\n    for nxt in g[node]:\n        if nxt not in seen:\n            seen.add(nxt); q.append(nxt)\nprint(q)", "[0, 1, 2]\n" },
    .{ "g={0:[1,2],1:[2],2:[]}; stack=[0]; seen=set(); order=[]\nwhile stack:\n    node=stack.pop()\n    if node not in seen:\n        seen.add(node); order.append(node)\n        for nxt in g[node]:\n            stack.append(nxt)\nprint(order)", "[0, 2, 1]\n" },
    .{ "a='ab'; b='ac'; d=[[0]*3 for i in range(3)]\nfor i in range(3):\n    d[i][0]=i; d[0][i]=i\nfor i in range(1,3):\n    for j in range(1,3):\n        d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))\nprint(d)", "[[0, 1, 2], [1, 0, 1], [2, 1, 1]]\n" },
};
pub const budget_case = "def edit_distance(a, b):\n    if len(a) > len(b):\n        a, b = b, a          # a is the shorter string\n    m = len(a)\n    dp = list(range(m + 1))  # dp[j] = distance to a[:j]\n\n    for i, ch in enumerate(b, 1):\n        prev = dp[0]\n        dp[0] = i\n        for j in range(1, m + 1):\n            old = dp[j]\n            cost = 0 if a[j - 1] == ch else 1\n            dp[j] = min(\n                old + 1,          # delete/insert\n                dp[j - 1] + 1,    # insert/delete\n                prev + cost       # match/substitute\n            )\n            prev = old\n\n    return dp[m]\n\nprint(edit_distance(\"kitten\",\"sitting\"))";

pub const over_budget_case = "a=[0]\ni=0\nwhile i<1000:\n    a[0]+=1\n    i+=1\nprint(a)";

pub const negatives = [_][]const u8{
    "__debug__=1;print([__debug__])",
    "__builtins__={}\ndef f():\n    print([1])\nf()",
    "print({1:2}.keys()=={1})",
    "print({}.items()==set())",
    "print([{1:2}.keys()]==[{1}])",
    "print(sorted([1],key=sum))",
    "a=[1];print(a);a",
    "d={1:10};e=enumerate(d);d[2]=20;print(list(e))",
    "d={1:10};e=zip(d.values(),[1]);d[2]=20;print(list(e))",
    "print(sorted([2,1],reverse=[]))",
    "a=[];a+=range(2);print(a)",
    "a=[0];b=[0]\nfor _ in range(7):\n    a=[a]*8;b=[b]*8\nprint(a==b)",
    "print({1,2})",
    "print(list(set([1,2])))",
    "for x in set([1,2]):\n    print(x)",
    "a=[]; a.append(a); print(a)",
    "d={}; d[1]=d; print(d)",
    "a=[1]; print(a[2])",
    "a=[1]; a[-2]=3; print(a)",
    "print(2147483647+1)",
    "print((-2147483647-1)//-1)",
    "print([1]*129)",
    "while True:\n    pass\nprint(0)",
    "print(1.5)",
    "import os\nprint(1)",
    "print(input())",
    "def f():\n    return f()\nprint(f())",
    "print(f())\ndef f():\n    return 1",
    "x=1\ndef f():\n    print(x)\n    x=2\nf()",
    "d={1:2}\nfor k in d:\n    d[3]=4\nprint(d)",
    "print=1; print([1])",
    "print('é')",
    "print([1]) #\x00",
    "def f(x):\n    return x\n",
};
test "collections proof keeps opaque original inputs and source control flow" {
    for (cases) |fixture| {
        const lower = (try collections.lowerWithCost(t.allocator, fixture[0])) orelse {
            std.debug.print("collections refused positive: {s}\n", .{fixture[0]});
            return error.UnexpectedAbstention;
        };
        defer t.allocator.free(lower.source);
        try t.expect(lower.estimated_steps > 0);
        try t.expect(std.mem.indexOf(u8, lower.source, "static volatile int inputs[]") != null);
        try t.expect(std.mem.indexOf(u8, lower.source, "py_hold") != null);
    }
}
test "collections proof refuses unsupported and invalid behavior" {
    for (negatives) |source| {
        const result = try collections.lowerWithCost(t.allocator, source);
        defer if (result) |body| t.allocator.free(body.source);
        if (result != null) std.debug.print("collections accepted negative: {s}\n", .{source});
        try t.expect(result == null);
    }
}
test "configured collections O1 ELF matches CPython and reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD"), .optimize = .O1, .step_budget = 0 } };
    const front = context.frontend();
    for (cases) |fixture| {
        const oracle = try std.process.run(t.allocator, t.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-c", fixture[0] }, .stdout_limit = .limited(16384), .stderr_limit = .limited(16384), .timeout = .{ .duration = .{ .raw = .fromSeconds(3), .clock = .awake } } });
        defer t.allocator.free(oracle.stdout);
        defer t.allocator.free(oracle.stderr);
        try t.expect(oracle.term.success());
        try t.expectEqualStrings(fixture[1], oracle.stdout);
        const translated = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = fixture[0] });
        defer if (translated == .program) t.allocator.free(translated.program.bytes);
        if (translated != .program) std.debug.print("collections compiler refused: {s}\n", .{fixture[0]});
        try t.expect(translated == .program);
        try reference.check(root, "/usr/bin/python3", translated.program, oracle.stdout);
    }
    // Larger admission is opt-in; a cached high-budget ELF must not bypass either lower bound.
    const cell: @import("runner_api.zig").Cell = .{ .form = .fence, .lang = "python", .code = budget_case };
    context.compiler.step_budget = 10522;
    try expectBudgetRefusal(front, cell);
    context.compiler.step_budget = 32768;
    const admitted = try front.translate(front.ctx, t.allocator, cell);
    defer if (admitted == .program) t.allocator.free(admitted.program.bytes);
    try t.expect(admitted == .program);
    try reference.check(root, "/usr/bin/python3", admitted.program, "3\n");
    context.compiler.step_budget = 28359;
    try expectBudgetRefusal(front, cell);
    context.compiler.step_budget = 10522;
    try expectBudgetRefusal(front, cell);
    const over: @import("runner_api.zig").Cell = .{ .form = .fence, .lang = "python", .code = over_budget_case };
    const over_cost = (try collections.lowerWithCost(t.allocator, over_budget_case)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(over_cost.source);
    try t.expect(over_cost.estimated_steps > 32768);
    context.compiler.step_budget = 32768;
    try expectBudgetRefusal(front, over);
}

test "collections nested dictionary comparisons retain quadratic cost" {
    const code = "d={}\nfor i in range(10):\n    d[i]=i\na=[d]*20;b=[d]*20;print(a==b)";
    const lowered = (try collections.lowerWithCost(t.allocator, code)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(lowered.source);
    try t.expect(lowered.estimated_steps > 10522);
}

// Apply the scalar lowering's executed O0 floor and ELF rodata guards to collections.
test "collections O1 retains runtime work against O0 folding guards" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    for ([_]usize{ 6, 14, 15 }) |index| {
        const fixture = cases[index];
        const lowered = (try collections.lowerWithCost(t.allocator, fixture[0])) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(lowered.source);
        try @import("runner_frontend_optimization_test.zig").checkCollectionsOptimization(&env, lowered.source, fixture[1]);
    }
}

fn expectBudgetRefusal(front: @import("runner_api.zig").Frontend, cell: @import("runner_api.zig").Cell) !void {
    const result = try front.translate(front.ctx, t.allocator, cell);
    defer if (result == .program) t.allocator.free(result.program.bytes);
    try t.expect(result == .abstain);
}
