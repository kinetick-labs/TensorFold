"""CPU numpy schedule emulator for our EXL3 routed-MoE kernel (zig/kernels/cuda/fn_exl3.cu grouped_kernel), no GPU."""

from __future__ import annotations

from collections import Counter

import numpy as np

WARPS = 4
TILE = 16


def _rows(members: np.ndarray, u: int, base: int, count: int, maxm: int, slots: int) -> np.ndarray:
    """Expert u's member codes [base, base+count) as flattened A rows ((code >> 5) * slots + (code & 31)); -1 dead."""

    m = np.arange(base, base + count)
    ok = m < maxm
    code = np.where(ok, members[u, np.where(ok, m, 0)], -1)
    return np.where(code >= 0, (code >> 5) * slots + (code & 31), -1).astype(np.int64)


def _chain(split: int, warp: int, KT: int, SK: int, nt0: int, i: int, h: int) -> tuple:
    """The (k tile, column tile, n8 half) mma one warp issues into a Z element, K tiles ascending."""

    per_split = KT // SK
    per_warp = per_split // WARPS
    kt0 = split * per_split + warp * per_warp
    return tuple((kt0 + it, nt0 + i, h) for it in range(per_warp))


def _warps(cache: dict, split: int, nt0: int, i: int, h: int, KT: int, SK: int) -> tuple:
    """The four warps' mma chains for one (column tile, half), cached: it is the same for every Z element."""

    key = (split, nt0, i, h)
    got = cache.get(key)
    if got is None:
        got = tuple((w, _chain(split, w, KT, SK, nt0, i, h)) for w in range(WARPS))
        cache[key] = got
    return got


def _store(z: dict, mat: int, split: int, P: int, N: int, r: int, column: int, chains: tuple, SK: int) -> None:
    """One Z element as address -> (A row, warps' mma chains); asserts a member row is never stored twice."""

    addr = ((mat * SK + split) * P + r) * N + column
    assert addr not in z, ("stored twice", addr)
    z[addr] = (int(r), chains)


def trace_grouped(members: np.ndarray, ucount: int, K: int, N: int, P: int, SK: int, slots: int, mats: int,
                  NT: int) -> dict:
    """grouped_kernel<CB, NT, W, PF>: 16-row member tiles, program (u, n block, (mat*SK+split)*MT+mtile)."""

    maxm = members.shape[1]
    MT = (maxm + TILE - 1) // TILE
    KT, z, cache = K // TILE, {}, {}
    for u in range(members.shape[0]):
        if u >= ucount:
            continue
        for by in range(N // (TILE * NT)):
            nt0 = by * NT
            for bz in range(mats * SK * MT):
                mtile, split, mat = bz % MT, (bz // MT) % SK, bz // MT // SK
                rows = _rows(members, u, mtile * TILE, TILE, maxm, slots)
                if rows[0] < 0:
                    continue
                for idx in range(TILE * NT * TILE):
                    row, col = idx // (NT * TILE), idx % (NT * TILE)
                    r = rows[row]
                    if r < 0:
                        continue
                    i, h = col // TILE, (col % TILE) // 8
                    _store(z, mat, split, P, N, r, nt0 * TILE + col, _warps(cache, split, nt0, i, h, KT, SK), SK)
    return z


def trace_member_pass(members: np.ndarray, ucount: int, K: int, N: int, P: int, SK: int, slots: int, mats: int,
                      NT: int, mtl: int) -> dict:
    """The wide variant we mean to transplant: mtl 16-row tiles a program, same per-tile mma, live tiles only."""

    maxm = members.shape[1]
    MG = (maxm + TILE * mtl - 1) // (TILE * mtl)
    KT, z, cache = K // TILE, {}, {}
    for u in range(members.shape[0]):
        if u >= ucount:
            continue
        for by in range(N // (TILE * NT)):
            nt0 = by * NT
            for bz in range(mats * SK * MG):
                mg, split, mat = bz % MG, (bz // MG) % SK, bz // MG // SK
                rows = _rows(members, u, mg * mtl * TILE, mtl * TILE, maxm, slots)
                if rows[0] < 0:
                    continue
                live = 0
                for t in range(mtl):
                    if rows[t * TILE] >= 0:
                        live = t + 1
                for t in range(live):
                    for idx in range(TILE * NT * TILE):
                        row, col = idx // (NT * TILE), idx % (NT * TILE)
                        r = rows[t * TILE + row]
                        if r < 0:
                            continue
                        i, h = col // TILE, (col % TILE) // 8
                        _store(z, mat, split, P, N, r, nt0 * TILE + col,
                               _warps(cache, split, nt0, i, h, KT, SK), SK)
    return z


def trace_member_pass_hoisted(members: np.ndarray, ucount: int, K: int, N: int, P: int, SK: int, slots: int,
                              mats: int, NT: int, mtl: int) -> tuple:
    """The decode-hoisted wide arm: n tile outer, mma chain inner, the live member tiles' passes inside it."""

    maxm = members.shape[1]
    MG = (maxm + TILE * mtl - 1) // (TILE * mtl)
    KT, z, cache = K // TILE, {}, {}
    per_split = KT // SK
    decodes: dict = {}
    for u in range(members.shape[0]):
        if u >= ucount:
            continue
        for by in range(N // (TILE * NT)):
            nt0 = by * NT
            for bz in range(mats * SK * MG):
                mg, split, mat = bz % MG, (bz // MG) % SK, bz // MG // SK
                rows = _rows(members, u, mg * mtl * TILE, mtl * TILE, maxm, slots)
                if rows[0] < 0:
                    continue
                live = 0
                for t in range(mtl):
                    if rows[t * TILE] >= 0:
                        live = t + 1
                sites: Counter = Counter()
                for i in range(NT):                              # the n tile loop the kernel runs outermost
                    for it in range(per_split):                  # the warp's own k tiles, decoded once each
                        sites[(nt0 + i, split * per_split + it)] += 1
                    for t in range(live):
                        for idx in range(TILE * TILE):           # one n tile's 16 columns for one member tile
                            row, col = idx // TILE, idx % TILE
                            r = rows[t * TILE + row]
                            if r < 0:
                                continue
                            h = col // 8
                            _store(z, mat, split, P, N, r, nt0 * TILE + i * TILE + col,
                                   _warps(cache, split, nt0, i, h, KT, SK), SK)
                decodes[(u, by, bz)] = sites
    return z, decodes   # (Z, {program: Counter of the (n tile, k tile) sites decoded}), one count a site


def decode_sites_member_pass(members: np.ndarray, ucount: int, K: int, N: int, P: int, SK: int, slots: int,
                             mats: int, NT: int, mtl: int) -> dict:
    """The per-tile arm's decode sites: every live member tile re-decodes its program's own (n tile, k tile) sites."""

    maxm = members.shape[1]
    MG = (maxm + TILE * mtl - 1) // (TILE * mtl)
    KT, per_split = K // TILE, K // TILE // SK
    decodes: dict = {}
    for u in range(members.shape[0]):
        if u >= ucount:
            continue
        for by in range(N // (TILE * NT)):
            nt0 = by * NT
            for bz in range(mats * SK * MG):
                mg, split = bz % MG, (bz // MG) % SK
                rows = _rows(members, u, mg * mtl * TILE, mtl * TILE, maxm, slots)
                if rows[0] < 0:
                    continue
                live = 0
                for t in range(mtl):
                    if rows[t * TILE] >= 0:
                        live = t + 1
                sites: Counter = Counter()
                _ = P
                for _t in range(live):
                    for i in range(NT):
                        for it in range(per_split):
                            sites[(nt0 + i, split * per_split + it)] += 1
                decodes[(u, by, bz)] = sites
    return decodes


def stage_stride(per_split: int) -> int:
    """The staged row stride in 32-bit words: the split's k slice (16 k a word) padded to 4 mod 32."""

    words = per_split * 8
    return words + ((4 + 32 - (words & 31)) & 31)


def staged_words(per_split: int) -> int:
    """The valid words one member row's staged slice holds; the pad past it is address space, never read."""

    return per_split * 8


def staged_lane_words(kt: int, t: int) -> tuple:
    """The two staged words a lane (g, t) reads for member rows g and g+8 of one k tile: the second is eight k on."""

    kw = kt * 8 + t
    return (kw, kw + 4)


def reread_lane_words(kt: int, t: int) -> tuple:
    """The same slice words the re-reading arm's load_pair takes from global: half k + 2t and k + 2t + 8."""

    return ((kt * 16 + 2 * t) // 2, (kt * 16 + 2 * t + 8) // 2)


def trace_member_pass_staged(members: np.ndarray, ucount: int, K: int, N: int, P: int, SK: int, slots: int, mats: int,
                             NT: int, mtl: int) -> tuple:
    """The staged wide arm: rows staged once a block, the same per-tile mma and store; returns (Z, A global bytes)."""

    maxm = members.shape[1]
    MG = (maxm + TILE * mtl - 1) // (TILE * mtl)
    KT, z, cache = K // TILE, {}, {}
    per_split = KT // SK
    gbytes = 0
    for u in range(members.shape[0]):
        if u >= ucount:
            continue
        for by in range(N // (TILE * NT)):
            nt0 = by * NT
            for bz in range(mats * SK * MG):
                mg, split, mat = bz % MG, (bz // MG) % SK, bz // MG // SK
                rows = _rows(members, u, mg * mtl * TILE, mtl * TILE, maxm, slots)
                if rows[0] < 0:
                    continue
                live = 0
                for t in range(mtl):
                    if rows[t * TILE] >= 0:
                        live = t + 1
                gbytes += 16 * live * staged_words(per_split) * 4      # one global read a live row, 4 bytes a word
                for t in range(live):
                    for idx in range(TILE * NT * TILE):
                        row, col = idx // (NT * TILE), idx % (NT * TILE)
                        r = rows[t * TILE + row]
                        if r < 0:
                            continue
                        i, h = col // TILE, (col % TILE) // 8
                        _store(z, mat, split, P, N, r, nt0 * TILE + col,
                               _warps(cache, split, nt0, i, h, KT, SK), SK)
    return z, gbytes


def a_bytes_reread(per_split: int, mtl: int, NT: int) -> int:
    """The A bytes the re-reading arm's loop structure loads a program: a member row's slice once a n tile."""

    return mtl * 16 * staged_words(per_split) * 4 * NT
