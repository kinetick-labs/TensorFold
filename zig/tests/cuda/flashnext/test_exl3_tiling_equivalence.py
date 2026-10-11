"""Bit-equality proof: our 16-row member tile and a wider member pass store the same Z, per trace dict, no GPU."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "zig"))
import x3_exl3_sched as EMU

SHAPES = [(2, 512, 128, 4), (1, 256, 256, 1), (1, 512, 128, 2)]
TILES = [(4, 4), (8, 2), (4, 8), (8, 4), (8, 8)]
COUNTS = [0, 1, 15, 16, 17, 64, 65, 127, 128, 129, 200, 33]
R, SLOTS = 200, 7


def _members(nexp: int, counts: list[int], R: int, slots: int, seed: int) -> np.ndarray:
    """Distinct-row member codes (row << 5 | slot), rows ascending, -1 padded to R: a row belongs to one expert."""

    g = random.Random(seed)
    free = {r: list(range(slots)) for r in range(R)}
    out = np.full((nexp, R), -1, dtype=np.int64)
    for i, c in enumerate(counts[:nexp]):
        for j, r in enumerate(sorted(g.sample([r for r in range(R) if free[r]], c))):
            out[i, j] = (r << 5) | free[r].pop(g.randrange(len(free[r])))
    return out


@pytest.mark.parametrize("nt,mtl", TILES)
@pytest.mark.parametrize("mats,K,N,SK", SHAPES)
def test_member_pass_equals_grouped(nt: int, mtl: int, mats: int, K: int, N: int, SK: int) -> None:
    """The wider member pass stores upstream's bits, for both a full expert list and dead experts past ucount."""

    members = _members(len(COUNTS), COUNTS, R, SLOTS, nt * 10 + mtl)
    P = R * SLOTS
    for ucount in (len(COUNTS), len(COUNTS) - 3):
        up = EMU.trace_grouped(members, ucount, K, N, P, SK, SLOTS, mats, nt)
        ours = EMU.trace_member_pass(members, ucount, K, N, P, SK, SLOTS, mats, nt, mtl)
        assert ours == up
        assert len(up) == sum(min(c, R) for c in COUNTS[:ucount]) * N * mats * SK


def test_width_one_is_the_16_row_tile() -> None:
    """A member pass of width 1 is the current kernel: the free width reduces to the 16-row tile exactly."""

    members = _members(2, [17, 200], R, SLOTS, 1)
    P = R * SLOTS
    grouped = EMU.trace_grouped(members, 2, 256, 128, P, 2, SLOTS, 1, 4)
    assert EMU.trace_member_pass(members, 2, 256, 128, P, 2, SLOTS, 1, 4, 1) == grouped


def test_dead_tiles_store_nothing() -> None:
    """A one-member expert: the 15 dead rows of its tile and every later tile store no Z element."""

    members = np.full((1, R), -1, dtype=np.int64)
    members[0, 0] = (3 << 5) | 1
    z = EMU.trace_grouped(members, 1, 256, 128, R * SLOTS, 1, SLOTS, 1, 4)
    assert len(z) == 128 and {r for r, _ in z.values()} == {3 * SLOTS + 1}


def test_non_multiple_member_counts_keep_all_rows() -> None:
    """Counts that leave a tile half full (1, 17, 129) still store every live row exactly once."""

    counts = [1, 17, 129]
    members = _members(len(counts), counts, R, SLOTS, 5)
    P = R * SLOTS
    for mtl in (2, 8):
        z = EMU.trace_member_pass(members, len(counts), 256, 128, P, 1, SLOTS, 1, 4, mtl)
        assert len(z) == sum(counts) * 128 and len({a for a in z}) == len(z)


@pytest.mark.parametrize("nt,mtl", TILES)
@pytest.mark.parametrize("mats,K,N,SK", SHAPES)
def test_hoisted_decode_stores_the_same_bits(nt: int, mtl: int, mats: int, K: int, N: int, SK: int) -> None:
    """The decode-hoisted wide arm stores the same Z, and its decode is a (n tile, k tile) site counted once."""

    members = _members(len(COUNTS), COUNTS, R, SLOTS, 100 + nt * 10 + mtl)
    P = R * SLOTS
    per_split = K // 16 // SK
    for ucount in (len(COUNTS), len(COUNTS) - 3):
        up = EMU.trace_grouped(members, ucount, K, N, P, SK, SLOTS, mats, nt)
        wide = EMU.trace_member_pass(members, ucount, K, N, P, SK, SLOTS, mats, nt, mtl)
        z, decodes = EMU.trace_member_pass_hoisted(members, ucount, K, N, P, SK, SLOTS, mats, nt, mtl)
        assert z == up == wide                       # hoisting the decode moves no bit and no store
        if not decodes:
            continue
        assert all(len(s) == nt * per_split for s in decodes.values())
        assert all(m == 1 for s in decodes.values() for m in s.values())   # one decode a site, never one a member tile


def test_hoist_decodes_once_a_k_tile_whatever_the_member_width() -> None:
    """One decode a (n tile, k tile) a program; the per-tile arm behind it re-decodes each site once a member tile."""

    members = _members(1, [200], R, SLOTS, 11)
    P = R * SLOTS
    per_split = 256 // 16 // 2
    for mtl in (1, 2, 4, 8):
        hoisted = EMU.trace_member_pass_hoisted(members, 1, 256, 128, P, 2, SLOTS, 1, 4, mtl)[1]
        per_tile = EMU.decode_sites_member_pass(members, 1, 256, 128, P, 2, SLOTS, 1, 4, mtl)
        assert hoisted.keys() == per_tile.keys()
        for prog, sites in hoisted.items():
            pt = per_tile[prog]
            assert set(sites) == set(pt)                              # the same (n tile, k tile) sites
            assert set(sites.values()) == {1}                          # whatever the member-tile width
            assert len(sites) == 4 * per_split
            live = next(iter(pt.values()))                             # every live member tile decodes the site once
            assert set(pt.values()) == {live}
            assert sum(pt.values()) == live * len(sites)
