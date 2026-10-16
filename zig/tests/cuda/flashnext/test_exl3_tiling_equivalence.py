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


# --- the G=2 staged A tile (cuda_exl3.zig wideStageSmem, fn_exl3.cu warp_tile_stage) ---


@pytest.mark.parametrize("per_split", [1, 8, 20, 40, 80, 100])
def test_the_staged_fragment_is_the_global_read(per_split: int) -> None:
    """The word a lane reads from the staged row is the word load_pair reads from the global row, for every k tile."""

    for kt in range(per_split):
        for t in range(4):
            assert EMU.staged_lane_words(kt, t) == EMU.reread_lane_words(kt, t)


@pytest.mark.parametrize("per_split", [20, 40, 80])
def test_the_staged_stride_banks_each_lane_apart(per_split: int) -> None:
    """The padded row stride is 4 mod 32, so the 32 lanes' fragment word lands on 32 distinct banks."""

    sw = EMU.stage_stride(per_split)
    assert sw % 32 == 4
    assert sw >= EMU.staged_words(per_split) and sw - EMU.staged_words(per_split) < 32
    for kt in range(per_split):
        for col in EMU.staged_lane_words(kt, 0):
            banks = [(g * sw + col + t) % 32 for g in range(8) for t in range(4)]
            assert len(set(banks)) == 32


def test_the_staged_tile_reads_a_member_row_once_a_block() -> None:
    """G=2 reads each member row's split slice once a block; the re-reading arm reads the same slice once a n tile."""

    mtl, NT = 2, 8
    for per_split, staged in ((40, 40960), (20, 20480)):
        once = mtl * 16 * EMU.staged_words(per_split) * 4
        assert once == staged                                     # 32 rows x the split's slice
        assert EMU.a_bytes_reread(per_split, mtl, NT) == staged * NT


@pytest.mark.parametrize("mats,K,N,SK", SHAPES)
def test_staged_member_pass_stores_the_same_bits(mats: int, K: int, N: int, SK: int) -> None:
    """The staged wide arm stores upstream's bits, and its A loads are one read a live member row (the re-read ends)."""

    members = _members(len(COUNTS), COUNTS, R, SLOTS, 300 + mats * 7 + K)
    P = R * SLOTS
    per_split = K // 16 // SK
    for ucount in (len(COUNTS), len(COUNTS) - 3):
        up = EMU.trace_grouped(members, ucount, K, N, P, SK, SLOTS, mats, 8)
        z, gbytes = EMU.trace_member_pass_staged(members, ucount, K, N, P, SK, SLOTS, mats, 8, 2)
        assert z == up
        # one 16-row tile's split slice, read once a block: whole tiles only, dead rows in a live tile included
        one_tile = 16 * EMU.staged_words(per_split) * 4
        assert gbytes % one_tile == 0
        # the re-reading arm pays that slice once a (n tile): mtl tiles x NT n tiles a program
        assert EMU.a_bytes_reread(per_split, 2, 8) == one_tile * 2 * 8


# the gm arm: members the B operand (staged, 128 a pass), weights the A operand, columns split across the warps

# (NT, mtl) the 4 warps cover: 8 column tiles two each, or 4 tiles one each (upstream's slice split)
GM_TILES = [(8, 2), (4, 1)]

# the gm tests run a smaller member set (the staged pass walks 128 rows a program, so the naive model is heavy)
GM_R = 48
GM_COUNTS = [0, 1, 15, 16, 17, 33, 48]


@pytest.mark.parametrize("nt,mtl", GM_TILES)
@pytest.mark.parametrize("mats,K,N,SK", SHAPES)
def test_gm_arm_stores_the_same_bits_as_the_default(nt: int, mtl: int, mats: int, K: int, N: int, SK: int) -> None:
    """The gm arm stores exactly the default arm's Z: per element the split-major, warp-sub-chain-in-order reduction."""

    members = _members(len(GM_COUNTS), GM_COUNTS, GM_R, SLOTS, 777 + nt * 10 + mtl)
    P = GM_R * SLOTS
    for ucount in (len(GM_COUNTS), len(GM_COUNTS) - 3):
        gm = EMU.trace_gm_pass(members, ucount, K, N, P, SK, SLOTS, mats, nt, mtl)
        folded = EMU.fold_grouped_trace(members, ucount, K, N, P, SK, SLOTS, mats, nt)
        assert gm == folded                       # the same element, same chains, same order -> same bits
        assert len(gm) == sum(min(c, GM_R) for c in GM_COUNTS[:ucount]) * N * mats


@pytest.mark.parametrize("nt,mtl", GM_TILES)
@pytest.mark.parametrize("mats,K,N,SK", SHAPES)
def test_the_faithful_single_chain_is_not_the_default_tree(nt: int, mtl: int, mats: int, K: int, N: int, SK: int) -> None:
    """The negative control: gm_kernel's own rule (one ascending chain an element) is a DIFFERENT reduction tree."""

    members = _members(len(GM_COUNTS), GM_COUNTS, GM_R, SLOTS, 555 + nt * 10 + mtl)
    P = GM_R * SLOTS
    single = EMU.trace_gm_single_chain(members, len(GM_COUNTS), K, N, P, SK, SLOTS, mats, nt, mtl)
    folded = EMU.fold_grouped_trace(members, len(GM_COUNTS), K, N, P, SK, SLOTS, mats, nt)
    assert single != folded
    assert set(single) == set(folded)              # same elements stored, different chains -> different fp32 sum


def test_gm_a_tile_reads_a_member_row_once_a_block() -> None:
    """The gm ring stages k a stage, so a 128-member block reads a row's whole K once, not once a n tile."""

    for _mats, K, _N, _SK in SHAPES:
        once = EMU.gm_a_global_bytes(EMU.GM_BM, K)
        assert once == EMU.GM_BM * K * 2
        assert once < EMU.GM_BM * K * 2 * 8         # not the 8x the hoisted arm's NT n tiles paid
