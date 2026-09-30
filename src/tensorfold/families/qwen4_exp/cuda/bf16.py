"""The NVFP4 checkpoint's bf16 linears on a row-invariant Triton matmul, and optional decode-side copies.

Two copies exist:

* **e4m3** (``TENSORFOLD_FACES_FP8``) — half the bytes, not bit-exact; the upstream rule declines it as a
  precision trade. Kept for the fork's existing Spark measurements.
* **12-bit shared-exponent** (``TENSORFOLD_FACES_12BIT``) — ~0.78×–0.90× the stored bytes, **bit-exact** BF16
  on unpack (the owner's suggested lossless attack on the dense faces). Groups of 32 along K share a max
  exponent; each value is sign + 4-bit delta + 7-bit mantissa when it fits, else an escape slot holds the
  original BF16 group. Decode reads the packed form; prompts keep the stored rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

HAS_TRITON = True
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    HAS_TRITON = False

BN = 64
BK = 64                     # the K block a program reads a step (the split keeps slices whole blocks)
GS = 32                     # the MLX group size this module's quantize4 emits; also the 12-bit group
ESC = 255                   # emax sentinel: this group is in esc_vals[esc_idx], not in the packed bytes


@dataclass
class B12:
    """Lossless 12-bit packing of a BF16 matrix: packed groups, per-group emax, dense escape index."""

    packed: torch.Tensor          # uint8 [n, k * 3 // 2]
    emax: torch.Tensor            # uint8 [n, k // GS]; ESC => esc_vals[esc_idx[n, g]]
    esc_idx: torch.Tensor         # int32 [n, k // GS]; -1 if packed
    esc_vals: torch.Tensor        # bf16 [max(n_esc, 1), GS] (dummy row when none escape)
    n: int
    k: int
    n_esc: int = 0                # real escape groups (nbytes ignores the dummy row)

    def nbytes(self) -> int:
        return (self.packed.numel() + self.emax.numel() + self.esc_idx.numel() * 4
                + self.n_esc * GS * 2)


@dataclass
class B16:
    """A BF16 matrix [n, k] as the checkpoint stores it, with optional decode-side copies."""

    weight: torch.Tensor      # [n, k] bf16, contiguous
    n: int
    k: int
    rows8: object = None      # tensorfold.cuda.nvfp4.linear.Mx8Linear, or None
    rows12: B12 | None = None

    def nbytes(self) -> int:
        n = self.weight.numel() * self.weight.element_size()
        if self.rows8 is not None:
            n += self.rows8.nbytes()
        if self.rows12 is not None:
            n += self.rows12.nbytes()
        return n


def faces_8bit() -> str:
    """Which BF16 faces also get an 8-bit copy, from ``TENSORFOLD_FACES_FP8`` across a load.

    ``""``: none. ``"1"``: the projections ``weights.face`` builds - the DeltaNet and attention linears, which
    are 110 of a layer's 147.6 MiB. ``"all"``: every BF16 face too (router, hyper-connections, shared expert),
    which measured a 28% cheaper round but 48% of drafts accepted against 64%: those small faces steer which
    experts run and how the streams mix, so a coarser copy of them costs the MTP head more than it saves.
    """

    return os.environ.get("TENSORFOLD_FACES_FP8", "").strip().lower()


def faces_12bit() -> str:
    """Which BF16 faces also get a lossless 12-bit copy, from ``TENSORFOLD_FACES_12BIT``.

    Same names as the 8-bit switch (``1`` / ``all``), but because the unpack is bit-identical to the stored
    rows the ``all`` lane may include the lm_head — the drafts' head stays calibrated. ``""``: none.
    """

    return os.environ.get("TENSORFOLD_FACES_12BIT", "").strip().lower()


DECODE_FP8 = True             # whether a round reads the e4m3 copy; a probe flips it, a server leaves it on
DECODE_12BIT = True           # whether a round reads the lossless 12-bit copy


def _want_fp8(copy: bool | None, *, face: bool) -> bool:
    mode = faces_8bit()
    if copy is False:
        return False
    if copy is True:
        return True
    if mode == "all":
        return True
    return mode == "1" and face


def _want_12bit(copy: bool | None, *, face: bool) -> bool:
    mode = faces_12bit()
    if mode == "all":
        return True                              # lossless: head and dense faces alike
    if mode == "1":
        return face or copy is True
    return False


def pack12(weight: torch.Tensor) -> B12:
    """BF16 [n, k] -> lossless 12-bit groups of ``GS`` along K (bit-identical ``unpack12``)."""

    w = weight.to(torch.bfloat16).contiguous()
    n, k = w.shape
    if k % GS:
        raise ValueError(f"12-bit pack: K {k} is not a multiple of the group {GS}")
    bits = w.view(torch.uint16).to(torch.int32)
    sign, exp, mant = (bits >> 15) & 1, (bits >> 7) & 0xFF, bits & 0x7F
    groups = k // GS
    exp_g, sign_g, mant_g = exp.view(n, groups, GS), sign.view(n, groups, GS), mant.view(n, groups, GS)
    special = (exp_g == 0xFF) | ((exp_g == 0) & (mant_g != 0))
    zero = (exp_g == 0) & (mant_g == 0)
    emax = torch.where(~special, exp_g, torch.zeros_like(exp_g)).amax(-1)
    delta = emax.unsqueeze(-1) - exp_g
    fits_norm = (~special) & (~zero) & (delta >= 0) & (delta <= 15)
    fits_zero = zero & (emax.unsqueeze(-1) <= 15)
    fits = fits_norm | fits_zero
    escaped = (~fits).any(-1)
    emax_out = torch.where(escaped, torch.full_like(emax, ESC, dtype=torch.int64), emax).to(torch.uint8)

    code = torch.zeros((n, groups, GS), dtype=torch.int32, device=w.device)
    code = torch.where(fits_norm, (sign_g << 11) | (delta << 7) | mant_g, code)
    code = torch.where(fits_zero, (sign_g << 11) | (emax.unsqueeze(-1) << 7), code)
    packed = _pack_12bit_codes(code.view(n, k))

    esc_index = escaped.nonzero(as_tuple=False)
    n_esc = int(esc_index.shape[0])
    esc_idx = torch.full((n, groups), -1, dtype=torch.int32, device=w.device)
    # Keep at least one escape row so the fused kernel always has a valid pointer (mask keeps it unread).
    esc_vals = torch.zeros((max(n_esc, 1), GS), dtype=torch.bfloat16, device=w.device)
    if n_esc:
        rows, blocks = esc_index[:, 0], esc_index[:, 1]
        esc_idx[rows, blocks] = torch.arange(n_esc, device=w.device, dtype=torch.int32)
        esc_vals[:n_esc] = w.view(n, groups, GS)[rows, blocks]
    return B12(packed, emax_out.contiguous(), esc_idx.contiguous(), esc_vals, n, k, n_esc)


def unpack12(b: B12) -> torch.Tensor:
    """The packed face back to BF16 [n, k], bit-identical to what ``pack12`` saw."""

    n, k, groups = b.n, b.k, b.k // GS
    codes = _unpack_12bit_codes(b.packed, n, k).view(n, groups, GS)
    emax = b.emax.to(torch.int32).clamp(max=254).unsqueeze(-1)          # ESC lanes overwritten below
    sign, delta, mant = (codes >> 11) & 1, (codes >> 7) & 0xF, codes & 0x7F
    bits = (sign << 15) | ((emax - delta) << 7) | mant
    out = bits.to(torch.uint16).view(torch.bfloat16).reshape(n, groups, GS).clone()
    hit = b.esc_idx >= 0
    if hit.any():
        rows, blocks = hit.nonzero(as_tuple=True)
        out[rows, blocks] = b.esc_vals[b.esc_idx[rows, blocks].long()]
    return out.reshape(n, k)


def _pack_12bit_codes(codes: torch.Tensor) -> torch.Tensor:
    """int32 codes [n, k] in 0..4095 -> uint8 [n, k*3//2], two codes -> three bytes."""

    n, k = codes.shape
    c = codes.view(n, k // 2, 2)
    b0 = (c[..., 0] & 0xFF).to(torch.uint8)
    b1 = (((c[..., 0] >> 8) & 0xF) | ((c[..., 1] & 0xF) << 4)).to(torch.uint8)
    b2 = ((c[..., 1] >> 4) & 0xFF).to(torch.uint8)
    return torch.stack((b0, b1, b2), dim=-1).reshape(n, k * 3 // 2).contiguous()


def _unpack_12bit_codes(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """uint8 [n, k*3//2] -> int32 codes [n, k]."""

    p = packed.view(n, k // 2, 3).to(torch.int32)
    c0 = p[..., 0] | ((p[..., 1] & 0xF) << 8)
    c1 = ((p[..., 1] >> 4) & 0xF) | (p[..., 2] << 4)
    return torch.stack((c0, c1), dim=-1).reshape(n, k)


def make_b16(weight: torch.Tensor, *, copy: bool | None = None, face: bool = False) -> B16:
    """A B16 face, with decode copies the environment asks for.

    ``face=True`` marks a projection ``weights.face`` builds (DeltaNet / attention): the ``TENSORFOLD_FACES_*=1``
    lane. ``copy=True`` / ``False`` force the e4m3 copy on or off; ``None`` defers to the environment.
    ``TENSORFOLD_FACES_12BIT=all`` packs every face — including the lm_head (``copy=False``) — because the
    unpack is bit-identical.
    """

    w = weight.to(torch.bfloat16).contiguous()
    rows8, rows12 = None, None
    # Prefer the lossless 12-bit copy when both lanes are asked for: decode already prefers it, and a second
    # e4m3 face would only burn HBM for a path that never runs.
    if _want_12bit(copy, face=face):
        rows12 = pack12(w)
    elif _want_fp8(copy, face=face):
        from tensorfold.cuda.nvfp4.linear import Mx8Linear

        rows8 = Mx8Linear.from_bf16(w)
    return B16(w, int(w.shape[0]), int(w.shape[1]), rows8, rows12)


def split_k(n: int, k: int, target: int = 160, bk: int = BK) -> int:
    """K slices by the weight's shape alone: a power of two with whole BK blocks a slice."""

    tiles = -(-n // BN)
    blocks = k // bk
    sk = 1
    while sk < 32 and tiles * sk < target and blocks % (sk * 2) == 0 and blocks // (sk * 2) >= 1:
        sk *= 2
    return sk


if HAS_TRITON:
    @triton.jit
    def _b16mm(X, W, OUT, PART, M, x_stride,
               N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
               BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
        """x [M, K] @ W.T -> [M, N]: K in BK steps in order, a tensor-core dot each, fp32 accumulators."""

        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        n_ok = rn < N
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for i in range(NB):
            k0 = (pid_s * NB + i) * BK
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        out_mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc if F32 else acc.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)

    @triton.jit
    def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
        """The K slices summed in slice order, one fp32 add a slice, in one launch for either output face."""

        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < total
        acc = tl.load(PART + offs, mask=ok, other=0.0)
        for s in tl.static_range(1, SK):
            acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
        tl.store(OUT + offs, acc if F32 else acc.to(tl.bfloat16), mask=ok)


    @triton.jit
    def _b12mm(X, PACKED, EMAX, ESC_IDX, ESC_VAL, OUT, PART, M, x_stride,
               N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
               BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, GS: tl.constexpr = 32):
        """x [M, K] @ packed12(W).T -> [M, N]: decode each K-group into SRAM, then a GS-wide dot.

        Packed bytes (and rare escape groups) leave HBM — not a second full BF16 face. Groups of ``GS``
        share an exponent; a micro-dot per group keeps the W tile at ``[BLOCK_N, GS]`` instead of staging
        a full ``[BLOCK_N, BK]`` decode.
        """

        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_ok = rm < M
        n_ok = rn < N
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK
        NG: tl.constexpr = BK // GS
        NPAIRS: tl.constexpr = GS // 2
        PACKED_STRIDE: tl.constexpr = K * 3 // 2
        NGROUPS: tl.constexpr = K // GS
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        pair = tl.arange(0, NPAIRS)
        lane = tl.arange(0, GS)
        for i in range(NB):
            k0 = (pid_s * NB + i) * BK
            for g in tl.static_range(0, NG):
                block = k0 // GS + g
                xg = tl.load(
                    X + rm[:, None] * x_stride + (k0 + g * GS + lane)[None, :],
                    mask=m_ok[:, None], other=0.0)
                idx = tl.load(ESC_IDX + rn * NGROUPS + block, mask=n_ok, other=-1)
                e = tl.load(EMAX + rn * NGROUPS + block, mask=n_ok, other=0).to(tl.int32)
                base = rn[:, None] * PACKED_STRIDE + block * (GS * 3 // 2) + pair[None, :] * 3
                b0 = tl.load(PACKED + base + 0, mask=n_ok[:, None], other=0).to(tl.int32)
                b1 = tl.load(PACKED + base + 1, mask=n_ok[:, None], other=0).to(tl.int32)
                b2 = tl.load(PACKED + base + 2, mask=n_ok[:, None], other=0).to(tl.int32)
                c0 = b0 | ((b1 & 0xF) << 8)
                c1 = ((b1 >> 4) & 0xF) | (b2 << 4)
                codes = tl.reshape(tl.join(c0, c1), (BLOCK_N, GS))
                sign = (codes >> 11) & 1
                delta = (codes >> 7) & 0xF
                mant = codes & 0x7F
                bits = ((sign << 15) | ((e[:, None] - delta) << 7) | mant).to(tl.uint16)
                decoded = bits.to(tl.bfloat16, bitcast=True)
                esc = tl.load(
                    ESC_VAL + idx[:, None] * GS + lane[None, :],
                    mask=n_ok[:, None] & (idx[:, None] >= 0), other=0.0)
                wg = tl.where(idx[:, None] >= 0, esc, decoded)
                wg = tl.where(n_ok[:, None], wg, 0.0)
                acc = tl.dot(xg, tl.trans(wg), acc)
        out_mask = m_ok[:, None] & n_ok[None, :]
        if SK == 1:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc if F32 else acc.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def _matmul12(x: torch.Tensor, b12: B12, *, out: torch.Tensor | None, f32: bool, sk: int,
              num_warps: int, num_stages: int, block_n: int, bk: int) -> torch.Tensor:
    """Decode-side matmul against a lossless 12-bit face (packed bytes leave HBM; bits match the stored rows)."""

    m, k = x.shape
    if out is None:
        out = torch.empty((m, b12.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, b12.n) or not out.is_contiguous() or (out.dtype == torch.float32) != f32:
        raise ValueError(f"b12 matmul: out {tuple(out.shape)} {out.dtype} must be contiguous ({m}, {b12.n})")
    part = torch.empty((sk, m, b12.n), dtype=torch.float32, device=x.device) if sk > 1 else out
    bm = 16  # decode windows are small; keep the W tile modest for the per-row unpack
    block_n = min(block_n, 16)
    grid = (triton.cdiv(m, bm), -(-b12.n // block_n), sk)
    _b12mm[grid](
        x, b12.packed, b12.emax, b12.esc_idx, b12.esc_vals, out, part, m, x.stride(0),
        N=b12.n, K=k, SK=sk, BM=bm, BLOCK_N=block_n, BK=bk, F32=f32, GS=GS,
        num_warps=num_warps, num_stages=num_stages)
    if sk > 1:
        total = m * b12.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def matmul(x: torch.Tensor, b: B16, *, out: torch.Tensor | None = None, f32: bool = False,
           sk: int | None = None, num_warps: int = 4, num_stages: int = 3,
           block_n: int = BN, bk: int = BK, prefill: bool = True) -> torch.Tensor:
    """x [M, K] bf16 @ b.T -> [M, N] bf16 (or fp32 sums), K slices summed in slice order.

    ``prefill`` says which step is asking, because that is what decides whether a face with a decode copy
    reads it: a round verifies a handful of rows, so its reads are the weight's bytes, while a prompt's rows
    are many and the stored rows stay the cheaper read. Row count would be the wrong test - a prompt arrives
    in chunks, and a row's bits must not depend on how the work was scheduled.
    """

    if not HAS_TRITON:
        raise RuntimeError("the BF16 matmul needs Triton (the CUDA engine's environment)")
    m, k = x.shape
    if k != b.k or x.stride(1) != 1:
        raise ValueError(f"b16 matmul: x {tuple(x.shape)} does not match K={b.k}")
    if k % bk:
        raise ValueError(f"b16 matmul: K {k} is not a multiple of the K block {bk}")
    if not f32 and not prefill:
        if DECODE_12BIT and b.rows12 is not None:
            sk = int(sk) if sk else split_k(b.n, b.k, bk=bk)
            return _matmul12(x, b.rows12, out=out, f32=f32, sk=sk, num_warps=num_warps,
                             num_stages=num_stages, block_n=block_n, bk=bk)
        if DECODE_FP8 and b.rows8 is not None:
            return b.rows8(x, out=out)
    sk = int(sk) if sk else split_k(b.n, b.k, bk=bk)
    if out is None:
        out = torch.empty((m, b.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, b.n) or not out.is_contiguous() or (out.dtype == torch.float32) != f32:
        raise ValueError(f"b16 matmul: out {tuple(out.shape)} {out.dtype} must be a contiguous ({m}, {b.n}), "
                         f"dtype matching f32={f32}")
    part = torch.empty((sk, m, b.n), dtype=torch.float32, device=x.device) if sk > 1 else out
    bm = 128 if m > 128 else 16
    grid = (triton.cdiv(m, bm), -(-b.n // block_n), sk)
    _b16mm[grid](x, b.weight, out, part, m, x.stride(0), N=b.n, K=k, SK=sk, BM=bm,
                 BLOCK_N=block_n, BK=bk, F32=f32, num_warps=num_warps, num_stages=num_stages)
    if sk > 1:
        total = m * b.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def quantize4(w: torch.Tensor, chunk: int = 8192, out: str = "q4"):
    """bf16 (N, K) -> MLX affine 4-bit in groups of 32, for weights that only draft (the MTP head's lm_head copy)."""

    from . import qmm

    n, k = w.shape
    words, scales, biases = [], [], []
    for lo in range(0, n, chunk):
        part = w[lo:lo + chunk].float()
        g = part.reshape(-1, k // GS, GS)
        mn = g.min(dim=-1, keepdim=True).values
        mx = g.max(dim=-1, keepdim=True).values
        s = ((mx - mn) / 15.0).clamp(min=1e-8)
        q = ((g - mn) / s).round().clamp(0, 15).to(torch.uint8).reshape(-1, k)
        packed = (q[:, 1::2] << 4 | q[:, 0::2]).view(torch.uint32)
        words.append(packed)
        scales.append(s.reshape(-1, k // GS).to(torch.bfloat16))
        biases.append(mn.reshape(-1, k // GS).to(torch.bfloat16))
    return qmm.make_q4(torch.cat(words), torch.cat(scales), torch.cat(biases))


def b16_from_rows(rows: torch.Tensor, *, copy: bool | None = None, face: bool = False) -> "_Routed":
    """A [n, k] bf16 matrix with the ``qmm.matmul`` face (a ``kernel`` tag)."""

    b = make_b16(rows, copy=copy, face=face)
    routed = _Routed(b)
    routed._face = face
    return routed


class _Routed:
    """A bf16 matrix tagged ``kernel == 'b16'`` so ``forward._mm`` routes it here; ``stack`` joins rows."""

    kernel = "b16"

    def __init__(self, b: B16) -> None:
        self.b = b
        self.n, self.k = b.n, b.k
        self._face = False

    @property
    def weight(self) -> torch.Tensor:
        return self.b.weight

    @property
    def rows8(self):
        return self.b.rows8

    @property
    def rows12(self):
        return self.b.rows12

    def nbytes(self) -> int:
        return self.b.nbytes()


def stack_b16(parts: list) -> _Routed:
    """Rows of several _Routed of the same K stacked in order (the hyper-connection's down + inject)."""

    rows = [p.b.weight if isinstance(p, _Routed) else p.weight for p in parts]
    face = any(getattr(p, "_face", False) for p in parts)
    return b16_from_rows(torch.cat(rows, dim=0), face=face)
