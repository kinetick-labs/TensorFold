"""CPU contract for the lossless 12-bit BF16 face layout (pack / unpack bit-identity)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import bf16  # noqa: E402


def test_pack12_unpack12_is_bit_identical_on_cpu() -> None:
    torch.manual_seed(11)
    w = (torch.randn(17, 256) * 0.05).to(torch.bfloat16)
    bits = w.view(torch.uint16).clone()
    bits[0, :32] = bits[0, :32] & 0x007F                 # subnormals -> escape group
    bits[1, 0] = 0x7F80                                   # +inf
    bits[2, 16] = 0                                       # exact zero amid normals
    # Force a wide exponent span in one group so some lanes escape via delta > 15.
    bits[3, :16] = 0x3F80                                 # ~1.0
    bits[3, 16:] = 0x0C00                                 # tiny normals
    w = bits.view(torch.bfloat16)
    packed = bf16.pack12(w)
    assert packed.n_esc > 0
    assert packed.nbytes() < w.numel() * 2
    assert packed.nbytes() >= packed.packed.numel() + packed.emax.numel()
    got = bf16.unpack12(packed)
    assert torch.equal(got, w)


def test_pack12_rejects_k_not_multiple_of_group() -> None:
    w = torch.zeros(4, 31, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of the group"):
        bf16.pack12(w)


def test_faces_12bit_env_selects_projection_lane(monkeypatch) -> None:
    monkeypatch.delenv("TENSORFOLD_FACES_FP8", raising=False)
    monkeypatch.delenv("TENSORFOLD_FACES_12BIT", raising=False)
    w = torch.zeros(8, 64, dtype=torch.bfloat16)
    assert bf16.make_b16(w, face=True).rows12 is None
    monkeypatch.setenv("TENSORFOLD_FACES_12BIT", "1")
    assert bf16.make_b16(w).rows12 is None
    assert bf16.make_b16(w, face=True).rows12 is not None
    assert bf16.make_b16(w, copy=False).rows12 is None
    monkeypatch.setenv("TENSORFOLD_FACES_12BIT", "all")
    assert bf16.make_b16(w, copy=False).rows12 is not None
    monkeypatch.setenv("TENSORFOLD_FACES_12BIT", "yes")
    assert bf16.make_b16(w, face=True).rows12 is None


def test_faces_12bit_preferred_over_fp8_when_both_set(monkeypatch) -> None:
    monkeypatch.setenv("TENSORFOLD_FACES_FP8", "all")
    monkeypatch.setenv("TENSORFOLD_FACES_12BIT", "all")
    w = torch.zeros(8, 64, dtype=torch.bfloat16)
    # Mx8Linear needs CUDA; with both set, make_b16 must take the 12-bit branch before importing FP8.
    face = bf16.make_b16(w, face=True)
    assert face.rows12 is not None
    assert face.rows8 is None
    monkeypatch.delenv("TENSORFOLD_FACES_12BIT", raising=False)
    # Without 12-bit, FP8 path would run — skip if no CUDA / no Mx8Linear.
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("TENSORFOLD_FACES_FP8", "1")
    assert bf16.make_b16(w, face=True).rows8 is not None
