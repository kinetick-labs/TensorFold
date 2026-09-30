"""Flash Next's n-gram rows asked for ahead of the verify: the pages a gather will copy, and nothing it would not."""

from __future__ import annotations

import json
import os
import struct
from types import SimpleNamespace

import numpy as np

from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.host_table import BF16Table, read_header
from tensorfold.families.qwen4_exp.cuda.ngram import NGram


def _shard(path, name: str, rows: np.ndarray) -> None:
    data = rows.tobytes()
    header = json.dumps({name: {"dtype": "BF16", "shape": list(rows.shape), "data_offsets": [0, len(data)]}})
    header = header.encode() + b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _table(tmp_path, counts=(7, 5), width=160):
    rng = np.random.default_rng(0)
    shards = [rng.integers(0, 1 << 16, size=(r, width), dtype=np.uint16) for r in counts]
    files = []
    for i, rows in enumerate(shards):
        path, name = tmp_path / f"s{i}.safetensors", f"ple.shard_{i}.weight"
        _shard(path, name, rows)
        files.append((path, read_header(path)[name]))
    return BF16Table(files), np.concatenate(shards), files


def _advice(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, at, size, advice: calls.append((fd, at, size, advice)),
                        raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_WILLNEED", 3, raising=False)
    return calls


def test_asked_rows_are_the_rows_bytes_in_their_files(tmp_path, monkeypatch):
    table, _, files = _table(tmp_path)
    calls = _advice(monkeypatch)
    table.will_need(np.array([[0, 8], [8, 11]]))                  # a repeated row is asked for once
    width = 160 * 2
    data = [8 + struct.unpack("<Q", open(p, "rb").read(8))[0] for p, _ in files]
    fds = {fd for fd, *_ in calls}
    assert len(fds) == 2 and all(advice == os.POSIX_FADV_WILLNEED for *_, advice in calls)
    got = sorted((os.readlink(f"/proc/self/fd/{fd}"), at, size) for fd, at, size, _ in calls)
    want = sorted([(str(files[0][0]), data[0], width), (str(files[1][0]), data[1] + 1 * width, width),
                   (str(files[1][0]), data[1] + 4 * width, width)])
    assert got == want


def test_a_decode_gather_asks_first_and_copies_the_same_bits(tmp_path, monkeypatch):
    table, rows, _ = _table(tmp_path)
    calls = _advice(monkeypatch)
    ids = np.array([3, 11, 7, 0, 3])
    assert np.array_equal(table.gather(ids), rows[ids])
    assert len(calls) == 4
    monkeypatch.setattr(host_table, "PREFETCH", False)
    (tmp_path / "off").mkdir()
    off, _, _ = _table(tmp_path / "off")
    calls.clear()
    assert np.array_equal(off.gather(ids), rows[ids]) and calls == []


def test_a_table_without_the_call_just_gathers(tmp_path, monkeypatch):
    table, rows, _ = _table(tmp_path)

    def refuse(*_):
        raise OSError("no advice here")

    monkeypatch.setattr(os, "posix_fadvise", refuse, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_WILLNEED", 3, raising=False)
    ids = np.array([1, 9])
    table.will_need(ids)
    assert np.array_equal(table.gather(ids), rows[ids])


def _model(eos: int = 2):
    ngram = NGram(vocab=1000, ngram_size=3, heads_per_ngram=4, vocab_base=500, divisor=16, shards=1, seed=7,
                  eos=eos, embed_dim=32)
    asked = []
    table = SimpleNamespace(will_need=lambda ids: asked.append(np.asarray(ids).copy()))
    layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=SimpleNamespace(table=table, ngram=ngram)),
              SimpleNamespace(ple=SimpleNamespace(table=SimpleNamespace(), ngram=ngram))]   # a table that can't
    return SimpleNamespace(cfg=SimpleNamespace(ngram_size=3), layers=layers), ngram, asked


def test_rows_asked_token_by_token_are_the_ids_stage_looks_up():
    from tensorfold.families.qwen4_exp.cuda.forward import ask_ple_rows

    w, ngram, asked = _model()
    rng = np.random.default_rng(1)
    for history in ([2, 2], [5, 2], [9, 17]):
        st = SimpleNamespace(ple_history=np.array(history, dtype=np.int64))
        window = [int(t) for t in rng.integers(0, 40, size=6)] + [2, 31, 2, 2, 8]   # end tokens reset n-grams
        asked.clear()
        for n in range(1, len(window) + 1):
            ask_ple_rows(w, st, window[:n])
        want = ngram.ids(st.ple_history, np.asarray(window, dtype=np.int64))
        assert np.array_equal(np.concatenate(asked), want)


def test_nothing_is_asked_without_an_ngram_history():
    from tensorfold.families.qwen4_exp.cuda.forward import ask_ple_rows

    w, _, asked = _model()
    ask_ple_rows(w, SimpleNamespace(ple_history=None), [4])
    ask_ple_rows(w, SimpleNamespace(ple_history=np.array([2, 2])), [])
    assert asked == []
