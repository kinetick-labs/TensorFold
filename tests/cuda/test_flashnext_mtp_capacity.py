"""A stream whose MTP draft cache can no longer hold its chain is served without drafting, never refused.

The served failure (an agent conversation past ~117k tokens) was a raise inside ``_draft_all``'s chained-draft
``mtp_stage``: the MTP head is optional, so running out of room for it must cost that stream its drafts, not the
request. These tests drive that exact branch — a window the memory gate has stopped growing while the draft chain
still wants one more row — and assert the stream finishes with serial (undrafted) decoding's tokens, while the
rest of the batch keeps drafting.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]


def _serial(w, prompt, count):
    """Serial decoding's tokens: the tokens a drafted (and a degraded, undrafted) stream must share."""

    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    return serial_decode(e, prefill(e, prompt, None), count, None).tokens


def _feed(dec, want):
    """Run rounds until ``want()`` holds (or every stream is done)."""

    while dec.live() and not want():
        dec.finish(dec.round())


def test_a_draft_chain_that_would_pass_the_cache_is_served_plainly():
    """The chain's next ``mtp_stage`` would take ``st.mtp_len`` past ``st.capacity``: stop chaining, serve the reply."""

    w = _model()
    prompt, count = list(PROMPTS[1]), 25
    ref = _serial(w, prompt, count)
    dec = MultiDecoder(w, slots=1, capacity=1024, depth=3, confidence=0.3)
    s = Stream(list(prompt), count, None, draft=True)
    dec.admit(s)
    _feed(dec, lambda: bool(s.drafts) and len(s.out) >= 3)
    assert s.drafts, "the stream must be drafting before its cache fills"

    st = s.st
    st.capacity = st.limit = st.capacity         # the gate froze this stream's window where it is
    st.mtp_drafted = 0
    keep = [1, 2, 3]
    st.set_mtp_len(st.capacity - len(keep))      # room for the kept rows, none for one more chain row
    dec._draft_all([(s, 0, keep)])               # the old code raised here, in the chain's mtp_stage
    assert st.mtp_off and not s.drafts           # it dropped to plain decode rather than refusing

    while dec.live():                            # the rest of the reply, without drafting
        dec.finish(dec.round())
    assert s.out == ref                          # served, and the tokens are serial decoding's


def test_only_the_stream_whose_draft_cache_is_full_stops_drafting():
    """One stream's full draft cache must not cost the others their drafts: the batch keeps drafting."""

    w = _model()
    refs = [_serial(w, list(p), 30) for p in (PROMPTS[0], PROMPTS[1])]
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3)
    a = Stream(list(PROMPTS[0]), 30, None, draft=True)
    b = Stream(list(PROMPTS[1]), 30, None, draft=True)
    dec.admit(a)
    dec.admit(b)
    _feed(dec, lambda: bool(a.drafts) and bool(b.drafts) and len(a.out) >= 3 and len(b.out) >= 3)
    assert a.drafts and b.drafts

    a.st.capacity = a.st.limit = a.st.capacity   # a's window is frozen, and its draft cache is exactly full
    a.st.mtp_drafted = 0
    a.st.set_mtp_len(a.st.capacity)
    before = b.drafted
    while dec.live():
        dec.finish(dec.round())

    assert a.st.mtp_off and not a.drafts             # a dropped to plain decode
    assert not b.st.mtp_off and b.drafted > before   # b kept drafting
    assert a.out == refs[0] and b.out == refs[1]     # both served their solo runs
