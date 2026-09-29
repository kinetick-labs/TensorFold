"""Video prompts on the Flash Next CPU frontend: sampled frames, frame-group blocks, rotary positions.

The tower encoding itself is CUDA-only (``tests/cuda/test_flashnext_vision.py`` covers images); everything
here runs without a GPU: the bounding of a clip, the marker expansion, and the three rotary axes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.images import ImageInputError, ImageLimits, ImageSource, split_images
from tensorfold.vision.qwen_processing import QwenImageProcessor, image_positions, media_positions
from tensorfold.vision.videos import (DEFAULT_VIDEO_LIMITS, VideoInput, VideoSource, sample_indices,
                                      video_source)


CONFIG = {"model_type": "qwen4_exp", "image_token_id": 10, "video_token_id": 11,
          "vision_start_token_id": 8, "vision_end_token_id": 9,
          "vision_config": {"spatial_merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
                            "out_hidden_size": 3, "deepstack_visual_indexes": [], "hidden_size": 4,
                            "intermediate_size": 8, "depth": 2, "in_channels": 3}}


class Tokenizer:
    TOKENS = {"<start>": 8, "<end>": 9, "<image>": 10, "<video>": 11}

    def convert_ids_to_tokens(self, token):
        return {value: key for key, value in self.TOKENS.items()}.get(token)

    def convert_tokens_to_ids(self, token):
        return self.TOKENS.get(token)

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
        ids, order = [], sorted(self.TOKENS, key=len, reverse=True)
        while text:
            found = next((t for t in order if text.startswith(t)), None)
            if found:
                ids.append(self.TOKENS[found])
                text = text[len(found):]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": ids}


class ImageProcessor:
    max_pixels, min_pixels = 1024**2, 32**2

    def __init__(self, **kwargs):
        self.options, self.calls = kwargs, []

    def __call__(self, *, images, **kwargs):
        self.calls.append((images, kwargs))
        return {"pixel_values": np.zeros((16, 1536), np.float32), "image_grid_thw": np.array([[1, 4, 4]])}


def frontend() -> QwenImageProcessor:
    return QwenImageProcessor(CONFIG, ImageProcessor(), Tokenizer())


def clip(frames: int = 4, side: int = 32, fps: float = 1.0) -> VideoInput:
    """A decoded clip already at the tower's resolution (4 frames of ``side`` x ``side`` at 1 fps)."""
    pixels = np.arange(frames * side * side * 3, dtype=np.uint8).reshape(frames, side, side, 3) % 251
    return VideoInput(pixels, tuple(range(frames)), fps)


def test_frame_choice_samples_the_rate_and_bounds_the_count():
    # 20 frames at 10 fps is two seconds: two seconds at 2 fps is four frames, spread over the whole clip
    assert sample_indices(20, 10.0, DEFAULT_VIDEO_LIMITS).tolist() == [0, 6, 13, 19]
    # a short clip keeps the minimum, a very long one the maximum, still spread over its whole length
    assert sample_indices(3, 24.0, DEFAULT_VIDEO_LIMITS).tolist() == [0, 1, 2]
    long = sample_indices(24 * 3600, DEFAULT_VIDEO_LIMITS.fps, DEFAULT_VIDEO_LIMITS)
    assert len(long) == DEFAULT_VIDEO_LIMITS.max_frames and long[0] == 0 and long[-1] == 24 * 3600 - 1


def test_frame_group_timestamps_are_the_groups_midpoints():
    assert clip(4, fps=1.0).timestamps(2) == [0.5, 2.5]
    assert clip(3, fps=2.0).timestamps(2) == [0.25, 1.0]       # an odd tail repeats its last frame (index 2)


def test_video_input_refuses_frames_that_are_not_the_contract():
    with pytest.raises(ImageInputError, match="RGB frames"):
        VideoInput(np.zeros((4, 32, 32), np.uint8), (0, 1, 2, 3), 1.0)
    with pytest.raises(ImageInputError, match="RGB frames"):
        VideoInput(np.zeros((4, 32, 32, 3), np.uint8), (0, 1, 2), 1.0)


def test_a_video_source_needs_a_data_url_or_an_allowed_https_url():
    assert video_source({"url": "data:video/mp4;base64,AAAA"}, DEFAULT_VIDEO_LIMITS, False).url.startswith("data:")
    with pytest.raises(ImageInputError, match="public HTTPS"):
        video_source({"url": "http://host/a.mp4"}, DEFAULT_VIDEO_LIMITS, False)
    with pytest.raises(ImageInputError, match="URLs are off"):
        video_source({"url": "https://host/a.mp4"}, DEFAULT_VIDEO_LIMITS, False)
    assert video_source({"url": "https://host/a.mp4"}, DEFAULT_VIDEO_LIMITS, True).url.endswith("a.mp4")
    with pytest.raises(ImageInputError, match="non-empty url"):
        video_source({"url": ""}, DEFAULT_VIDEO_LIMITS, True)


def messages(*parts):
    return [{"role": "user", "content": [{"type": "text", "text": "look"}, *parts]}]


def test_split_images_takes_video_parts_only_when_the_frontend_encodes_them():
    template, sources = split_images(messages({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}),
                                     allow_videos=True)
    assert template[0]["content"][1] == {"type": "video"} and isinstance(sources[0], VideoSource)
    with pytest.raises(ImageInputError, match="audio and video are unsupported"):
        split_images(messages({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}))
    with pytest.raises(ImageInputError, match="only in user messages"):
        split_images([{"role": "assistant", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
                     allow_videos=True)


def test_videos_do_not_count_against_the_image_budget_and_keep_their_order():
    parts = [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    template, sources = split_images(messages(*parts), limits=ImageLimits(max_images=1), allow_videos=True)
    assert [type(s).__name__ for s in sources] == ["VideoSource", "ImageSource"]
    assert [p["type"] for p in template[0]["content"]] == ["text", "video", "image"]
    with pytest.raises(ImageInputError, match="at most 1 images"):
        split_images(messages(*parts, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}),
                     limits=ImageLimits(max_images=1), allow_videos=True)


def test_media_positions_gives_every_frame_group_its_own_block():
    tokens = [100, 8, 11, 11, 11, 11, 9, 8, 11, 11, 11, 11, 9, 101]
    positions, delta, spans, frames = media_positions(tokens, [], [[2, 4, 4]], CONFIG)
    assert spans == () and frames == ((2, 6), (8, 12))
    assert positions.shape == (3, 1, len(tokens))
    assert positions[0, 0].tolist() == [0, 1, 2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 8, 9]
    assert positions[1, 0, 2:6].tolist() == [2, 2, 3, 3] and positions[2, 0, 2:6].tolist() == [2, 3, 2, 3]
    assert positions[1, 0, 8:12].tolist() == [6, 6, 7, 7] and positions[2, 0, 8:12].tolist() == [6, 7, 6, 7]
    assert len(tokens) + delta == int(positions.max()) + 1 == 10


def test_image_positions_stays_the_three_axis_image_only_view():
    ids = [8, 10, 10, 10, 10, 9, 100]
    positions, delta, spans = image_positions(ids, [[1, 4, 4]], CONFIG)
    assert spans == ((1, 5),) and positions.shape == (3, 1, len(ids)) and delta == -2


@pytest.mark.parametrize("tokens, image_grids, video_grids, message", [
    ([8, 11, 11, 9], [], None, "Video inputs are not supported"),          # a video token, no video grid
    ([8, 11, 11, 11, 11, 9], [], [[2, 4, 4]], "matching video tokens"),   # two frame groups wanted, one there
    ([8, 11, 11, 11, 11, 9], [], [[2, 3, 4]], "merge-aligned"),
    ([8, 11, 11, 11, 11, 9], [], [[2, 4]], "temporal, height and width"),
    ([8, 11, 11, 11, 11, 9], [], [[2, 4, 4], [2, 4, 4]], "matching video tokens"),
    ([8, 10, 10, 10, 10, 9], [[2, 4, 4]], None, "one frame"),             # an image grid may not carry time
    ([8, 10, 10, 10, 10, 9], [], None, "without a corresponding image"),
])
def test_a_payload_that_does_not_fit_the_prompt_is_refused(tokens, image_grids, video_grids, message):
    with pytest.raises(ValueError, match=message):
        media_positions(tokens, image_grids, video_grids, CONFIG)


def test_prepare_expands_a_video_marker_into_timestamped_frame_blocks():
    front = frontend()
    source = clip(4, side=64)
    prepared = front.prepare("a<start><video><end>b", [], videos=[source], max_prompt_tokens=64)
    assert prepared.image_spans == () and prepared.video_spans == ((15, 19), (34, 38))
    assert prepared.visual_tokens == 8 and prepared.video_hashes == (source.content_hash,)
    assert prepared.token_ids[:14] == tuple([97] + [ord(c) for c in "<0.5 seconds>"])
    assert prepared.token_ids[14:20] == (8, 11, 11, 11, 11, 9)
    assert prepared.token_ids[33:39] == (8, 11, 11, 11, 11, 9) and prepared.token_ids[39] == 98
    assert prepared.video_grid_thw.tolist() == [[2, 4, 4]]
    # [frame groups * h * w, channels * temporal * patch * patch]: 2 groups, 4 x 4 patches of 2 frames of 16 x 16
    assert prepared.video_pixel_values.shape == (32, 1536) and prepared.pixel_values.shape == (0, 1536)
    assert prepared.position_ids[0, 0].tolist() == ([*range(15)] + [15] * 4 + [*range(17, 32)] + [32] * 4
                                                    + [34, 35])
    assert prepared.position_ids[1, 0, 15:19].tolist() == [15, 15, 16, 16]
    assert prepared.position_ids[2, 0, 15:19].tolist() == [15, 16, 15, 16]
    assert prepared.rope_delta == -4 and len(prepared.token_ids) + prepared.rope_delta == 36
    assert all(not array.flags.writeable for array in (prepared.video_pixel_values, prepared.video_grid_thw))


def test_prepare_refuses_a_video_the_checkpoint_or_the_prompt_cannot_take():
    front = frontend()
    with pytest.raises(ValueError, match="exactly one video marker"):
        front.prepare("<start><video><end><start><video><end>", [], videos=[clip(4)])
    with pytest.raises(ValueError, match="names no video token"):
        QwenImageProcessor({key: value for key, value in CONFIG.items() if key != "video_token_id"},
                           ImageProcessor(), Tokenizer()).prepare("<start><video><end>", [], videos=[clip(4)])
    with pytest.raises(ValueError, match="patch grid"):
        front.prepare("<start><video><end>", [], videos=[clip(4, side=48)])      # 48 is not patch * merge
    # the visual-token budget itself is the image budget: a clip is bounded by ``video_size``/``load_videos``
    assert front.prepare("<start><video><end>", [], videos=[clip(4)], max_visual_tokens=1).visual_tokens == 2


def test_video_size_keeps_frames_at_the_tower_resolution_within_the_budget():
    front = frontend()
    assert front.video_size(4, 512, 512) == (512, 512)
    height, width = front.video_size(256, 1080, 1920)
    assert height % 32 == 0 and width % 32 == 0 and abs(height / width - 1080 / 1920) < 0.1
    with pytest.raises(ValueError, match="at least 2 frames"):
        front.video_size(1, 512, 512)
    with pytest.raises(ValueError, match="aspect ratio"):
        front.video_size(4, 32, 10000)


def test_a_request_bounds_its_videos_and_refuses_urls_when_they_are_off():
    from tensorfold.vision.videos import load_videos

    size = frontend().video_size
    with pytest.raises(ImageInputError, match="at most 2 videos"):
        load_videos([VideoSource("data:video/mp4;base64,AA")] * 3, size)
    with pytest.raises(ImageInputError, match="URLs are off"):
        load_videos([VideoSource("https://host/a.mp4")], size)
    with pytest.raises(ImageInputError, match="require PyAV"):
        load_videos([VideoSource("data:video/mp4;base64,AAAA")], size)


def test_the_video_holder_is_not_an_image_source():
    # ``prepare_images`` tells the two apart by type: a clip must never be handed to ``load_images``
    assert not isinstance(VideoSource("data:video/mp4;base64,AAAA"), ImageSource)
