"""Console rendering, cards and the narration timeline."""

from __future__ import annotations

import numpy as np
import pytest

from heatvision.config import DEFAULT
from heatvision.viz.cards import caption, metrics_card, title_card
from heatvision.viz.narration import SCRIPT, Line, caption_at, check_overlaps
from heatvision.viz.overlay import annotate_view, attach_cell_items, compose_frame, draw_cell
from tests.test_pipeline import make_station


@pytest.fixture(scope="module")
def recorded():
    station = make_station(seed=4242, hazard_rate=0.5)
    record = None
    for _ in range(40):
        record = station.step()
    attach_cell_items(record, station.scene)
    return record, station


class TestNarrationTimeline:
    def test_the_script_is_ordered_and_starts_at_zero(self):
        starts = [line.start_s for line in SCRIPT]
        assert starts == sorted(starts)
        assert starts[0] == 0.0

    def test_caption_at_returns_the_most_recent_cue(self):
        assert caption_at(-1.0) == ""
        assert caption_at(SCRIPT[1].start_s + 0.1) == SCRIPT[1].caption
        assert caption_at(10_000.0) == SCRIPT[-1].caption

    def test_overlap_check_flags_a_line_that_runs_long(self, tmp_path):
        # two "clips" one second apart; the checker needs real audio, so with
        # missing files it must simply report nothing rather than crash
        clips = [(0.0, tmp_path / "a.wav"), (1.0, tmp_path / "b.wav")]
        assert check_overlaps(clips) == []

    def test_the_script_fits_a_three_minute_video(self):
        assert SCRIPT[-1].start_s < 120.0

    def test_lines_are_immutable(self):
        with pytest.raises(Exception):
            SCRIPT[0].start_s = 5.0  # type: ignore[misc]

    def test_a_custom_line_can_be_built(self):
        line = Line(1.0, "cap", "text")
        assert line.start_s == 1.0 and line.caption == "cap"


class TestConsole:
    def test_compose_frame_is_1080p_bgr(self, recorded):
        record, station = recorded
        frame = compose_frame(record, station.stats.summary(), DEFAULT)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8

    def test_the_console_is_not_a_blank_canvas(self, recorded):
        record, station = recorded
        frame = compose_frame(record, station.stats.summary(), DEFAULT)
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 200

    def test_annotate_view_upscales_and_keeps_shape(self, recorded):
        record, _station = recorded
        view = annotate_view(record.rgb, record, 512)
        assert view.shape == (512, 512, 3)

    def test_cell_view_renders(self, recorded):
        record, _station = recorded
        assert draw_cell(record, DEFAULT, 320).shape == (320, 320, 3)

    def test_caption_overlay_changes_the_bottom_strip(self, recorded):
        record, station = recorded
        frame = compose_frame(record, station.stats.summary(), DEFAULT)
        captioned = caption(frame.copy(), "a caption")
        assert not np.array_equal(frame[-120:], captioned[-120:])

    def test_empty_caption_is_a_no_op(self, recorded):
        record, station = recorded
        frame = compose_frame(record, station.stats.summary(), DEFAULT)
        assert np.array_equal(caption(frame.copy(), ""), frame)


class TestCards:
    def test_title_card_renders(self):
        assert title_card(960, 540, bullets=("one", "two")).shape == (540, 960, 3)

    def test_metrics_card_survives_an_empty_report(self):
        assert metrics_card({}).shape == (1080, 1920, 3)

    def test_metrics_card_renders_a_real_report(self):
        report = {
            "detection": {
                "n_images": 400,
                "mAP@0.5": 0.9,
                "mAP@[0.5:0.95]": 0.5,
                "inference_ms_per_frame": 9.0,
            },
            "end_to_end": {
                "totals": {
                    "hazards_presented": 200,
                    "hazards_removed": 160,
                    "removal_rate": 0.8,
                    "hazards_reached_crusher": 20,
                    "false_positive_picks": 3,
                    "emergency_stops": 6,
                }
            },
        }
        assert metrics_card(report).shape == (1080, 1920, 3)
