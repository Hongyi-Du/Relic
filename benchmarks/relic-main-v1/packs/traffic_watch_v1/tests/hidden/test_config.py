"""Config validation."""

from __future__ import annotations

import pytest

from traffic_violation_system import CameraConfig, FlowDirection, SystemConfig


def test_camera_config_rejects_zero_speed_limit():
    with pytest.raises(ValueError):
        CameraConfig(
            camera_id="c", location="x",
            stop_line=((0, 0), (1, 0)),
            allowed_flow=FlowDirection.NORTH,
            speed_limit_kmh=0.0,
            pixels_per_meter=10.0,
            fps=10.0,
        )


def test_camera_config_rejects_zero_fps():
    with pytest.raises(ValueError):
        CameraConfig(
            camera_id="c", location="x",
            stop_line=((0, 0), (1, 0)),
            allowed_flow=FlowDirection.NORTH,
            speed_limit_kmh=60.0,
            pixels_per_meter=10.0,
            fps=0.0,
        )


def test_system_config_defaults():
    cfg = SystemConfig()
    assert cfg.iou_match_threshold > 0
    assert cfg.track_max_missed_frames > 0
