import json

from terrainlens.av_perception import (
    _normalise_payload,
    decide_driving_action,
    point_in_polygon,
)


def test_normalise_payload_rejects_invalid_boxes_and_unknown_types():
    data = _normalise_payload({
        "hazards": [
            {"type": "made_up", "severity": "wild", "confidence": 4,
             "bbox": [0.1, 0.1, 0.4, 0.5]},
            {"type": "pothole", "bbox": [0.2, 0.2, 1.4, 0.5]},
            {"type": "crack", "bbox": [0.4, 0.4, 0.2, 0.6]},
        ],
        "drivable_corridor": [[0.2, 0.9], [0.8, 0.9], [0.5, 0.2]],
    })
    assert len(data["hazards"]) == 1
    assert data["hazards"][0]["type"] == "other_obstacle"
    assert data["hazards"][0]["severity"] == "medium"
    assert data["hazards"][0]["confidence"] == 1.0
    assert len(data["drivable_corridor"]) == 3


def test_point_in_polygon_inside_and_outside():
    polygon = [[0.2, 0.9], [0.8, 0.9], [0.6, 0.2], [0.4, 0.2]]
    assert point_in_polygon((0.5, 0.6), polygon)
    assert not point_in_polygon((0.05, 0.5), polygon)


def test_critical_corridor_hazard_escalates_to_emergency_stop():
    perception = {
        "visibility": "clear",
        "drivable_corridor": [[0.2, 0.95], [0.8, 0.95], [0.65, 0.2], [0.35, 0.2]],
        "hazards": [{
            "type": "pothole", "severity": "critical", "confidence": 0.9,
            "bbox": [0.45, 0.55, 0.55, 0.65], "in_drivable_corridor": False,
        }],
    }
    decision = decide_driving_action(perception)
    assert decision["action"] == "EMERGENCY_STOP"
    assert decision["advisory_only"] is True


def test_unknown_visibility_never_returns_proceed_safe():
    decision = decide_driving_action({
        "visibility": "unknown", "drivable_corridor": [],
        "hazards": [],
    })
    assert decision["action"] == "REDUCE_SPEED"


def test_no_corridor_hazards_with_clear_corridor_is_only_advisory():
    decision = decide_driving_action({
        "visibility": "clear",
        "drivable_corridor": [[0.2, 0.95], [0.8, 0.95], [0.6, 0.2], [0.4, 0.2]],
        "hazards": [],
    })
    assert decision["action"] == "PROCEED_SAFE"
    assert "not a safety guarantee" in decision["reason"]
