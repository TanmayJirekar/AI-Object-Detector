"""
tests/test_detector.py

Basic unit tests. Tests that require the actual YOLO weights (i.e. that
call detector.predict / _load_model) are skipped automatically if the
`ultralytics` package or model weights can't be loaded in the current
environment (e.g. no internet access), so this suite is safe to run in
CI environments with restricted network access too.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ai.detector import Detection, DetectionResult, ObjectDetector, get_detector


def test_detection_to_dict():
    d = Detection(class_id=0, class_name="person", confidence=0.9123, box=[1.0, 2.0, 3.0, 4.0])
    result = d.to_dict()
    assert result["class_name"] == "person"
    assert result["confidence"] == 0.9123
    assert result["box"] == [1.0, 2.0, 3.0, 4.0]


def test_detection_result_to_dict():
    d = Detection(class_id=1, class_name="car", confidence=0.5, box=[0, 0, 10, 10])
    result = DetectionResult(
        detections=[d],
        inference_time_ms=12.345,
        image_width=640,
        image_height=480,
        model_name="yolov8n.pt",
    )
    payload = result.to_dict()
    assert payload["detection_count"] == 1
    assert payload["inference_time_ms"] == 12.35
    assert payload["image_width"] == 640


def test_singleton_detector():
    d1 = get_detector()
    d2 = get_detector()
    assert d1 is d2


def test_to_pil_from_numpy():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    pil_img = ObjectDetector._to_pil(arr)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (10, 10)


def test_to_pil_invalid_type():
    with pytest.raises(TypeError):
        ObjectDetector._to_pil(12345)


@pytest.mark.skipif(
    True,
    reason=(
        "Requires downloading real YOLO weights and torch/ultralytics "
        "installed; run manually with `pytest -m integration --no-skip` "
        "after `pip install -r requirements.txt` and with internet access."
    ),
)
def test_real_inference_smoke():
    detector = ObjectDetector()
    img = Image.new("RGB", (640, 480), color=(120, 120, 120))
    result, annotated = detector.predict(img)
    assert result.image_width == 640
    assert isinstance(annotated, Image.Image)
