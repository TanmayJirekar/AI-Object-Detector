"""
ai/detector.py

Core AI inference engine for the Object Detection System.
Wraps an Ultralytics YOLO model and exposes a simple, reusable
predict() API consumed by both the Flask backend and the Streamlit
frontend (the frontend can also run fully offline without the backend).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("ai.detector")

# Model is downloaded automatically by Ultralytics on first use if not
# already present locally (requires internet access on first run).
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


@dataclass
class Detection:
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float
    box: list[float] = field(default_factory=list)  # [x1, y1, x2, y2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 2) for v in self.box],
        }


@dataclass
class DetectionResult:
    """Full result of a single inference call."""

    detections: list[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "detection_count": len(self.detections),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
        }


class ObjectDetector:
    """
    Thin, lazy-loading wrapper around an Ultralytics YOLO model.

    Usage:
        detector = ObjectDetector()
        result, annotated_image = detector.predict(pil_image)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device  # None -> Ultralytics auto-selects cuda/cpu
        self._model = None  # loaded lazily on first predict() call

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLO  # imported here so the module can be
        # imported (e.g. for tests) even in environments without the
        # ultralytics package installed at import time.

        logger.info("Loading YOLO model '%s' ...", self.model_name)
        start = time.time()
        self._model = YOLO(self.model_name)
        if self.device:
            self._model.to(self.device)
        logger.info("Model loaded in %.2fs", time.time() - start)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        self._load_model()
        return self._model.names  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> tuple[DetectionResult, Image.Image]:
        """
        Run inference on a single image.

        Args:
            image: PIL Image, numpy array (RGB), or a path to an image file.
            conf_threshold: overrides the instance default if provided.
            iou_threshold: overrides the instance default if provided.

        Returns:
            (DetectionResult, annotated_pil_image)
        """
        self._load_model()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        pil_image = self._to_pil(image)

        start = time.time()
        results = self._model.predict(  # type: ignore[union-attr]
            source=np.array(pil_image),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        elapsed_ms = (time.time() - start) * 1000

        result = results[0]
        detections: list[Detection] = []
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        box=xyxy,
                    )
                )

        annotated_bgr = result.plot()  # numpy array, BGR, with boxes drawn
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_rgb)

        detection_result = DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_width=pil_image.width,
            image_height=pil_image.height,
            model_name=self.model_name,
        )
        return detection_result, annotated_image

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def bytes_to_pil(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def get_model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model_name": self.model_name,
            "loaded": self.is_loaded,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device or "auto",
        }
        if self.is_loaded:
            info["num_classes"] = len(self._model.names)  # type: ignore[union-attr]
            info["class_names"] = list(self._model.names.values())  # type: ignore[union-attr]
        return info


# A module-level singleton so both the Flask backend and any script that
# imports this module directly reuse the same loaded model instead of
# reloading weights on every call.
_default_detector: ObjectDetector | None = None


def get_detector(**kwargs: Any) -> ObjectDetector:
    """Return a process-wide singleton ObjectDetector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ObjectDetector(**kwargs)
    return _default_detector
