"""
backend/app.py

Flask REST API for the AI Object Detection System.

Endpoints:
    GET  /health         -> service liveness check
    GET  /model-info      -> details about the loaded model
    GET  /metrics          -> lightweight runtime metrics
    POST /detect-image    -> run detection on an uploaded image

Run directly with:
    python backend/app.py
or via gunicorn / Docker (see Dockerfile).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request

# Allow running this file directly (python backend/app.py) as well as
# as part of the package (gunicorn backend.app:app).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.detector import get_detector  # noqa: E402
from backend.config import Config  # noqa: E402

os.makedirs(Config.LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(Config.LOG_FILE, maxBytes=5_000_000, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("backend.app")

_START_TIME = time.time()
_REQUEST_COUNT = {"detect-image": 0}
_TOTAL_INFERENCE_MS = {"detect-image": 0.0}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    detector = get_detector(
        model_name=Config.MODEL_NAME,
        conf_threshold=Config.CONF_THRESHOLD,
        iou_threshold=Config.IOU_THRESHOLD,
        device=Config.DEVICE,
    )

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "File too large"}), 413

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.exception("Unhandled error: %s", e)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "uptime_seconds": round(time.time() - _START_TIME, 1),
                "model_loaded": detector.is_loaded,
            }
        )

    @app.route("/model-info", methods=["GET"])
    def model_info():
        try:
            return jsonify(detector.get_model_info())
        except Exception as e:  # model failed to load (e.g. no internet)
            logger.exception("Could not load model info")
            return jsonify({"error": str(e)}), 500

    @app.route("/metrics", methods=["GET"])
    def metrics():
        count = _REQUEST_COUNT["detect-image"]
        avg_ms = (_TOTAL_INFERENCE_MS["detect-image"] / count) if count else 0.0
        return jsonify(
            {
                "uptime_seconds": round(time.time() - _START_TIME, 1),
                "total_detect_requests": count,
                "average_inference_time_ms": round(avg_ms, 2),
                "model_loaded": detector.is_loaded,
            }
        )

    @app.route("/detect-image", methods=["POST"])
    def detect_image():
        if "image" not in request.files:
            return jsonify({"error": "No 'image' file part in the request"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not Config.allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": (
                            "Unsupported file type. Allowed: "
                            f"{sorted(Config.ALLOWED_EXTENSIONS)}"
                        )
                    }
                ),
                400,
            )

        conf = request.form.get("conf_threshold", type=float)
        iou = request.form.get("iou_threshold", type=float)
        return_image = request.form.get("return_image", "true").lower() == "true"

        try:
            pil_image = detector.bytes_to_pil(file.read())
        except Exception:
            return jsonify({"error": "Could not read image file"}), 400

        try:
            result, annotated = detector.predict(
                pil_image, conf_threshold=conf, iou_threshold=iou
            )
        except Exception as e:
            logger.exception("Inference failed")
            return jsonify({"error": f"Inference failed: {e}"}), 500

        _REQUEST_COUNT["detect-image"] += 1
        _TOTAL_INFERENCE_MS["detect-image"] += result.inference_time_ms

        payload = result.to_dict()

        if return_image:
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            payload["annotated_image_base64"] = base64.b64encode(buf.getvalue()).decode(
                "utf-8"
            )

        logger.info(
            "detect-image: %d objects, %.1fms",
            payload["detection_count"],
            result.inference_time_ms,
        )
        return jsonify(payload)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
