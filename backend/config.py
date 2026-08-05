"""
backend/config.py

Central configuration for the Flask backend, driven by environment
variables so the same code works locally, in Docker, and on any
cloud platform without modification.
"""

from __future__ import annotations

import os


class Config:
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "yolov8n.pt")
    CONF_THRESHOLD: float = float(os.environ.get("CONF_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.environ.get("IOU_THRESHOLD", "0.45"))
    DEVICE: str | None = os.environ.get("DEVICE") or None  # "cpu", "cuda:0", or None (auto)

    MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "bmp", "webp"}

    HOST: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("BACKEND_PORT", "5000"))
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    LOG_DIR: str = os.environ.get("LOG_DIR", "logs")
    LOG_FILE: str = os.path.join(LOG_DIR, "backend.log")

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
        )
