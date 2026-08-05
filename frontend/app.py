"""
frontend/app.py

Streamlit dashboard for the AI Object Detection System.

Design notes:
- If a Flask backend is reachable at BACKEND_URL, the app calls it over
  REST for inference (true 3-layer architecture).
- If not (e.g. a simple single-service deploy on Streamlit Community
  Cloud where only this app is running), it falls back to running the
  detector in-process. This keeps the app deployable with zero extra
  configuration while still supporting the full client/server setup
  when both services are run (see README.md).

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
BACKEND_TIMEOUT = 2  # seconds, used only for the initial health probe

st.set_page_config(
    page_title="AI Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- #
# Custom styling
# --------------------------------------------------------------------- #

def inject_css(dark_mode: bool) -> None:
    if dark_mode:
        bg, card, text, accent, subtext = "#0e1117", "#161b22", "#f0f2f6", "#6c5ce7", "#9aa4b2"
    else:
        bg, card, text, accent, subtext = "#f7f8fa", "#ffffff", "#1a1a1a", "#6c5ce7", "#5a6472"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .metric-card {{
            background-color: {card};
            border-radius: 14px;
            padding: 18px 20px;
            border: 1px solid rgba(128,128,128,0.15);
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }}
        .metric-card h3 {{ margin: 0; font-size: 26px; color: {accent}; }}
        .metric-card p {{ margin: 2px 0 0 0; color: {subtext}; font-size: 13px; }}
        .app-title {{
            font-size: 40px; font-weight: 800; color: {text};
            margin-bottom: 0px;
        }}
        .app-subtitle {{ color: {subtext}; font-size: 16px; margin-top: 0; }}
        .badge-ok {{
            background: #1fbf75; color: white; padding: 3px 10px;
            border-radius: 999px; font-size: 12px; font-weight: 600;
        }}
        .badge-off {{
            background: #e0555e; color: white; padding: 3px 10px;
            border-radius: 999px; font-size: 12px; font-weight: 600;
        }}
        .det-row {{
            background-color: {card}; border-radius: 10px; padding: 10px 14px;
            margin-bottom: 6px; border: 1px solid rgba(128,128,128,0.15);
        }}
        section[data-testid="stSidebar"] {{ background-color: {card}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {time, filename, count, ms}
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# --------------------------------------------------------------------- #
# Backend / local detector helpers
# --------------------------------------------------------------------- #
@st.cache_data(ttl=5, show_spinner=False)
def check_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=BACKEND_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner=False)
def get_local_detector():
    from ai.detector import get_detector

    return get_detector()


def run_detection_via_backend(image_bytes: bytes, filename: str, conf: float, iou: float):
    files = {"image": (filename, image_bytes)}
    data = {"conf_threshold": conf, "iou_threshold": iou, "return_image": "true"}
    r = requests.post(f"{BACKEND_URL}/detect-image", files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

    import base64

    annotated = None
    if "annotated_image_base64" in payload:
        annotated = Image.open(io.BytesIO(base64.b64decode(payload["annotated_image_base64"])))
    return payload, annotated


def run_detection_locally(pil_image: Image.Image, conf: float, iou: float):
    detector = get_local_detector()
    result, annotated = detector.predict(pil_image, conf_threshold=conf, iou_threshold=iou)
    return result.to_dict(), annotated


def get_model_info_display() -> dict:
    if st.session_state.get("backend_up"):
        try:
            r = requests.get(f"{BACKEND_URL}/model-info", timeout=BACKEND_TIMEOUT)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
    try:
        detector = get_local_detector()
        return detector.get_model_info()
    except Exception as e:  # pragma: no cover - depends on env
        return {"error": str(e)}


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🎯 AI Object Detector")
    page = st.radio(
        "Navigate",
        ["🖼️ Detect", "📊 Model Info", "🕓 History", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

    st.markdown("### Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold", 0.05, 1.0, 0.45, 0.05)

    st.markdown("---")
    backend_up = check_backend()
    st.session_state["backend_up"] = backend_up
    status_html = (
        '<span class="badge-ok">Backend connected</span>'
        if backend_up
        else '<span class="badge-off">Backend offline · using local engine</span>'
    )

inject_css(st.session_state.dark_mode)

with st.sidebar:
    st.markdown(status_html, unsafe_allow_html=True)
    st.caption(f"Backend URL: `{BACKEND_URL}`")


# --------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------- #
st.markdown('<p class="app-title">AI Object Detection Platform</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Upload an image and detect real-world objects '
    "in real time using a YOLO-based deep learning model.</p>",
    unsafe_allow_html=True,
)
st.write("")


# --------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------- #
def render_detect_page() -> None:
    uploaded = st.file_uploader(
        "Drag and drop an image, or click to browse",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded is None:
        st.info("Upload an image to run detection. JPG, PNG, BMP and WEBP are supported.")
        return

    image_bytes = uploaded.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original")
        st.image(pil_image, use_container_width=True)

    run = st.button("🚀 Run Detection", type="primary", use_container_width=True)

    if not run:
        return

    with st.spinner("Running inference..."):
        t0 = time.time()
        try:
            if st.session_state.get("backend_up"):
                payload, annotated = run_detection_via_backend(
                    image_bytes, uploaded.name, conf_threshold, iou_threshold
                )
            else:
                payload, annotated = run_detection_locally(
                    pil_image, conf_threshold, iou_threshold
                )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return
        wall_ms = (time.time() - t0) * 1000

    with col2:
        st.markdown("#### Detected Objects")
        if annotated is not None:
            st.image(annotated, use_container_width=True)

    detections = payload.get("detections", [])
    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [
        (m1, "Objects Found", len(detections)),
        (m2, "Inference Time", f'{payload.get("inference_time_ms", 0):.1f} ms'),
        (m3, "Round-trip Time", f"{wall_ms:.1f} ms"),
        (m4, "Model", payload.get("model_name", "n/a")),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Detections")
    if not detections:
        st.warning("No objects detected above the current confidence threshold.")
    else:
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            bar_pct = int(d["confidence"] * 100)
            st.markdown(
                f"""
                <div class="det-row">
                    <b>{d['class_name']}</b> — {d['confidence']*100:.1f}% confidence
                    <div style="background:rgba(128,128,128,0.25); border-radius:6px; height:8px; margin-top:6px;">
                        <div style="width:{bar_pct}%; background:#6c5ce7; height:8px; border-radius:6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if annotated is not None:
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"detected_{uploaded.name}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.session_state.history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded.name,
            "count": len(detections),
            "ms": round(payload.get("inference_time_ms", 0), 1),
        }
    )


def render_model_info_page() -> None:
    st.markdown("#### Model Information")
    info = get_model_info_display()
    if "error" in info:
        st.error(
            "Could not load the model. This usually means the required "
            "packages (ultralytics/torch) aren't installed yet or there's "
            f"no internet access to download weights.\n\nDetail: {info['error']}"
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", info.get("model_name", "n/a"))
    c2.metric("Device", info.get("device", "n/a"))
    c3.metric("Classes", info.get("num_classes", "n/a"))

    st.markdown("#### Current Thresholds")
    st.write(f"Confidence: `{info.get('conf_threshold')}` · IoU: `{info.get('iou_threshold')}`")

    class_names = info.get("class_names")
    if class_names:
        st.markdown("#### Supported Classes")
        st.write(", ".join(sorted(class_names)))

    if st.session_state.get("backend_up"):
        with st.expander("Backend metrics"):
            try:
                r = requests.get(f"{BACKEND_URL}/metrics", timeout=BACKEND_TIMEOUT)
                st.json(r.json())
            except requests.RequestException as e:
                st.warning(f"Could not fetch metrics: {e}")


def render_history_page() -> None:
    st.markdown("#### Recent Detections")
    if not st.session_state.history:
        st.info("No detections yet. Run a detection from the Detect page.")
        return
    st.table(list(reversed(st.session_state.history)))
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


def render_about_page() -> None:
    st.markdown(
        """
        #### About this platform

        This dashboard is the frontend layer of a three-tier AI Object
        Detection System:

        1. **Frontend** — this Streamlit dashboard
        2. **Backend** — a Flask REST API (`backend/app.py`)
        3. **AI Engine** — a YOLO model wrapped by `ai/detector.py`

        When the Flask backend is reachable, all inference requests are
        sent over HTTP to `/detect-image`. When it isn't (for example, a
        single-service deployment where only the Streamlit app is
        running), the app transparently falls back to running the model
        in-process, so the same codebase works in both setups.

        See `README.md` in the project root for local, Docker, and
        Streamlit Community Cloud deployment instructions.
        """
    )


pages = {
    "🖼️ Detect": render_detect_page,
    "📊 Model Info": render_model_info_page,
    "🕓 History": render_history_page,
    "ℹ️ About": render_about_page,
}
pages[page]()
