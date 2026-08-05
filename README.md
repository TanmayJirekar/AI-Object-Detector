# 🎯 AI Object Detection Platform

A production-ready, three-layer object detection system:

```
Streamlit Frontend  →  Flask REST API  →  YOLO Inference Engine
```

Detects everyday objects (people, furniture, electronics, vehicles,
animals, food, household items, and everything else in the COCO
80-class set) in uploaded images, with adjustable confidence/IoU
thresholds, a live-updating dashboard, detection history, and a
downloadable annotated result.

> **Scope note:** this build focuses on a genuinely complete,
> runnable image-detection product (upload → detect → visualize →
> download, backed by a real REST API). Heavier items from a full
> MLOps spec — custom training loops, Optuna hyperparameter search,
> RTSP/webcam streaming, TensorRT/OpenVINO export, multi-GPU training
> — are intentionally out of scope here so that everything included
> actually runs. The "Extending this project" section at the bottom
> tells you exactly where to plug those in.

---

## 1. Project structure

```
AI-Object-Detector/
├── ai/
│   ├── __init__.py
│   └── detector.py          # YOLO model wrapper (ObjectDetector class)
├── backend/
│   ├── __init__.py
│   ├── app.py                # Flask REST API
│   └── config.py             # Env-var driven configuration
├── frontend/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_detector.py
├── .streamlit/
│   └── config.toml           # Theme + server settings
├── logs/                     # Created automatically at runtime
├── requirements.txt
├── runtime.txt
├── packages.txt              # apt packages needed by OpenCV on Linux hosts
├── Dockerfile
├── docker-entrypoint.sh
└── README.md
```

---

## 2. How it works

- **`ai/detector.py`** loads an Ultralytics YOLO model (`yolov8n.pt` by
  default — small, fast, downloads automatically on first use) and
  exposes `ObjectDetector.predict(image)`, returning structured
  detections (class, confidence, bounding box) plus an annotated copy
  of the image.
- **`backend/app.py`** is a Flask REST API around that detector:
  - `GET  /health` — liveness probe
  - `GET  /model-info` — model name, device, classes, thresholds
  - `GET  /metrics` — request count & average inference time
  - `POST /detect-image` — multipart image upload → JSON detections
    (+ optional base64 annotated image)
- **`frontend/app.py`** is the Streamlit dashboard. It first checks
  whether the Flask backend is reachable at `BACKEND_URL`
  (`http://localhost:5000` by default):
  - **Backend reachable** → every detection is sent over HTTP to the
    Flask API (true 3-tier setup).
  - **Backend not reachable** → the dashboard transparently falls
    back to running the same `ObjectDetector` in-process. This means
    **you can deploy the Streamlit app on its own** (e.g. on
    Streamlit Community Cloud, where you typically only get one
    service) and it still works end to end.

---

## 3. Run it locally (step by step)

### Step 1 — Prerequisites
- Python 3.11 (recommended; 3.10–3.12 also work)
- ~2 GB free disk space (PyTorch + model weights)
- Internet access on first run (to `pip install` and to download the
  YOLO weights file, `yolov8n.pt`, automatically)

### Step 2 — Get the code
Unzip the project and open a terminal in the `AI-Object-Detector/`
folder.

### Step 3 — Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 4 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This installs PyTorch, Ultralytics YOLO, OpenCV, Flask, Streamlit, etc.
(On a machine with an NVIDIA GPU + CUDA installed, PyTorch will
automatically use the GPU; otherwise it runs on CPU.)

### Step 5 — Run the backend (Terminal 1)
```bash
python backend/app.py
```
You should see Flask start on `http://0.0.0.0:5000`. The first time
you hit an endpoint that needs the model, Ultralytics will download
`yolov8n.pt` automatically (~6 MB).

Verify it's alive:
```bash
curl http://localhost:5000/health
```

### Step 6 — Run the frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```
This opens `http://localhost:8501` in your browser. The sidebar shows
a green **"Backend connected"** badge once it detects the Flask API
from Step 5.

### Step 7 — Use it
1. Go to **🖼️ Detect** in the sidebar.
2. Upload a JPG/PNG/BMP/WEBP image.
3. Adjust confidence / IoU sliders if you like.
4. Click **Run Detection**.
5. View bounding boxes, per-object confidence bars, and download the
   annotated result.
6. Check **📊 Model Info** and **🕓 History** for details on the model
   and your past detections.

> Only running Streamlit (skipping Step 5)? That's fine — the app
> will just say "Backend offline · using local engine" and run
> detection in-process instead.

---

## 4. Run with Docker

```bash
docker build -t ai-object-detector .
docker run -p 8501:8501 -p 5000:5000 ai-object-detector
```
Then open `http://localhost:8501`. The container starts the Flask
backend and the Streamlit frontend together (see
`docker-entrypoint.sh`).

---

## 5. Deploy to Streamlit Community Cloud

Streamlit Community Cloud runs a single Python service, so deploy the
**frontend only** — it will automatically fall back to local
in-process inference (see section 2 above), no backend hosting
required.

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repository: your repo. Branch: `main`. Main file path:
   `frontend/app.py`.
4. Deploy. Streamlit Cloud will read `requirements.txt`,
   `packages.txt` (for `libgl1`, needed by OpenCV), and
   `.streamlit/config.toml` automatically.
5. First load will be slower while PyTorch/Ultralytics install and the
   model weights download — subsequent loads are fast.

If you *do* want the full 2-service architecture in the cloud, deploy
`backend/app.py` separately (e.g. Render, Railway, Fly.io, a small VM,
or Hugging Face Spaces with a Docker SDK) and set the `BACKEND_URL`
environment variable on the Streamlit Cloud app to point at it, e.g.:
```
BACKEND_URL = https://your-backend-host.example.com
```
(Add this under your Streamlit Cloud app's **Settings → Secrets** or
**Environment variables**.)

---

## 6. Deploy to Hugging Face Spaces

Use the **Docker** SDK option and push this repo as-is — the included
`Dockerfile` exposes both `8501` and `5000`; set the Space's "App
port" to `8501`.

---

## 7. Configuration (environment variables)

| Variable         | Default              | Description                              |
|------------------|-----------------------|-------------------------------------------|
| `MODEL_NAME`      | `yolov8n.pt`           | Any Ultralytics YOLO checkpoint name/path |
| `CONF_THRESHOLD`  | `0.25`                 | Default confidence threshold              |
| `IOU_THRESHOLD`   | `0.45`                 | Default IoU (NMS) threshold               |
| `DEVICE`          | auto                    | `cpu`, `cuda:0`, etc.                     |
| `BACKEND_HOST`    | `0.0.0.0`              | Flask bind host                           |
| `BACKEND_PORT`    | `5000`                 | Flask port                                |
| `BACKEND_URL`     | `http://localhost:5000`| URL the Streamlit app calls               |
| `MAX_UPLOAD_MB`   | `20`                   | Max image upload size                     |

For a larger, more accurate (but slower) model, set e.g.
`MODEL_NAME=yolov8m.pt` or `yolov8l.pt` — Ultralytics downloads it
automatically the first time it's used.

---

## 8. Testing

```bash
pip install pytest
pytest tests/ -v
```
The included tests cover the detector's data structures and helper
methods without requiring a network connection. A skipped
integration test (`test_real_inference_smoke`) shows how to test real
inference once `requirements.txt` is fully installed with internet
access.

---

## 9. Extending this project

This codebase is intentionally modular so you can grow it toward the
full MLOps spec:

- **Custom training** — add a `ai/training/train.py` that calls
  `YOLO("yolov8n.pt").train(data="your_data.yaml", epochs=..., ...)`;
  Ultralytics handles mosaic/mixup/copy-paste/HSV augmentation via the
  `data.yaml` / training-argument config already.
- **Hyperparameter search** — wrap the same `.train()` call in an
  [Optuna](https://optuna.org) objective function.
- **Video / webcam / RTSP** — `ObjectDetector.predict()` already
  accepts numpy frames, so you can loop `cv2.VideoCapture(source)` and
  call `predict()` per frame; add a `/detect-video` Flask route that
  streams results, and a Streamlit `st.camera_input` or
  `streamlit-webrtc` component on the frontend.
- **Export** — `YOLO(model_name).export(format="onnx")` (also
  supports `torchscript`, `engine` for TensorRT, `openvino`).
- **Evaluation reports** — `YOLO(model_name).val(data="data.yaml")`
  returns precision/recall/mAP50/mAP50-95 and writes PR curves,
  confusion matrices, and loss curves to disk automatically.

---

## 10. Troubleshooting

- **"Could not load the model" on Model Info page** — no internet
  access to download `yolov8n.pt`, or `ultralytics`/`torch` aren't
  installed. Run `pip install -r requirements.txt` with internet
  access, or manually place a `yolov8n.pt` file in the project root
  and set `MODEL_NAME=yolov8n.pt`.
- **OpenCV import error on Linux servers** — install the packages in
  `packages.txt` (`apt-get install libgl1 libglib2.0-0`); already
  handled for you in the Dockerfile and on Streamlit Cloud.
- **Slow first request** — the model is loaded lazily on first use;
  subsequent requests reuse the loaded model (a process-wide
  singleton) and are much faster.
