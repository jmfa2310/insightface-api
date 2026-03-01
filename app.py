import os
import threading
import time
import base64
import binascii
import io
import re
from typing import Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

import gradio as gr
import uvicorn

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "buffalo_s"
DET_SIZE: Tuple[int, int] = (320, 320)
MAX_SIDE = 640
MAX_IMAGE_BYTES = 4_000_000
THRESHOLD = 0.60

DATA_URI_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

# -----------------------------
# FastAPI
# -----------------------------
api = FastAPI(title="InsightFace API", version="1.0.0")

_faceapp = None
_face_lock = threading.Lock()

def get_faceapp():
    global _faceapp
    if _faceapp is None:
        with _face_lock:
            if _faceapp is None:
                from insightface.app import FaceAnalysis
                fa = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
                fa.prepare(ctx_id=-1, det_size=DET_SIZE)
                _faceapp = fa
                print(f"[startup] loaded {MODEL_NAME} det_size={DET_SIZE}")
    return _faceapp

@api.on_event("startup")
def startup():
    get_faceapp()

def _decode_b64(b64: str) -> bytes:
    if not b64 or not b64.strip():
        raise HTTPException(status_code=422, detail="img base64 vazio")
    b64 = b64.strip()
    b64 = DATA_URI_RE.sub("", b64).replace("\n", "").replace("\r", "")
    try:
        raw = base64.b64decode(b64, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=422, detail=f"base64 inválido: {e}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"imagem grande demais ({len(raw)} bytes)")
    return raw

def b64_to_np(b64: str) -> np.ndarray:
    raw = _decode_b64(b64)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="conteúdo não é imagem válida")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img, dtype=np.uint8)

class CompareReq(BaseModel):
    img1: str = Field(..., description="base64 img1 (pode ter data:image/...;base64,)")
    img2: str = Field(..., description="base64 img2 (pode ter data:image/...;base64,)")

@api.get("/")
def root():
    return {"ok": True}

@api.get("/warmup")
def warmup():
    get_faceapp()
    return {"ok": True, "model": MODEL_NAME, "det_size": DET_SIZE}

@api.post("/compare")
def compare(req: CompareReq):
    fa = get_faceapp()
    img1 = b64_to_np(req.img1)
    img2 = b64_to_np(req.img2)

    faces1 = fa.get(img1)
    faces2 = fa.get(img2)

    if not faces1:
        return {"ok": False, "error": "NO_FACE_1"}
    if not faces2:
        return {"ok": False, "error": "NO_FACE_2"}

    f1 = max(faces1, key=lambda f: float(getattr(f, "det_score", 0.0)))
    f2 = max(faces2, key=lambda f: float(getattr(f, "det_score", 0.0)))

    dist = float(np.linalg.norm(f1.embedding - f2.embedding))
    return {"ok": True, "match": dist < THRESHOLD, "distance": dist, "threshold": THRESHOLD}

def run_api():
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="info")

# -----------------------------
# Gradio UI (só informativa)
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# InsightFace API online ✅")
    gr.Markdown("Endpoints:\n- `GET /warmup`\n- `POST /compare`\n- `GET /docs`")

# Inicia a API em background e depois sobe a UI
threading.Thread(target=run_api, daemon=True).start()
time.sleep(1)
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
