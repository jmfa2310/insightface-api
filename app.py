import base64
import binascii
import io
import re
import threading
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "buffalo_s"           # leve
DET_SIZE: Tuple[int, int] = (320, 320)  # mais rápido/menos RAM
MAX_SIDE = 640                    # limita dimensão para poupar CPU/RAM
MAX_IMAGE_BYTES = 4_000_000       # ~4MB (anti-abuso)
THRESHOLD = 0.60                  # distância L2 (ajusta se quiser)

DATA_URI_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="InsightFace API", version="1.0.0")

_faceapp = None
_faceapp_lock = threading.Lock()


def get_faceapp():
    """
    Lazy singleton with lock (thread-safe).
    """
    global _faceapp
    if _faceapp is None:
        with _faceapp_lock:
            if _faceapp is None:
                from insightface.app import FaceAnalysis

                fa = FaceAnalysis(
                    name=MODEL_NAME,
                    providers=["CPUExecutionProvider"],
                )
                fa.prepare(ctx_id=-1, det_size=DET_SIZE)
                _faceapp = fa
                print(f"[startup] InsightFace loaded: {MODEL_NAME}, det_size={DET_SIZE}")
    return _faceapp


@app.on_event("startup")
def startup():
    # Carrega o modelo no boot para evitar 1º request lento
    get_faceapp()


def _decode_base64_image(b64: str) -> bytes:
    if not b64 or not b64.strip():
        raise HTTPException(status_code=422, detail="img base64 vazio")

    b64 = b64.strip()
    b64 = DATA_URI_RE.sub("", b64)
    b64 = b64.replace("\n", "").replace("\r", "")

    try:
        raw = base64.b64decode(b64, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=422, detail=f"base64 inválido: {e}")

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"imagem grande demais ({len(raw)} bytes). Limite: {MAX_IMAGE_BYTES} bytes",
        )

    return raw


def b64_to_img_np(b64: str) -> np.ndarray:
    raw = _decode_base64_image(b64)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="conteúdo não é uma imagem válida")

    # resize para poupar CPU/RAM
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))

    return np.asarray(img, dtype=np.uint8)


class CompareReq(BaseModel):
    img1: str = Field(..., description="Base64 da imagem 1 (pode ter data:image/...;base64,)")
    img2: str = Field(..., description="Base64 da imagem 2 (pode ter data:image/...;base64,)")


@app.get("/")
def root():
    return {"ok": True}


@app.get("/warmup")
def warmup():
    get_faceapp()
    return {"ok": True, "model": MODEL_NAME, "det_size": DET_SIZE}


@app.post("/compare")
def compare(req: CompareReq):
    fa = get_faceapp()

    img1 = b64_to_img_np(req.img1)
    img2 = b64_to_img_np(req.img2)

    try:
        faces1 = fa.get(img1)
        faces2 = fa.get(img2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FaceAnalysis falhou: {repr(e)}")

    if not faces1:
        return {"ok": False, "error": "NO_FACE_1"}
    if not faces2:
        return {"ok": False, "error": "NO_FACE_2"}

    # melhor rosto = maior det_score
    f1 = max(faces1, key=lambda f: float(getattr(f, "det_score", 0.0)))
    f2 = max(faces2, key=lambda f: float(getattr(f, "det_score", 0.0)))

    e1 = f1.embedding
    e2 = f2.embedding

    dist = float(np.linalg.norm(e1 - e2))
    match = dist < THRESHOLD

    return {
        "ok": True,
        "match": match,
        "distance": dist,
        "threshold": THRESHOLD,
    }
