# app.py
import base64
import io
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="InsightFace API (CPU)")

faceapp = None


# ---------- Helpers ----------
def _strip_data_url(b64: str) -> str:
    # aceita "data:image/jpeg;base64,...." ou só "...."
    if not b64:
        return ""
    b64 = b64.strip()
    if "," in b64 and b64.lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def b64_to_rgb_np(b64: str) -> np.ndarray:
    b64 = _strip_data_url(b64)
    if not b64:
        raise ValueError("EMPTY_BASE64")

    try:
        img_bytes = base64.b64decode(b64, validate=False)
    except Exception:
        # alguns base64 vêm com espaços/linhas; tenta mais permissivo
        img_bytes = base64.b64decode(b64 + "===")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def get_faceapp():
    global faceapp
    if faceapp is None:
        # Import aqui para carregar só quando necessário
        from insightface.app import FaceAnalysis

        # CPU-only
        faceapp = FaceAnalysis(name="buffalo_l")
        # No Render (CPU), isto é o mais estável
        faceapp.prepare(ctx_id=0)
    return faceapp


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    sim = float(np.dot(a, b))
    return float(1.0 - sim)


# ---------- Models ----------
class CompareReq(BaseModel):
    Img1: str
    Img2: str


class WarmupRes(BaseModel):
    Ok: bool
    Info: str = ""


class CompareRes(BaseModel):
    Ok: bool
    Match: bool = False
    Distance: float | None = None
    Error: str = ""


# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "insightface-cpu", "version": "1.0"}


@app.post("/warmup", response_model=WarmupRes)
def warmup():
    try:
        _ = get_faceapp()
        return WarmupRes(Ok=True, Info="model loaded")
    except Exception as e:
        return WarmupRes(Ok=False, Info=str(e))


@app.post("/compare", response_model=CompareRes)
def compare(req: CompareReq):
    try:
        fa = get_faceapp()

        img1 = b64_to_rgb_np(req.Img1)
        img2 = b64_to_rgb_np(req.Img2)

        faces1 = fa.get(img1)
        faces2 = fa.get(img2)

        if not faces1:
            return CompareRes(Ok=False, Error="NO_FACE_IMG1")
        if not faces2:
            return CompareRes(Ok=False, Error="NO_FACE_IMG2")

        # usa a face com maior área (mais provável ser a principal)
        f1 = max(faces1, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        f2 = max(faces2, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        emb1 = f1.normed_embedding
        emb2 = f2.normed_embedding

        dist = cosine_distance(emb1, emb2)

        # threshold inicial (ajustas depois)
        # quanto menor, mais parecido. Começa com 0.35~0.45 para cosine-distance.
        threshold = float(os.getenv("THRESHOLD", "0.40"))
        is_match = dist <= threshold

        return CompareRes(Ok=True, Match=is_match, Distance=dist)

    except Exception as e:
        return CompareRes(Ok=False, Error=str(e))
