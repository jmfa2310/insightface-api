import base64
import binascii
import io
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="InsightFace API")

faceapp = None
DATA_URI_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")


def get_faceapp():
    global faceapp
    if faceapp is None:
        from insightface.app import FaceAnalysis

        # ✅ modelo leve + CPU
        faceapp = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
        )

        # ✅ det_size menor = mais rápido + menos RAM
        faceapp.prepare(ctx_id=-1, det_size=(320, 320))
        print("InsightFace carregado: buffalo_s / CPU (det_size=320)")

    return faceapp


@app.on_event("startup")
def startup():
    # ✅ carrega no boot (evita timeout no primeiro /compare)
    get_faceapp()


def b64_to_img(b64: str) -> np.ndarray:
    if not b64:
        raise HTTPException(status_code=422, detail="img base64 vazio")

    b64 = b64.strip()
    b64 = DATA_URI_RE.sub("", b64)
    b64 = b64.replace("\n", "").replace("\r", "")

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=422, detail=f"base64 inválido: {e}")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="bytes decodificados não são imagem válida")

    # ✅ resize para reduzir RAM/CPU
    max_side = 640
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))

    return np.array(img)


class CompareReq(BaseModel):
    img1: str
    img2: str


@app.get("/")
def root():
    return {"ok": True}


@app.get("/warmup")
def warmup():
    # ✅ garante que o modelo está carregado (já fica pelo startup)
    get_faceapp()
    return {"ok": True}


@app.post("/compare")
def compare(req: CompareReq):
    fa = get_faceapp()

    img1 = b64_to_img(req.img1)
    img2 = b64_to_img(req.img2)

    try:
        faces1 = fa.get(img1)
        faces2 = fa.get(img2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FaceAnalysis falhou: {repr(e)}")

    if not faces1:
        return {"ok": False, "error": "NO_FACE_1"}
    if not faces2:
        return {"ok": False, "error": "NO_FACE_2"}

    # pega o rosto com maior score
    f1 = max(faces1, key=lambda x: float(getattr(x, "det_score", 0.0)))
    f2 = max(faces2, key=lambda x: float(getattr(x, "det_score", 0.0)))

    dist = float(np.linalg.norm(f1.embedding - f2.embedding))
    match = dist < 0.6

    return {"ok": True, "match": match, "distance": dist}
