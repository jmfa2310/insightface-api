import base64
import binascii
import io
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="InsightFace API")

# Remove prefixo tipo: data:image/jpeg;base64,
DATA_URI_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")

faceapp = None


def get_faceapp():
    global faceapp
    if faceapp is None:
        from insightface.app import FaceAnalysis

        # Força CPU (importante em serviços tipo Render)
        faceapp = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        faceapp.prepare(ctx_id=-1)
    return faceapp


def clean_b64(s: str) -> str:
    s = s.strip()
    s = DATA_URI_RE.sub("", s)
    s = s.replace("\n", "").replace("\r", "")
    return s


def b64_to_img(b64: str) -> np.ndarray:
    b64 = clean_b64(b64)

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except binascii.Error as e:
        raise HTTPException(status_code=422, detail=f"Base64 inválido: {e}")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Bytes decodificados não são uma imagem válida")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha ao abrir imagem: {e}")

    return np.array(img)


class CompareReq(BaseModel):
    img1: str
    img2: str


@app.on_event("startup")
def startup():
    # Carrega o modelo no boot (se der erro, aparece no log do Render)
    try:
        get_faceapp()
    except Exception as e:
        print("Erro ao inicializar InsightFace:", repr(e))
        raise


@app.get("/")
def root():
    return {"ok": True}


@app.post("/compare")
def compare(req: CompareReq):
    fa = get_faceapp()

    img1 = b64_to_img(req.img1)
    img2 = b64_to_img(req.img2)

    try:
        f1 = fa.get(img1)
        f2 = fa.get(img2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha no FaceAnalysis: {repr(e)}")

    if not f1:
        return {"ok": False, "error": "NO_FACE_1"}
    if not f2:
        return {"ok": False, "error": "NO_FACE_2"}

    # Pega o rosto com maior score de detecção
    best1 = max(f1, key=lambda x: float(getattr(x, "det_score", 0.0)))
    best2 = max(f2, key=lambda x: float(getattr(x, "det_score", 0.0)))

    e1 = best1.embedding
    e2 = best2.embedding

    dist = float(np.linalg.norm(e1 - e2))
    match = dist < 0.6  # ajuste se quiser

    return {"ok": True, "match": match, "distance": dist}
