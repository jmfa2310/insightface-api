import base64
import io
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="InsightFace API")

faceapp = None

def get_faceapp():
    global faceapp
    if faceapp is None:
        from insightface.app import FaceAnalysis
        faceapp = FaceAnalysis(name="buffalo_l")
        faceapp.prepare(ctx_id=-1)
    return faceapp


def b64_to_img(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


class CompareReq(BaseModel):
    img1: str
    img2: str


@app.get("/")
def root():
    return {"ok": True}


@app.post("/compare")
def compare(req: CompareReq):
    appface = get_faceapp()

    img1 = b64_to_img(req.img1)
    img2 = b64_to_img(req.img2)

    faces1 = appface.get(img1)
    faces2 = appface.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"ok": False, "error": "Face not detected"}

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    dist = np.linalg.norm(emb1 - emb2)
    match = dist < 1.2

    return {
        "ok": True,
        "match": match,
        "distance": float(dist)
    }
