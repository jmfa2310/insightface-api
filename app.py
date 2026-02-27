import base64, io
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
    fa = get_faceapp()

    img1 = b64_to_img(req.img1)
    img2 = b64_to_img(req.img2)

    f1 = fa.get(img1)
    f2 = fa.get(img2)

    if not f1:
        return {"ok": False, "error": "NO_FACE_1"}
    if not f2:
        return {"ok": False, "error": "NO_FACE_2"}

    e1 = f1[0].embedding
    e2 = f2[0].embedding

    dist = float(np.linalg.norm(e1 - e2))
    match = dist < 0.6

    return {"ok": True, "match": match, "distance": dist}