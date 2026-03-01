import base64
import io
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="InsightFace API")

_faceapp = None
_faceapp_last_init_error = None
_faceapp_init_time = None


def _b64_to_rgb_np(b64: str) -> np.ndarray:
    """
    Aceita:
      - base64 puro "iVBORw0KGgo..."
      - data-uri "data:image/jpeg;base64,/9j/..."
    """
    if not isinstance(b64, str) or len(b64) < 10:
        raise ValueError("IMG_BASE64_VAZIO")

    # remove prefix data-uri se existir
    if "," in b64 and b64.strip().lower().startswith("data:image"):
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64, validate=False)
    except Exception:
        raise ValueError("IMG_BASE64_INVALIDO")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise ValueError("IMG_NAO_E_IMAGEM")

    # numpy array RGB
    return np.asarray(img)


def get_faceapp():
    """
    Inicializa o InsightFace uma vez (lazy), e reaproveita.
    """
    global _faceapp, _faceapp_last_init_error, _faceapp_init_time

    if _faceapp is not None:
        return _faceapp

    try:
        from insightface.app import FaceAnalysis

        fa = FaceAnalysis(name="buffalo_l")  # modelo conhecido
        # ctx_id=-1 => CPU (Render free não tem GPU)
        fa.prepare(ctx_id=-1)
        _faceapp = fa
        _faceapp_last_init_error = None
        _faceapp_init_time = time.time()
        return _faceapp
    except Exception as e:
        _faceapp_last_init_error = str(e)
        raise


def _extract_embedding(faceapp, img_rgb: np.ndarray) -> np.ndarray:
    """
    Devolve embedding normalizada (float32).
    """
    faces = faceapp.get(img_rgb)
    if not faces:
        raise ValueError("SEM_FACE")

    # escolhe a face maior
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    emb = best.embedding
    if emb is None:
        raise ValueError("SEM_EMBEDDING")

    emb = np.asarray(emb, dtype=np.float32)

    # normaliza
    n = np.linalg.norm(emb)
    if n == 0:
        raise ValueError("EMBEDDING_ZERO")
    emb = emb / n
    return emb


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings já normalizadas => cosine similarity = dot
    sim = float(np.dot(a, b))
    # distância coseno (0=igual, 2=oposto) -> normalmente fica ~[0..1]
    return 1.0 - sim


class CompareReq(BaseModel):
    img1: str
    img2: str
    # threshold em distância coseno: mais baixo = mais rígido
    # valores típicos: 0.35 a 0.55 (depende do cenário)
    threshold: Optional[float] = 0.45


class CompareResp(BaseModel):
    ok: bool
    match: bool
    distance: Optional[float] = None
    error: str = ""


class EmbedReq(BaseModel):
    img: str


class EmbedResp(BaseModel):
    ok: bool
    embedding_b64: str = ""
    error: str = ""


@app.get("/")
def health():
    return {
        "ok": True,
        "faceapp_loaded": _faceapp is not None,
        "faceapp_init_error": _faceapp_last_init_error,
        "faceapp_init_time": _faceapp_init_time,
    }


@app.post("/compare", response_model=CompareResp)
def compare(req: CompareReq):
    try:
        faceapp = get_faceapp()

        img1 = _b64_to_rgb_np(req.img1)
        img2 = _b64_to_rgb_np(req.img2)

        emb1 = _extract_embedding(faceapp, img1)
        emb2 = _extract_embedding(faceapp, img2)

        dist = _cosine_distance(emb1, emb2)
        is_match = dist <= float(req.threshold)

        return CompareResp(ok=True, match=is_match, distance=dist, error="")
    except ValueError as ve:
        return CompareResp(ok=False, match=False, distance=None, error=str(ve))
    except Exception as e:
        return CompareResp(ok=False, match=False, distance=None, error=f"ERRO_INTERNO: {str(e)}")


@app.post("/embed", response_model=EmbedResp)
def embed(req: EmbedReq):
    """
    Endpoint para guardares "a identidade" do empregado sem guardar foto:
    - Recebe 1 imagem base64
    - Devolve o embedding (vetor) em base64
    """
    try:
        faceapp = get_faceapp()
        img = _b64_to_rgb_np(req.img)
        emb = _extract_embedding(faceapp, img)

        # guarda como bytes float32 e depois base64
        emb_bytes = emb.astype(np.float32).tobytes()
        emb_b64 = base64.b64encode(emb_bytes).decode("ascii")

        return EmbedResp(ok=True, embedding_b64=emb_b64, error="")
    except ValueError as ve:
        return EmbedResp(ok=False, embedding_b64="", error=str(ve))
    except Exception as e:
        return EmbedResp(ok=False, embedding_b64="", error=f"ERRO_INTERNO: {str(e)}")


class CompareEmbReq(BaseModel):
    img: str
    embedding_b64: str
    threshold: Optional[float] = 0.45


class CompareEmbResp(BaseModel):
    ok: bool
    match: bool
    distance: Optional[float] = None
    error: str = ""


@app.post("/compare_embedding", response_model=CompareEmbResp)
def compare_embedding(req: CompareEmbReq):
    """
    Fluxo ideal para ponto:
    - BD guarda embedding do empregado (texto base64)
    - No ponto, tiras selfie e comparas selfie vs embedding guardado
    """
    try:
        faceapp = get_faceapp()
        img = _b64_to_rgb_np(req.img)
        emb_live = _extract_embedding(faceapp, img)

        # decode embedding guardado
        emb_bytes = base64.b64decode(req.embedding_b64)
        emb_db = np.frombuffer(emb_bytes, dtype=np.float32)

        # normaliza o embedding do BD (para segurança)
        n = np.linalg.norm(emb_db)
        if n == 0:
            raise ValueError("EMBEDDING_BD_ZERO")
        emb_db = emb_db / n

        dist = _cosine_distance(emb_live, emb_db)
        is_match = dist <= float(req.threshold)

        return CompareEmbResp(ok=True, match=is_match, distance=dist, error="")
    except ValueError as ve:
        return CompareEmbResp(ok=False, match=False, distance=None, error=str(ve))
    except Exception as e:
        return CompareEmbResp(ok=False, match=False, distance=None, error=f"ERRO_INTERNO: {str(e)}")
