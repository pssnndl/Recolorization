from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import base64
import io
import json
import torch

from infer import load_model, recolor_image


MODEL_PATH = "checkpoint/checkpoint_epoch_90.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
model = load_model(MODEL_PATH, DEVICE)


class RecolorRequest(BaseModel):
    image_base64: str
    palette: list[list[int]]


@app.post("/recolor")
def recolor(req: RecolorRequest):
    if len(req.palette) != 6:
        raise HTTPException(400, "Palette must contain 6 colors")

    try:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    output = recolor_image(
        model=model,
        image=image,
        palette_rgb=req.palette,
        device=DEVICE
    )

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
