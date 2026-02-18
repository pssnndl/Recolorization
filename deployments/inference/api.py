from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import os
from PIL import Image
import base64
import io
import json
import torch
import psutil
import logging
from infer import load_model, recolor_image


MODEL_PATH = "checkpoint/checkpoint_epoch_90.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://recolorization.vercel.app",
        # "http://localhost:5173",
        # "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(MODEL_PATH, DEVICE)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory-logger")
process = psutil.Process(os.getpid())


def log_memory(stage: str):
    mem = process.memory_info().rss / (1024 * 1024)
    vmem = process.memory_info().vms / (1024 * 1024)
    sys_mem = psutil.virtual_memory()

    log = {
        "stage": stage,
        "rss_mb": round(mem, 2),
        "vms_mb": round(vmem, 2),
        "available_mb": round(sys_mem.available / (1024 * 1024), 2)
    }

    if torch.cuda.is_available():
        log["gpu_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024 * 1024), 2
        )

    logger.info(log)
    return mem

class RecolorRequest(BaseModel):
    image_base64: str
    palette: list[list[int]]


@app.get("/health")
def health():
    port = os.environ.get("PORT")
    process = psutil.Process(os.getpid())

    mem_info = process.memory_info()
    
    memory_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size
    memory_vms_mb = mem_info.vms / (1024 * 1024)
    return {
        "status": "ok",
        "port_set": port is not None,
        "port_numeric": port.isdigit() if port is not None else False,
        "port": port,
        "memory_usage_mb": round(memory_mb, 2),
        "virtual_memory_mb": round(memory_vms_mb, 2)
    }

@app.post("/recolor")
def recolor(req: RecolorRequest):

    start_mem = log_memory("start")

    if len(req.palette) != 6:
        raise HTTPException(400, "Palette must contain 6 colors")

    try:
        image_bytes = base64.b64decode(req.image_base64)
        decode_mem = log_memory("after_base64_decode")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_mem = log_memory("after_pil_load")

    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    output = recolor_image(
        model=model,
        image=image,
        palette_rgb=req.palette,
        device=DEVICE
    )

    infer_mem = log_memory("after_inference")

    buf = io.BytesIO()
    output.save(buf, format="PNG")

    serialize_mem = log_memory("after_serialization")

    logger.info({
        "memory_delta_infer_mb": round(infer_mem - pil_mem, 2),
        "memory_delta_total_mb": round(serialize_mem - start_mem, 2)
    })

    return Response(buf.getvalue(), media_type="image/png")
