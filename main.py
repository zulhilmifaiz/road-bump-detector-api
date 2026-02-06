import base64
from io import BytesIO

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("model.pt")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...), conf: float = 0.25):
    img_bytes = await image.read()
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")

    results = model.predict(pil, conf=conf, verbose=False)
    r0 = results[0]

    plotted = r0.plot()              # numpy BGR
    plotted_rgb = plotted[..., ::-1]  # BGR -> RGB
    out = Image.fromarray(plotted_rgb.astype(np.uint8))

    buf = BytesIO()
    out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image_base64_png": b64}
