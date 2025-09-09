import os, io, math
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
THRESH = float(os.getenv("THRESH", "0.5"))

app = FastAPI(title="Document Quality Gate")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()
id2label = model.config.id2label  # {0:"GOOD", 1:"BAD"}

def reason_heuristics(img: Image.Image) -> str:
    # grayscale
    g = img.convert("L")
    # brightness
    mean_brightness = ImageStat.Stat(g).mean[0]
    # blur via Laplacian-of-Gaussian approximated by variance after edge enhance
    edges = g.filter(ImageFilter.FIND_EDGES)
    blur_score = np.array(edges, dtype=np.float32).var()
    if mean_brightness < 60:    # dark
        return "low-light"
    if blur_score < 80:         # weak edges
        return "blur/motion"
    return "compression/skew"

@app.get("/")
def root():
    return {"status": "ok", "labels": list(id2label.values())}

@app.post("/predict")
@torch.no_grad()
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    bad_prob = float(probs[1])  # index 1 == BAD
    verdict = "BAD" if bad_prob >= THRESH else "GOOD"
    resp = {"verdict": verdict, "prob_bad": round(bad_prob, 4)}
    if verdict == "BAD":
        resp["reason"] = reason_heuristics(img)
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
