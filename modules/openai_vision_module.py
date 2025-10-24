# openai_vision_module.py
# pip install openai opencv-python numpy
# export OPENAI_API_KEY=...
import os, glob, base64, time
import numpy as np
import cv2
from openai import OpenAI
import re

# ------------- helpers (from your original) -------------
def ensure_u8(img):
    if img is None: return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def upscale(img, scale=2.0):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def gamma(img, g=1.0):
    base = img if img.ndim == 2 else to_gray(img)
    inv = 1.0 / max(1e-6, g)
    lut = ((np.arange(256)/255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(ensure_u8(base), lut)

def clahe_gray(img, clip=2.0, tile=(8,8)):
    g = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(g)

def thresh_otsu(gray):
    _, th = cv2.threshold(ensure_u8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def thresh_adapt_mean(gray, b=31, C=5):
    return cv2.adaptiveThreshold(ensure_u8(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, b|1, C)

def unsharp(gray):
    blur = cv2.GaussianBlur(ensure_u8(gray), (0,0), 1.0)
    return cv2.addWeighted(ensure_u8(gray), 1.5, blur, -0.5, 0)

def contrast_brightness(gray, alpha=1.8, beta=0):
    return cv2.convertScaleAbs(ensure_u8(gray), alpha=alpha, beta=beta)

def build_preprocesses():
    return [
#        ("original",           lambda img: img),
#        ("gray",               lambda img: to_gray(img)),
#        ("gray_clahe",         lambda img: clahe_gray(img)),
#        ("unsharp",            lambda img: unsharp(to_gray(img))),
#        ("thresh_otsu",        lambda img: thresh_otsu(to_gray(img))),
#        ("thresh_adapt_mean",  lambda img: thresh_adapt_mean(to_gray(img), 31, 5)),
#        ("gamma_1.2",          lambda img: gamma(img, 1.2)),
#        ("contrast_up",        lambda img: contrast_brightness(to_gray(img), 1.8, 0)),
        ("upscale_2x",         lambda img: to_gray(upscale(img, 2.0))),
    ]

def to_bgr(img):
    if img is None: return None
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def im_to_data_url(img, ext=".jpg", jpg_quality=85):
    img = ensure_u8(img)
    if img is None: return None
    img = to_bgr(img)
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if ext.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

def _clean_text(text):
    """Keep only letters and numbers, remove everything else."""
    if not text:
        return ""
    # Remove everything except uppercase/lowercase letters and digits
    clean = re.sub(r'[^A-Za-z0-9]', '', text)
    return clean.strip()

def _call_openai_vision(client, model, data_url, max_retries=3):
    messages = [
        {"role": "system",
         "content": "You are a precise license-plate OCR. Return only the plate in UPPERCASE without spaces or separators. If no plate, return EMPTY."},
        {"role": "user",
         "content": [
            {"type": "text", "text": "Extract the license plate."},
            {"type": "image_url", "image_url": {"url": data_url}}
         ]}
    ]
    delay = 1.0
    for attempt in range(1, max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
            return _clean_text(resp.choices[0].message.content.strip())
        except Exception as e:
            if attempt == max_retries:
                return f"[error:{type(e).__name__}]"
            time.sleep(delay)
            delay *= 2

def run_batch(folder, model="gpt-4o-mini", ext=".jpg", jpg_quality=85, limit=0):
    solution = "openai_vision"
    client = OpenAI()

    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    paths = sorted([p for e in exts for p in glob.glob(os.path.join(folder, e))])
    if limit and limit > 0:
        paths = paths[:limit]

    preprocesses = build_preprocesses()
    variants = [name for name,_ in preprocesses]
    results = {}

    for p in paths:
        base = os.path.basename(p)
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            results[base] = ["[error:load_failed]"] * len(preprocesses)
            continue

        row = []
        for _, fn in preprocesses:
            proc = ensure_u8(fn(img))
            data_url = im_to_data_url(proc, ext=ext, jpg_quality=jpg_quality)
            if not data_url:
                row.append("[error:encode]")
                continue
            row.append(_call_openai_vision(client, model, data_url))
        results[base] = row

    return solution, variants, results
