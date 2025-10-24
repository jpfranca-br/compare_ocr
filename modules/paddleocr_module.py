# paddleocr_module.py
# pip install "paddleocr>=2.0.1" "paddlepaddle"  # or paddlepaddle-gpu
import os, glob, numpy as np, cv2
from paddleocr import PaddleOCR
import re

# ---- helpers (from your updated v3 script) ----
def ensure_u8(img):
    if img is None: return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

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
    _, th = cv2.threshold(ensure_u8(gray), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def thresh_adapt_mean(gray, b=31, C=5):
    return cv2.adaptiveThreshold(ensure_u8(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b|1, C)

def thresh_adapt_gauss(gray, b=31, C=5):
    return cv2.adaptiveThreshold(ensure_u8(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b|1, C)

def denoise(gray):
    return cv2.fastNlMeansDenoising(ensure_u8(gray), h=15, templateWindowSize=7, searchWindowSize=21)

def bilateral(gray):
    return cv2.bilateralFilter(ensure_u8(gray), d=7, sigmaColor=50, sigmaSpace=50)

def median(gray, k=3): return cv2.medianBlur(ensure_u8(gray), k)
def gaussian(gray, k=3):
    k = k | 1
    return cv2.GaussianBlur(ensure_u8(gray), (k,k), 0)

def morph_open(gray, k=3):
    k = k | 1
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(ensure_u8(gray), cv2.MORPH_OPEN, kernel)

def morph_close(gray, k=3):
    k = k | 1
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(ensure_u8(gray), cv2.MORPH_CLOSE, kernel)

def unsharp(gray):
    blur = cv2.GaussianBlur(ensure_u8(gray), (0,0), 1.0)
    return cv2.addWeighted(ensure_u8(gray), 1.5, blur, -0.5, 0)

def contrast_brightness(gray, alpha=1.8, beta=0):
    return cv2.convertScaleAbs(ensure_u8(gray), alpha=alpha, beta=beta)

def build_preprocesses():
    return [
        ("original",           lambda img: img),
        ("gray",               lambda img: to_gray(img)),
        ("gray_clahe",         lambda img: clahe_gray(img)),
        ("gray_bilateral",     lambda img: bilateral(to_gray(img))),
        ("gray_median3",       lambda img: median(to_gray(img), 3)),
        ("gray_gauss3",        lambda img: gaussian(to_gray(img), 3)),
        ("unsharp",            lambda img: unsharp(to_gray(img))),
        ("thresh_otsu",        lambda img: thresh_otsu(to_gray(img))),
        ("thresh_adapt_mean",  lambda img: thresh_adapt_mean(to_gray(img), 31, 5)),
        ("thresh_adapt_gauss", lambda img: thresh_adapt_gauss(to_gray(img), 31, 5)),
        ("morph_open",         lambda img: morph_open(to_gray(img), 3)),
        ("morph_close",        lambda img: morph_close(to_gray(img), 3)),
        ("gamma_0.8",          lambda img: gamma(img, 0.8)),
        ("gamma_1.2",          lambda img: gamma(img, 1.2)),
        ("contrast_up",        lambda img: contrast_brightness(to_gray(img), 1.8, 0)),
        ("bright_up",          lambda img: contrast_brightness(to_gray(img), 1.0, 30)),
        ("denoise",            lambda img: denoise(to_gray(img))),
        ("upscale_2x",         lambda img: to_gray(upscale(img, 2.0))),
        ("upscale_3x",         lambda img: to_gray(upscale(img, 3.0))),
    ]

def _clean_text(text):
    """Keep only letters and numbers, remove everything else."""
    if not text:
        return ""
    # Remove everything except uppercase/lowercase letters and digits
    clean = re.sub(r'[^A-Za-z0-9]', '', text)
    return clean.strip()

def _run_paddle(reader, img):
    try:
        import numpy as np, cv2
        if img is None or (isinstance(img, np.ndarray) and img.size == 0):
            return "[error:empty_image]"
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        res = reader.predict(img)
        texts = []
        for item in res:
            if isinstance(item, dict) and "rec_texts" in item:
                texts.extend(item["rec_texts"])
        return _clean_text(" ".join(texts).strip())
    except Exception as e:
        return f"[error:{type(e).__name__}]"

def run_batch(folder, lang="en"):
    solution = "paddleocr"
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    paths = sorted([p for e in exts for p in glob.glob(os.path.join(folder, e))])

    preprocesses = build_preprocesses()
    variants = [name for name,_ in preprocesses]
    results = {}

    if not paths:
        return solution, variants, results

    reader = PaddleOCR(lang=lang)

    for p in paths:
        base = os.path.basename(p)
        img = load_image(p)
        if img is None:
            results[base] = ["[error:load_failed]"] * len(preprocesses)
            continue
        row = []
        for _, fn in preprocesses:
            proc = ensure_u8(fn(img))
            row.append(_run_paddle(reader, proc))
        results[base] = row

    return solution, variants, results
