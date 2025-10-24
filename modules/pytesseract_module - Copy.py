# pytesseract_module.py
# pip install pytesseract opencv-python numpy
# sudo apt install tesseract-ocr
import os, glob, numpy as np, cv2, pytesseract

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
    gimg = to_gray(img)
    inv = 1.0 / max(1e-6, g)
    lut = ((np.arange(256)/255.0)**inv * 255.0).astype(np.uint8)
    return cv2.LUT(ensure_u8(gimg), lut)

def clahe(img, clip=2.0, tile=(8,8)):
    g = to_gray(img)
    cla = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return cla.apply(g)

def otsu(img):
    g = to_gray(img)
    _, th = cv2.threshold(ensure_u8(g), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def adapt_mean(img):
    g = to_gray(img)
    return cv2.adaptiveThreshold(ensure_u8(g), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)

def adapt_gauss(img):
    g = to_gray(img)
    return cv2.adaptiveThreshold(ensure_u8(g), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

def contrast(img, alpha=1.8, beta=0):
    g = to_gray(img)
    return cv2.convertScaleAbs(g, alpha=alpha, beta=beta)

def build_preprocesses():
    return [
        ("original",        lambda img: img),
        ("gray",            to_gray),
        ("clahe",           clahe),
        ("otsu",            otsu),
        ("adapt_mean",      adapt_mean),
        ("adapt_gauss",     adapt_gauss),
        ("gamma_0.8",       lambda img: gamma(img, 0.8)),
        ("gamma_1.2",       lambda img: gamma(img, 1.2)),
        ("contrast_up",     lambda img: contrast(img, 1.8, 0)),
        ("bright_up",       lambda img: contrast(img, 1.0, 30)),
        ("upscale_2x",      lambda img: to_gray(upscale(img, 2.0))),
    ]

def _run_tesseract(img, lang="eng"):
    try:
        text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        return text.strip().replace("\n", " ")
    except Exception as e:
        return f"[error:{type(e).__name__}]"

def run_batch(folder, lang="eng"):
    solution = "pytesseract"
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    paths = sorted([p for e in exts for p in glob.glob(os.path.join(folder, e))])
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
            row.append(_run_tesseract(proc, lang=lang))
        results[base] = row
    return solution, variants, results
