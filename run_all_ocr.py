#!/usr/bin/env python3
# run_all_ocr.py
# Combined OCR runner with simple ground-truth column.
# Column order: image, truth, <solution:variant...>
# For each image <stem>.<ext>, looks for <stem>.txt in the same folder.

import os, csv, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules import easyocr_module as easy_mod
from modules import paddleocr_module as paddle_mod
from modules import pytesseract_module as tess_mod
from modules import openai_vision_module as vis_mod

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def _run_solution(name, folder, opts):
    if name == "easyocr":
        return easy_mod.run_batch(
            folder,
            langs=tuple(opts.get("langs_easy", ["en"])),
            gpu=bool(opts.get("easy_gpu", True)),
            allow=opts.get("easy_allow", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        )
    elif name == "paddleocr":
        return paddle_mod.run_batch(folder, lang=opts.get("lang_paddle", "en"))
    elif name == "pytesseract":
        return tess_mod.run_batch(folder, lang=opts.get("lang_tess", "eng"))
    elif name == "openai_vision":
        return vis_mod.run_batch(
            folder,
            model=opts.get("openai_model", "gpt-4o-mini"),
            ext=opts.get("openai_ext", ".jpg"),
            jpg_quality=int(opts.get("openai_q", 85)),
            limit=int(opts.get("limit", 0)),
        )
    else:
        raise ValueError(f"Unknown solution: {name}")

def _read_truth(folder, image_basename):
    """Read <stem>.txt for <stem>.<ext> image."""
    stem = os.path.splitext(image_basename)[0]
    txt_path = os.path.join(folder, f"{stem}.txt")
    if not os.path.exists(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        try:
            with open(txt_path, "r", encoding="latin-1") as f:
                return f.read().strip()
        except Exception:
            return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="plates", help="Input folder with images and .txt ground truths")
    ap.add_argument("--out", default="csv/combined_ocr.csv", help="Output CSV path")

    ap.add_argument("--enable", nargs="+",
                    default=["easyocr", "paddleocr", "pytesseract", "openai_vision"],
                    choices=["easyocr", "paddleocr", "pytesseract", "openai_vision"])

    # Per-solution options
    ap.add_argument("--langs_easy", nargs="+", default=["en"])
    ap.add_argument("--easy_gpu", type=int, default=1)
    ap.add_argument("--easy_allow", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    ap.add_argument("--lang_paddle", default="en")
    ap.add_argument("--lang_tess", default="eng")

    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--openai_ext", default=".jpg", choices=[".jpg", ".png"])
    ap.add_argument("--openai_q", type=int, default=85)

    # Global
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_workers", type=int, default=4)
    args = ap.parse_args()

    folder = args.inp
    enable = args.enable

    opts = {
        "langs_easy": args.langs_easy,
        "easy_gpu": args.easy_gpu,
        "easy_allow": args.easy_allow,
        "lang_paddle": args.lang_paddle,
        "lang_tess": args.lang_tess,
        "openai_model": args.openai_model,
        "openai_ext": args.openai_ext,
        "openai_q": args.openai_q,
        "limit": args.limit,
    }

    # Run OCR modules concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for name in enable:
            futures[ex.submit(_run_solution, name, folder, opts)] = name

        gathered = []
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                solution, variants, results = fut.result()
                gathered.append((solution, variants, results))
                print(f"[OK] {solution} -> {len(variants)} variants, {len(results)} images")
            except Exception as e:
                print(f"[ERR] {name}: {e}")

    if not gathered:
        print("No results to write.")
        return

    # Collect all image basenames
    all_images = sorted([
        n for n in os.listdir(folder)
        if os.path.splitext(n)[1].lower() in IMG_EXTS
    ])

    # Header: image, truth, then OCR columns
    header = ["image", "truth"]
    for solution, variants, _ in gathered:
        header.extend([f"{solution}:{v}" for v in variants])

    # Write CSV
    out_path = args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for img in all_images:
            truth = _read_truth(folder, img)
            row = [img, truth]
            for solution, variants, results in gathered:
                vals = results.get(img)
                row.extend(vals if vals is not None else [""] * len(variants))
            w.writerow(row)

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
