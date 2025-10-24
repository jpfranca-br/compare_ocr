#!/usr/bin/env python3
from ultralytics import YOLO
import torch, cv2, time, os, json

rtsp_url   = "video/traffic2.mp4"
model_path = "models/license.pt"          # your trained plate model
output_dir = "plates"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model = YOLO(model_path)

TOP_N = 1         # keep the 5 largest crops per ID (by pixel area)
SAVE_DELAY = 20.0  # seconds to wait from first sighting
IMG_EXT = ".png"  # low CPU to write; change to ".png" or ".jpg" if you prefer

tracks = {}  # id -> {"t0": time, "best": [(area, crop)], "records": []}

for result in model.track(source=rtsp_url, stream=True, device=device, persist=True, verbose=False):
    frame = result.orig_img
    if frame is None:
        continue
    h, w = frame.shape[:2]
    now = time.time()

    if not result.boxes or result.boxes.id is None:
        continue

    # --- ingest current frame detections ---
    ids   = result.boxes.id.cpu().numpy()
    xyxy  = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for i, tid in enumerate(ids):
        x1, y1, x2, y2 = map(int, xyxy[i])
        # clip to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        crop = frame[y1:y2, x1:x2].copy()

        if tid not in tracks:
            tracks[tid] = {"t0": now, "best": [], "records": []}
            print(f"[NEW] id {int(tid)}")

        t = tracks[tid]
        t["records"].append({"ts": now, "bbox": [x1, y1, x2, y2], "area": int(area), "conf": float(confs[i])})

        # keep only the TOP_N largest crops in memory
        t["best"].append((area, crop))
        t["best"] = sorted(t["best"], key=lambda x: -x[0])[:TOP_N]

    # --- save when ID reached SAVE_DELAY, then CLEANUP memory ---
    for tid, t in list(tracks.items()):
        if "saved" not in t and now - t["t0"] > SAVE_DELAY:
            print(f"[SAVE] id {int(tid)}")

            # save crops (ranked by size)
            for j, (a, crop) in enumerate(t["best"], 1):
                path = os.path.join(output_dir, f"{int(tid)}{IMG_EXT}")
                #path = os.path.join(output_dir, f"{int(tid)}_{j}{IMG_EXT}")
                cv2.imwrite(path, crop)
                print(f"  ↳ wrote {path} (area={a})")

            # save JSON (the YOLO snapshots we kept)
            #jpath = os.path.join(output_dir, f"{int(tid)}.json")
            #with open(jpath, "w", encoding="utf-8") as f:
            #    json.dump(t["records"], f, indent=2)
            #print(f"  ↳ wrote {jpath}")

            t["saved"] = True

            # ---- CLEANUP: free memory for this ID ----
            # Drop references to crops and records, then remove the whole entry.
            # (Python's GC will free the arrays.)
            del t["best"]
            del t["records"]
            del tracks[tid]
