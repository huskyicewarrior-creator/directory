"""Stage 1: cache per-page face detections (boxes + scores) from images/*.HEIC.

Extra detections (31 net above the roster count) are removed: any box whose IoU
with a higher-scored box exceeds IOU_DROP is treated as a duplicate detection
of the same portrait. Everything else is kept for the pairing stage.

Output: cache/faces_raw.json  {page: [{"x","y","w","h","score"}, ...]}
"""
import glob
import json
import os

import cv2
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET = os.path.join(BASE, "models", "face_detection_yunet_2023mar.onnx")
OUT = os.path.join(BASE, "cache", "faces_raw.json")
IOU_DROP = 0.35
SCORE_MIN = 0.55


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union


def main():
    register_heif_opener()
    os.makedirs(os.path.join(BASE, "cache"), exist_ok=True)
    pages = sorted(glob.glob(os.path.join(BASE, "images", "*.HEIC")))
    out = {}
    for path in pages:
        page = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.5, 0.3, 4000)
        det.setInputSize((W, H))
        ok, arr = det.detect(img)
        raw = [f for f in (arr.tolist() if arr is not None else [])
               for f in [f] if f[14] >= SCORE_MIN]
        raw.sort(key=lambda f: -f[14])
        keep = []
        for f in raw:
            box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14]), f[4:14])
            if all(iou((box[0], box[1], box[2], box[3]), k[:4]) < IOU_DROP for k in keep):
                keep.append(box)
        keep.sort(key=lambda b: (b[1], b[0]))
        out[page] = [{"x": b[0], "y": b[1], "w": b[2], "h": b[3],
                      "score": round(b[4], 3), "landmarks": [round(float(v), 3) for v in b[5]]}
                     for b in keep]
        print(f"{page}: {len(raw)} raw -> {len(keep)} after dedupe")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    total = sum(len(v) for v in out.values())
    print(f"TOTAL kept: {total}")


if __name__ == "__main__":
    main()
