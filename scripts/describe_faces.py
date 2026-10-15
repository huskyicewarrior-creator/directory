"""Enrich face_index.json with detectable photo descriptors.

Descriptors are mechanical, not identity judgments: glasses (dark vertical
frame strokes at both eye positions, calibrated on the cohort), hair tone
buckets (top-of-head strip medians), and photo flags. Filters that use them
search what is visible in the yearbook photo, not what a person "looks like"
between photos.

OpenCV 5 note: CascadeClassifier is gone in this build, so detection is
image statistics on the YuNet face box + landmarks.
"""
import json
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GLASSES_MIN_REL = 2.2


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def glasses_components(img, box, lms):
    """(score, relL, relR): peak vertical-stroke energy at both eye rims."""
    pad_lr = int(box["w"] * 0.18)
    pad_top = int(box["h"] * 0.30)
    factor = img.shape[1] / (box["w"] + 2 * pad_lr)
    if not lms or len(lms) < 4:
        return 0.0, 0.0, 0.0
    resh = box["h"] * factor
    resh2 = box["w"] * factor

    def tp(px, py):
        return (px - box["x"] + pad_lr) * factor, (py - box["y"] + pad_top) * factor

    rex, rey = tp(lms[0], lms[1])
    lex, ley = tp(lms[2], lms[3])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yy0 = max(0, int(min(ley, rey) - resh * 0.045))
    yy1 = min(gray.shape[0] - 1, int(max(ley, rey) + resh * 0.045))
    x0 = max(0, int(min(lex, rex) - resh2 * 0.10))
    x1 = min(gray.shape[1], int(max(lex, rex) + resh2 * 0.10))
    if yy1 <= yy0 or x1 <= x0 + 4:
        return 0.0, 0.0, 0.0
    band = gray[yy0:yy1, x0:x1]
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    col = np.abs(gx).mean(axis=0)
    med = float(np.median(col)) + 1e-6

    def rim(cx, lo_off, hi_off):
        a = max(0, int(cx + lo_off))
        b = max(a + 1, min(gray.shape[1] - 1, int(cx + hi_off)))
        if b - a < 2:
            return 0.0
        seg = col[max(0, a - x0): max(1, b - x0)]
        return float(seg.max() / med) if seg.size else 0.0

    xe = sorted([lex, rex])
    relL = rim(xe[0], -resh2 * 0.10, -resh2 * 0.01)
    relR = rim(xe[1], resh2 * 0.01, resh2 * 0.10)
    score = min(relL, relR)
    return score, relL, relR


def hair_info(img, box, lms):
    """Bucket the top-of-head strip color by medians; masked against bg white."""
    factor = img.shape[1] / (box["w"] + 2 * box["w"] * 0.18 * 2 / 2 + (box["w"] * 0.18) * 2) \
        if False else img.shape[1] / (box["w"] + int(box["w"] * 0.36))
    resh = box["h"] * factor
    strip = img[0: max(4, int(resh * 0.32)),
                int(box["w"] * 0.12 * factor): int(box["w"] * 0.88 * factor)]
    if strip.size == 0 or strip.shape[0] < 4:
        return "unknown", 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0].astype(np.float32)
    ss = hsv[:, :, 1].astype(np.float32) / 255.0
    vv = hsv[:, :, 2].astype(np.float32) / 255.0
    m = ~((vv > 0.88) & (ss < 0.10))
    if int(m.sum()) < 30:
        return "unknown", 0.0, 0.0, 0.0
    hm = float(np.median(hh[m]))
    sm = float(np.median(ss[m]))
    vm = float(np.median(vv[m]))
    label = "brown"
    if vm < 0.33:
        label = "black"
    elif sm < 0.12 and vm > 0.55:
        label = "gray"
    elif sm <= 0.55 and vm >= 0.60:
        label = "blonde"
    elif hm <= 10 and sm >= 0.55 and vm >= 0.40:
        label = "red"
    conf = round(clamp(0.35 + 0.5 * sm + 0.15 * (1.0 if vm < 0.45 else 0.0)), 2)
    return label, round(hm, 1), round(sm, 2), conf


def brightness_of(crop_bgr):
    ycc = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    return float(np.mean(ycc[:, :, 0]))


def main():
    idx_path = os.path.join(BASE, "static", "face_index.json")
    face_index = json.load(open(idx_path, encoding="utf-8"))
    n_glasses = 0
    scores = []
    hair_counts = {}
    for page, data in face_index.items():
        for f in data["faces"]:
            path = os.path.join(BASE, "static", "faces", f["file"])
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            desc = {}
            gs, relL, relR = glasses_components(img, f["box"], f.get("landmarks"))
            scores.append(gs)
            desc["glasses"] = bool(gs >= GLASSES_MIN_REL)
            desc["glasses_score"] = round(gs, 2)
            hair, hm, sm, hconf = hair_info(img, f["box"], f.get("landmarks"))
            desc["hair"] = hair
            desc["hair_conf"] = hconf
            desc["brightness"] = round(brightness_of(img), 1)
            f["desc"] = desc
            if desc["glasses"]:
                n_glasses += 1
            hair_counts[hair] = hair_counts.get(hair, 0) + 1
        print(f"{page}: glasses so far {n_glasses}")
    with open(idx_path, "w", encoding="utf-8") as out:
        json.dump(face_index, out, ensure_ascii=False)
    if scores:
        arr = np.array(sorted(scores))
        print(f"TOTAL glasses: {n_glasses}/{len(scores)}; score "
              f"p25={np.percentile(arr, 25):.2f} p50={np.percentile(arr, 50):.2f} "
              f"p75={np.percentile(arr, 75):.2f} p85={np.percentile(arr, 85):.2f} max={arr.max():.2f}")
    print("hair:", hair_counts)


if __name__ == "__main__":
    main()
