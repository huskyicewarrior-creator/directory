"""Stage 2+3: assign margin-name lines to roster rows and faces to lines.

Inputs:
  static/face_index.json      (previous run's OCR margin_lines per page - reused, no re-OCR)
  cache/faces_raw.json        (YuNet face boxes per page, from extract_face_boxes.py)
  roster_rows.json            (336 roster rows: id/first/last/teacher)

Name assignment (line -> roster row): scored global assignment (scipy
linear_sum_assignment) in three passes:
  A. threshold 0.75 over plausible name lines
  B. threshold 0.55 over remaining plausible lines
  C. single-token fragment lines (OCR-truncated names) that prefix-match or
     fuzzy-match exactly one unclaimed roster token
Bottom-of-page caption zone (activity photos below the portrait grid) is
excluded from names by the same band geometry used for faces.

Face assignment (face -> line): portrait grid bands cluster face y centers;
within a band, faces sorted by x pair with the band's name lines sorted by y
(yearbook reading order). Mismatch counts leave extras unassigned rather than
wrong. Bands separated from the portrait grid by a large vertical gap are
activity-photo groups and are dropped.

Output: static/faces/*.jpg crops + static/face_index.json (name/id/files)
"""
import glob
import json
import os
import re

import cv2
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
from scipy.optimize import linear_sum_assignment

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STOPWORDS = {"and", "are", "in", "is", "their", "out", "that", "this", "with",
             "pour", "pours", "mixture", "project", "town", "class", "art",
             "finished", "activity", "wait", "for", "brownie", "team",
             "sam", "are", "just", "his", "her", "ms", "cte"}


def norm_text(s):
    s = (s or "").upper()
    s = re.sub(r"[^A-Z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_name_line(text):
    toks = [t for t in re.split(r"[^A-Za-z']+", text) if t]
    if not toks or len(toks) > 4:
        return False
    core = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS:
            return False
        # a real name line's tokens are alphabetic, >=2 chars; fragments/captions
        # of prose tend to include words <=2 chars after junk stripping
        if not t.isalpha() and not all(c.isalpha() or c == "'" for c in t):
            return False
        core.append(t)
    letters = sum(len(t) for t in core)
    if letters < 4 or not (4 <= letters <= 26):
        return False
    return len(core) >= 2


def dl(a, b, cap=4):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        prev2, prev = prev, cur
    return prev[-1]


def pair_score(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.85
    if len(a) >= 3 and len(b) > len(a) and a == b[:len(a)] and len(b) - len(a) <= 4:
        return 0.72  # OCR-truncated name fragment
    if abs(len(a) - len(b)) > max(len(a), len(b)) / 2:
        return 0.0
    d = dl(a, b, cap=max(len(a), len(b)))
    score = 1.0 - d / max(len(a), len(b))
    return score if score >= 0.5 else 0.0


def name_score(text, first, last):
    t = [w for w in norm_text(text).split() if w]
    fn = [w for w in norm_text(first).split() if w]
    ln = [w for w in norm_text(last).split() if w]
    if not t or (not fn and not ln):
        return 0.0
    rost = fn + ln
    best = 0.0
    for order in (rost, ln + fn):
        full = "".join(order)
        ocr = "".join(t)
        if full == ocr:
            s = 1.0
        elif full and (full in ocr or ocr in full) and min(len(full), len(ocr)) >= 6:
            s = 0.9
        elif abs(len(full) - len(ocr)) > max(len(full), len(ocr)) / 2:
            s = 0.0
        else:
            s = max(0.0, 1.0 - dl(full, ocr, cap=max(len(full), len(ocr)))
                    / max(len(full), len(ocr), 1))
        best = max(best, s)
    tok_scores = [max((pair_score(tok, w) for w in t), default=0.0) for tok in rost]
    # geometric mean: a wholly unmatched roster token voids the token path
    tok_geo = 0.0
    if tok_scores and all(s > 0 for s in tok_scores):
        tok_geo = 1.0
        for s in tok_scores:
            tok_geo *= s
        tok_geo = tok_geo ** (1.0 / len(tok_scores)) * 0.95
    all_matched = all(s >= 0.4 for s in tok_scores)
    return max(best if all_matched else 0.0, tok_geo)


def fragment_candidates(text, roster):
    t = [w for w in norm_text(text).split() if w]
    if len(t) != 1 or len(t[0]) < 4:
        return []
    frag = t[0]
    hits = []
    for r in roster:
        toks = [w for w in norm_text(f"{r.get('first', '')} {r.get('last', '')}").split() if w]
        ok = False
        for w in toks:
            if pair_score(w, frag) >= 0.7:
                ok = True
                break
            if len(w) >= 4 and len(frag) < len(w) and dl(frag, w[:len(frag)], cap=2) <= 1:
                ok = True
                break
        if ok:
            hits.append(r)
    return hits


def build_scores(lines_all, roster):
    M = np.zeros((len(lines_all), len(roster)))
    for i, (p, l) in enumerate(lines_all):
        if not is_name_line(l["text"]):
            continue
        for j, r in enumerate(roster):
            M[i, j] = name_score(l["text"], r.get("first", ""), r.get("last", ""))
    return M


def assign_pass(M, allowed, thresh, claimed_lines, used_rows, roster, lines_all):
    rows = []
    for i, j in zip(*np.where(M >= thresh)):
        if not allowed[i]:
            continue
        rows.append((i, j))
    if not rows:
        return {}
    idx_map = {i: k for k, (i, _) in enumerate(rows)}
    idx_cols = {k: j for k, (_, j) in enumerate(rows)}
    C = -np.array([M[i, j] for i, j in rows]).reshape(1, -1) if False else None
    # build rectangular cost for optimal assignment restricted to allowed pairs
    n = len(rows)
    big = 10.0
    lin = -M[rows[:, 0] if False else [i for i, _ in rows], [j for _, j in rows]]
    rr, cc = linear_sum_assignment(lin)
    out = {}
    for k1, k2 in zip(rr, cc):
        i, j = rows[k1], rows[k2]
        if i in claimed_lines or j in used_rows:
            continue
        out[i] = j
    return out


def main():
    register_heif_opener()
    roster = json.load(open(os.path.join(BASE, "roster_rows.json"), encoding="utf-8"))
    old = json.load(open(os.path.join(BASE, "static", "face_index.json"), encoding="utf-8"))
    raw_boxes = json.load(open(os.path.join(BASE, "cache", "faces_raw.json"), encoding="utf-8"))

    lines_all = []
    for page in sorted(old):
        for l in old[page]["margin_lines"]:
            lines_all.append((page, l))
    print(f"names: {len(lines_all)} lines, roster: {len(roster)} rows")

    M = build_scores(lines_all, roster)
    plausible = [bool(is_name_line(l["text"])) for _, l in lines_all]
    print("plausible name lines:", sum(plausible), "/", len(plausible))

    assignment = {}
    claimed_lines = set()
    used_rows = set()
    for thresh in (0.75, 0.55):
        pidx = [i for i in range(M.shape[0]) if plausible[i] and i not in claimed_lines]
        if not pidx:
            continue
        sub = M[pidx].copy()
        sub[sub < thresh] = -1.0
        rr, cc = linear_sum_assignment(-sub)
        for r, c in zip(rr, cc):
            if sub[r, c] < thresh:
                continue
            i, j = pidx[r], int(c)
            if j in used_rows:
                continue
            assignment[i] = j
            claimed_lines.add(i)
            used_rows.add(j)
        print(f"    pass {thresh}: kept {len(assignment)} total, used rows {len(used_rows)}")
    used_ids = set(roster[j]["id"] for j in used_rows)
    # pass C: fragments
    for i, (p, l) in enumerate(lines_all):
        if i in claimed_lines or not l["text"]:
            continue
        hits = fragment_candidates(l["text"], roster)
        unclaimed = [r for r in hits if r["id"] not in used_ids]
        if len(unclaimed) == 1:
            j = roster.index(unclaimed[0])
            assignment[i] = j
            claimed_lines.add(i)
            used_rows.add(j)
            used_ids.add(unclaimed[0]["id"])
    # pass D: mutual-best reconciliation among leftovers. Yearbook margin names
    # print in alphabetical order and leftover roster rows are surname-sorted,
    # but plain gap-filling over-aligns, so only accept a pair that is the
    # best remaining option for BOTH its line and its roster row.
    changed = True
    while changed:
        changed = False
        free_lines = [i for i in range(len(lines_all))
                      if plausible[i] and i not in claimed_lines]
        free_rows = [j for j in range(len(roster)) if j not in used_rows]
        if not free_lines or not free_rows:
            break
        best_by_line = {}
        best_by_row = {}
        for i in free_lines:
            for j in free_rows:
                s = name_score(lines_all[i][1]["text"], roster[j].get("first", ""),
                               roster[j].get("last", ""))
                if s >= 0.62:
                    if i not in best_by_line or s > best_by_line[i][0]:
                        best_by_line[i] = (s, j)
                    if j not in best_by_row or s > best_by_row[j][0]:
                        best_by_row[j] = (s, i)
        for i in free_lines:
            if i not in best_by_line or i in claimed_lines:
                continue
            s, j = best_by_line[i]
            if j in used_rows:
                continue
            if best_by_row.get(j, (None, None))[1] == i:
                assignment[i] = j
                claimed_lines.add(i)
                used_rows.add(j)
                print(f"    mutual-best {s:.2f} [{lines_all[i][0]}] "
                      f"'{lines_all[i][1]['text']}' -> {roster[j]['first']} "
                      f"{roster[j]['last']}")
                changed = True
                break

    used_ids = set(roster[j]["id"] for j in used_rows)

    unmatched_lines = [lines_all[i][1]["text"] for i in range(len(lines_all)) if i not in claimed_lines]
    unmatched_ids = [r for r in roster if r["id"] not in used_ids]
    final_scores = [(M[i][j], lines_all[i][0], lines_all[i][1]["text"],
                     roster[j]["first"] + " " + roster[j]["last"])
                    for i, j in assignment.items()]
    low = [x for x in final_scores if x[0] < 0.7]
    print(f"assigned lines: {len(assignment)}; roster covered: {len(used_ids)}/{len(roster)}")
    print("LOW-CONF (<0.70):")
    for s, p, txt, name in sorted(low):
        print(f"   {s:.2f} {p} '{txt}' -> {name}")
    print("UNMATCHED LINES:", unmatched_lines)
    print("UNMATCHED ROSTER:")
    for r in unmatched_ids:
        print(f"   {r['first']} {r['last']} ({r['teacher']})")

    # Stage 3: faces -> lines, per page
    line_owner = {lines_all[i][0]: {} for i in range(len(lines_all))}
    for i, j in assignment.items():
        page, l = lines_all[i]
        line_owner[page][id(l)] = roster[j]

    crops_dir = os.path.join(BASE, "static", "faces")
    os.makedirs(crops_dir, exist_ok=True)
    face_index = {}
    for page in sorted(old):
        llist = old[page]["margin_lines"]
        boxes = raw_boxes.get(page, [])
        if not boxes:
            continue
        ys = sorted(b["y"] for b in boxes)
        bands = []
        band = [ys[0]]
        for y in ys[1:]:
            if y - band[-1] < 300:
                band.append(y)
            else:
                bands.append(sum(band) / len(band))
                band = [y]
        bands.append(sum(band) / len(band))
        # drop activity-photo groups: bands separated from the previous band by
        # a gap > 420 AND holding fewer than 5 faces, or beyond the last dense
        # portrait band
        keep_bands = []
        for bi in range(len(bands)):
            prev_gap = bands[bi] - bands[bi - 1] if bi else 0
            cnt = sum(1 for b in boxes if abs(b["y"] + b["h"] / 2 - bands[bi]) < 220)
            if bi and prev_gap > 420 and cnt < 4:
                continue
            keep_bands.append(bands[bi])
        portrait_boxes = [b for b in boxes
                          if any(abs(b["y"] + b["h"] / 2 - kb) < 220 for kb in keep_bands)]
        bands_kept = [bi for bi in range(len(bands)) if bi < len(keep_bands) and abs(bands[bi] - keep_bands[bi if bi < len(keep_bands) else -1]) < 0.01]

        def band_id(y):
            for k, kb in enumerate(keep_bands):
                if abs(y - kb) < 220:
                    return k
            return None

        names_by_band = {}
        for l in llist:
            k = band_id(l["y"])
            if k is None:
                continue
            names_by_band.setdefault(k, []).append(l)
        faces_by_band = {}
        for b in portrait_boxes:
            faces_by_band.setdefault(band_id(b["y"] + b["h"] / 2), []).append(b)

        img_path = os.path.join(BASE, "images", page + ".HEIC")
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(img_path)), cv2.COLOR_RGB2BGR)

        assigned_ids = {}
        faces_out = []
        for k in sorted(faces_by_band):
            bf = sorted(faces_by_band[k], key=lambda b: b["x"])
            bn = sorted(names_by_band.get(k, []), key=lambda l: l["y"])
            n = min(len(bf), len(bn))
            for b, l in zip(bf[:n], bn[:n]):
                r = line_owner[page].get(id(l))
                if r is None or r["id"] in assigned_ids:
                    continue
                lm = next((c["landmarks"] for c in raw_boxes[page]
                           if c["x"] == b["x"] and c["y"] == b["y"] and c["w"] == b["w"]), None)
                assigned_ids[r["id"]] = True
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                pad_top, pad_bot = int(h * 0.30), int(h * 0.95)
                pad_lr = int(w * 0.18)
                x0, y0 = max(0, x - pad_lr), max(0, y - pad_top)
                x1, y1 = min(W_page(img), x + w + pad_lr), min(img.shape[0], y + h + pad_bot)
                crop = img[y0:y1, x0:x1]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if crop_pil.width > 260:
                    crop_pil.thumbnail((260, 400))
                safe = re.sub(r"[^A-Za-z0-9_]", "", r["id"]) or "unk"
                fname = f"{page}_{safe}.jpg"
                crop_pil.save(os.path.join(crops_dir, fname), quality=86)
                faces_out.append({
                    "page": page,
                    "file": fname,
                    "id": r["id"],
                    "first": r.get("first"),
                    "last": r.get("last"),
                    "ocr_line": l["text"],
                    "det_score": round(b["score"], 3),
                    "band": k,
                    "col": bf.index(b),
                    "box": {"x": x, "y": y, "w": w, "h": h},
                    "landmarks": lm,
                })
        face_index[page] = {"faces": faces_out, "margin_lines": llist}
        print(f"{page}: {len(boxes)} raw / {len(portrait_boxes)} portrait / {len(faces_out)} matched / {len(llist)} names")

    with open(os.path.join(BASE, "static", "face_index.json"), "w", encoding="utf-8") as f:
        json.dump(face_index, f, ensure_ascii=False)
    total = sum(len(v["faces"]) for v in face_index.values())
    ids = set(f["id"] for v in face_index.values() for f in v["faces"])
    print(f"TOTAL: {total} crops, {len(ids)} unique students")


def W_page(img):
    return img.shape[1]


if __name__ == "__main__":
    main()
