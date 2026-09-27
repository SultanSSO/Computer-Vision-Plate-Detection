"""Fase A.1 — Ekstraksi kandidat crop plat dari log deteksi video uji.

Acuan: docs/PRD_decision_plat.md §4 Fase A langkah 1-2 dan §5 (parameter OCR baseline uji 2).

Yang dilakukan:
  1. Baca log JSONL hasil inferensi (uji 1 & uji 2) — 1 baris per deteksi per frame.
  2. Kelompokkan deteksi menjadi TRACK kendaraan (greedy IoU antar frame) agar crop
     dari kendaraan yang sama tidak dihitung sebagai banyak sampel.
  3. Untuk tiap track, ambil beberapa crop representatif (frame tersebar + confidence
     OCR tertinggi) — preprocessing DISAMAKAN dengan pipeline inferensi
     (crop + padding 10% + grayscale + CLAHE + upscale 3x) supaya recognizer nanti
     belajar dari citra yang identik dengan saat dipakai (PRD Fase C).
  4. Tulis `dataset_ft/labels_draft.csv` (kolom `gt_text` KOSONG — diisi manual)
     + galeri HTML `dataset_ft/labeling/gallery.html` untuk memudahkan pelabelan.

Tidak mengubah apa pun di `dataset/` (Phase 1 dibekukan) dan tidak menghapus file.
Crop yang sudah ada tidak ditimpa (aman dijalankan ulang).

Jalankan:  python scripts/extract_crops.py
"""

from __future__ import annotations

import csv
import html
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field

import cv2

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PADDING = 0.10       # sama dengan pipeline inferensi (PRD_decision_plat.md §5)
UPSCALE = 3          # sama dengan pipeline inferensi (uji 2)
CLAHE_CLIP = 2.0

IOU_TRACK = 0.25     # ambang IoU agar deteksi dianggap kendaraan yang sama
MAX_FRAME_GAP = 15   # jarak frame maksimum antar deteksi dalam satu track (frame-skip 3)
MIN_TRACK_LEN = 3    # track pendek tidak dipakai (perlu ≥3 frame agar ada frame lain sbg konteks)
MAX_PER_TRACK = 6    # maksimum crop per track
MIN_FRAME_SPREAD = 12  # jarak frame minimum antar crop dalam satu track (hindari duplikat)
OCR_CONF_REF_MIN = 0.35  # crop paling tajam wajib ≥ ini (kalau tidak, track dibuang — blur total)
OCR_CONF_MIN = 0.20      # crop pendukung wajib ≥ ini; kalau tidak dan track tanpa referensi bagus, dibuang

OUT_CROPS = os.path.join(PROJECT, "dataset_ft", "crops")
OUT_CSV = os.path.join(PROJECT, "dataset_ft", "labels_draft.csv")
OUT_GALLERY = os.path.join(PROJECT, "dataset_ft", "labeling", "gallery.html")

SOURCES = [
    {
        "tag": "uji_2",
        "video": os.path.join(PROJECT, "test_video", "sample_video_1min_fix.mp4"),
        "log": os.path.join(PROJECT, "logs", "uji_2", "sample_video_1min_fix.mp4_log.jsonl"),
        "min_height": 20,   # bbox lebih pendek dari ini tidak terbaca OCR (MIN_HEIGHT pipeline)
    },
    {
        "tag": "uji_1",
        "video": os.path.join(PROJECT, "test_video", "sample_video_1min.mp4"),
        "log": os.path.join(PROJECT, "output", "uji_1_sample_video_1min",
                            "predictions_txt", "sample_video_1min_log.jsonl"),
        "min_height": 24,   # uji 1 platnya jauh lebih kecil -> ambang dinaikkan
    },
]

# Peta substitusi karakter yang sering tertukar: huruf <-> angka
LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6",
                   "B": "8", "A": "4", "T": "7", "E": "3", "Q": "0", "D": "0"}
DIGIT_TO_LETTER = {v: k for k, v in LETTER_TO_DIGIT.items() if k not in ("L", "Q", "D")}


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def resolve_video(path: str) -> str | None:
    """Cari file video; terima nama tanpa ekstensi maupun dengan .mp4."""
    for cand in (path, path + ".mp4", os.path.splitext(path)[0]):
        if os.path.isfile(cand):
            return cand
    return None


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_text(raw: str) -> str:
    return "".join(ch for ch in str(raw).upper() if ch.isalnum())


def crop_plate(frame, box, pad: float = PADDING, upscale: int = UPSCALE):
    """Crop bbox + padding -> grayscale + CLAHE -> upscale (identik pipeline inferensi)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(gray)
    if upscale and upscale != 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return gray


def suggest_plate(text: str) -> tuple[str, str]:
    """Draft koreksi berbasis posisi karakter (cikal bakal D3, sekadar bantu pelabelan).

    Mencoba beberapa template posisi (huruf/angka) dan memilih yang paling sedikit
    butuh substitusi. Mengembalikan (teks_usulan, nama_template).
    """
    t = normalize_text(text)
    if len(t) < 5:
        return t, "-"

    templates = []
    # Format UK modern: 2 huruf + 2 angka + 3 huruf (video uji berisi plat UK)
    if len(t) == 7:
        templates.append(("UK LLDDLLL", "LLDDLLL"))
    # Format Indonesia: 1-2 huruf + 1-4 angka + 0-3 huruf
    for a in (1, 2):
        for b in range(1, 5):
            for c in range(0, 4):
                if a + b + c == len(t):
                    templates.append((f"ID H{a}A{b}H{c}", "L" * a + "D" * b + "L" * c))

    best = None
    for label, tpl in templates:
        out, cost = [], 0
        for ch, expect in zip(t, tpl):
            if expect == "L" and ch.isdigit():
                ch = DIGIT_TO_LETTER.get(ch, ch); cost += 1
            elif expect == "D" and ch.isalpha():
                ch = LETTER_TO_DIGIT.get(ch, ch); cost += 1
            out.append(ch)
        cand = ("".join(out), label, cost)
        if best is None or (cost, label) < (best[2], best[1]):
            best = cand
    return best[0], best[1]


@dataclass
class Track:
    track_id: int
    tag: str
    dets: list = field(default_factory=list)   # list of dict
    last_box: tuple = None
    last_frame: int = -999

    def add(self, det):
        self.dets.append(det)
        self.last_box = tuple(det["bbox"])
        self.last_frame = det["frame"]


# ----------------------------------------------------------------------------
# 1. Baca log & bentuk track
# ----------------------------------------------------------------------------
def build_tracks(tag: str, log_path: str, min_height: int) -> tuple[list[Track], dict]:
    if not os.path.isfile(log_path):
        print(f"  ! log tidak ada, dilewati: {log_path}")
        return [], {}

    dets, stats = [], Counter()
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total_log"] += 1
            if d.get("skipped_small"):
                stats["skip_flag"] += 1
                continue
            x1, y1, x2, y2 = d["bbox"]
            if (y2 - y1) < min_height:
                stats["skip_terlalu_kecil"] += 1
                continue
            dets.append(d)
    dets.sort(key=lambda d: (d["frame"], d["bbox"][0]))

    active: list[Track] = []
    tracks: list[Track] = []
    for d in dets:
        box = tuple(d["bbox"])
        best, best_iou = None, 0.0
        for tr in active:
            if d["frame"] - tr.last_frame > MAX_FRAME_GAP:
                continue
            score = iou(box, tr.last_box)
            if score > best_iou:
                best, best_iou = tr, score
        if best is not None and best_iou >= IOU_TRACK:
            best.add(d)
        else:
            tr = Track(track_id=len(tracks), tag=tag)
            tr.add(d)
            tracks.append(tr)
            active.append(tr)
        active = [tr for tr in active if d["frame"] - tr.last_frame <= MAX_FRAME_GAP]

    tracks = [t for t in tracks if len(t.dets) >= MIN_TRACK_LEN]
    stats["track"] = len(tracks)
    stats["det_dipakai"] = sum(len(t.dets) for t in tracks)
    return tracks, stats


# ----------------------------------------------------------------------------
# 2. Pilih crop representatif per track
# ----------------------------------------------------------------------------
def pick_representatives(track: Track) -> list[dict]:
    """Pilih crop representatif per track dengan kontrol kualitas blur.

    - Crop 'referensi' = deteksi dgn ocr_conf tertinggi; wajib ≥ OCR_CONF_REF_MIN,
      kalau tidak track dibuang (blur total, tidak akan bisa dilabeli).
    - Crop pendukung (frame tersebar) wajib ≥ OCR_CONF_MIN; jika tidak ada yang
      lolos, pakai 2 deteksi terbaik saja.
    """
    by_conf = sorted(track.dets, key=lambda d: (-d.get("ocr_conf", 0.0), d["frame"]))
    ref = by_conf[0]
    if ref.get("ocr_conf", 0.0) < OCR_CONF_REF_MIN:
        return []  # track blur total

    chosen, frames = [ref], [ref["frame"]]
    for d in by_conf:
        if len(chosen) >= MAX_PER_TRACK:
            break
        if d is ref or any(abs(d["frame"] - f) < MIN_FRAME_SPREAD for f in frames):
            continue
        if d.get("ocr_conf", 0.0) < OCR_CONF_MIN and ref.get("ocr_conf", 0.0) < 0.5:
            continue  # crop pendukung buram & referensi tidak cukup kuat
        chosen.append(d); frames.append(d["frame"])
    return sorted(chosen, key=lambda d: d["frame"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_CROPS, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_GALLERY), exist_ok=True)

    rows, galleries = [], {}
    for src in SOURCES:
        video = resolve_video(src["video"])
        print(f"\n=== {src['tag']} ===")
        if not video:
            print(f"  ! video tidak ditemukan: {src['video']}")
            continue
        tracks, stats = build_tracks(src["tag"], src["log"], src["min_height"])
        print(f"  log: {stats.get('total_log', 0)} baris | "
              f"skip flag: {stats.get('skip_flag', 0)} | "
              f"skip <{src['min_height']}px: {stats.get('skip_terlalu_kecil', 0)} | "
              f"deteksi dipakai: {stats.get('det_dipakai', 0)}")
        print(f"  track terbentuk: {stats.get('track', 0)}")

        if not tracks:
            continue

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"  ! video tidak bisa dibaca: {video}")
            continue

        cached_frame, cached_img = -1, None
        n_crop = 0
        for tr in tracks:
            votes = Counter(normalize_text(d.get("plat", "")) for d in tr.dets
                            if normalize_text(d.get("plat", "")))
            draft = votes.most_common(1)[0][0] if votes else ""
            usul, tpl = suggest_plate(draft) if draft else ("", "-")
            for det in pick_representatives(tr):
                if det["frame"] != cached_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, det["frame"])
                    ok, img = cap.read()
                    cached_frame, cached_img = (det["frame"], img) if ok else (-1, None)
                if cached_img is None:
                    continue
                gray = crop_plate(cached_img, det["bbox"])
                if gray is None or gray.size == 0:
                    continue
                stem = f"{src['tag']}_t{tr.track_id:03d}_f{det['frame']:04d}"
                name = f"{stem}.jpg"
                path = os.path.join(OUT_CROPS, name)
                if not os.path.exists(path):
                    cv2.imwrite(path, gray)
                n_crop += 1
                rows.append({
                    "crop_id": stem,
                    "file": f"crops/{name}",
                    "src": src["tag"],
                    "track_id": tr.track_id,
                    "frame": det["frame"],
                    "timestamp_s": det.get("timestamp_s", ""),
                    "bbox": " ".join(str(int(v)) for v in det["bbox"]),
                    "det_conf": round(float(det.get("det_conf", 0)), 4),
                    "ocr_raw": det.get("plat", ""),
                    "ocr_conf": round(float(det.get("ocr_conf", 0)), 4),
                    "plat_valid": det.get("plat_valid", False),
                    "n_frame_track": len(tr.dets),
                    "n_variasi_ocr": len(votes),
                    "ocr_top": draft,
                    "usul_koreksi": usul,
                    "template": tpl,
                    "gt_text": "",          # <- DIISI MANUAL
                    "catatan": "",
                })
                galleries.setdefault((src["tag"], tr.track_id), {
                    "draft": draft, "usul": usul, "n": len(tr.dets), "items": []
                })["items"].append((name, det))
        cap.release()

        n_track_used = len({r["track_id"] for r in rows if r["src"] == src["tag"]})
        print(f"  crop disimpan: {n_crop} dari {n_track_used} track")

    if not rows:
        print("\nTidak ada crop yang dihasilkan.")
        return

    # --- CSV ---
    cols = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    # --- Galeri HTML (per track, agar pelabelan per kendaraan) ---
    parts = ["<!doctype html><html lang='id'><head><meta charset='utf-8'>",
             "<title>Labeling crop plat — dataset_ft</title><style>",
             "body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}",
             "h2{font-size:15px;border-left:3px solid #4ea1ff;padding-left:8px;margin:22px 0 6px}",
             ".t{display:flex;flex-wrap:wrap;gap:8px}",
             "figure{margin:0;background:#1b1b1b;padding:6px;border-radius:6px;width:190px}",
             "img{width:100%;image-rendering:pixelated;background:#000}",
             "figcaption{font-size:10px;line-height:1.35;color:#9ad}",
             ".draft{color:#ffd166}code{color:#8ef}</style></head><body>",
             f"<h1 style='font-size:19px'>Labeling crop (Fase A.1) — {len(rows)} crop</h1>",
             "<p style='font-size:12px;color:#aaa'>Isi kolom <code>gt_text</code> di "
             f"<code>dataset_ft/labels_draft.csv</code>. Kolom <code>usul_koreksi</code> hanya draft.</p>"]
    for (tag, tid), g in sorted(galleries.items()):
        parts.append(f"<h2>{tag} · track #{tid} · {g['n']} frame deteksi "
                     f"· OCR dominan: <span class='draft'>{html.escape(g['draft'] or '-')}</span> "
                     f"→ usulan <code>{html.escape(g['usul'] or '-')}</code></h2><div class='t'>")
        for name, det in g["items"]:
            parts.append(
                f"<figure><img src='../crops/{name}' loading='lazy'>"
                f"<figcaption>f{det['frame']} · det {det.get('det_conf', 0):.2f} · "
                f"ocr {det.get('ocr_conf', 0):.2f}<br>"
                f"<span class='draft'>{html.escape(det.get('plat', '') or '(kosong)')}</span><br>"
                f"<code>{name[:-4]}</code></figcaption></figure>")
        parts.append("</div>")
    parts.append("</body></html>")
    with open(OUT_GALLERY, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))

    # --- Ringkasan ---
    confs = [r["ocr_conf"] for r in rows]
    print("\n=== RINGKASAN ===")
    print(f"crop total        : {len(rows)}")
    print(f"track total       : {len({(r['src'], r['track_id']) for r in rows})}")
    print(f"per sumber        : { {t: sum(1 for r in rows if r['src'] == t) for t in {r['src'] for r in rows}} }")
    print(f"crop ada teks OCR : {sum(1 for r in rows if r['ocr_raw'].strip())}")
    print(f"ocr_conf median   : {statistics.median(confs):.3f}")
    print(f"CSV               : {OUT_CSV}")
    print(f"galeri            : {OUT_GALLERY}")
    print(f"crop              : {OUT_CROPS}")


if __name__ == "__main__":
    main()
