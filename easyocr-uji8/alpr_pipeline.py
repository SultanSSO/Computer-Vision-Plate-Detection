"""ALPR pipeline — deteksi YOLO + EasyOCR + koreksi D3 + tracking D4 + locked display.

Modul ini diekstrak dari `ocr_inference.ipynb` (sel 10) agar bisa dipakai ulang
oleh Gradio UI (Colab & lokal) DAN backend (FastAPI/dll) tanpa duplikasi kode.

Lokasi Drive: PMLD/notebook/alpr_pipeline.py
Lokasi lokal: notebook/alpr_pipeline.py

Pemakaian:
    from alpr_pipeline import ALPRPipeline, ALPRConfig
    pipe = ALPRPipeline(ALPRConfig(model_path='.../model/best.pt'))
    dets = pipe.detect_and_read(frame_bgr)              # gambar / frame tunggal
    tracker = pipe.new_tracker()                        # video: 1 tracker per video
    dets = pipe.detect_and_read(frame_bgr, tracker=tracker, frame_idx=i)
    finals = tracker.get_final_plates()
    annotated = pipe.annotate(frame_bgr, dets)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass
class ALPRConfig:
    """Seluruh parameter pipeline (sel 6 notebook ocr_inference.ipynb)."""
    model_path: str = "model/best.pt"
    ocr_custom: str | None = None          # folder model EasyOCR fine-tune (opsional)
    gpu: bool = False
    # Deteksi
    conf_det: float = 0.4
    img_size: int = 640
    padding: float = 0.10
    min_height: int = 20
    # OCR
    ocr_langs: tuple = ("en",)
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    min_ocr_conf: float = 0.30
    upscale: int = 3
    # D3 / D4 / PRD v2
    char_fix: bool = True
    temporal: bool = True
    vote_min: int = 3
    iou_track: float = 0.25
    max_frame_gap: int = 15
    display_lock: bool = True
    lock_min: int = 3
    override_ratio: float = 2.0
    # D8
    domain_regex: str = "auto"             # 'auto' | 'ID' | 'UK'


# ---------------------------------------------------------------------------
# D3 — Character-position correction
# ---------------------------------------------------------------------------
LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6',
                   'B': '8', 'A': '4', 'T': '7', 'E': '3', 'Q': '0', 'D': '0'}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
_FORBID_SUBS = {'Q', 'W', 'V'}

TEMPLATES = {
    'ID': [
        ('LDDDDLLL', 'ID H1A4H3'), ('LDDDDLL', 'ID H1A4H2'), ('LDDDDL', 'ID H1A4H1'), ('LDDDD', 'ID H1A4'),
        ('LDDDLLL', 'ID H1A3H3'), ('LDDDLL', 'ID H1A3H2'), ('LDDDL', 'ID H1A3H1'), ('LDDD', 'ID H1A3'),
        ('LDDLLL', 'ID H1A2H3'), ('LDDLL', 'ID H1A2H2'), ('LDDL', 'ID H1A2H1'), ('LDD', 'ID H1A2'),
        ('LDLLL', 'ID H1A1H3'), ('LDLL', 'ID H1A1H2'), ('LDL', 'ID H1A1H1'),
        ('LLDDDDLLL', 'ID H2A4H3'), ('LLDDDDLL', 'ID H2A4H2'), ('LLDDDDL', 'ID H2A4H1'), ('LLDDDD', 'ID H2A4'),
        ('LLDDDLLL', 'ID H2A3H3'), ('LLDDDLL', 'ID H2A3H2'), ('LLDDDL', 'ID H2A3H1'), ('LLDDD', 'ID H2A3'),
        ('LLDDLLL', 'ID H2A2H3'), ('LLDDLL', 'ID H2A2H2'), ('LLDDL', 'ID H2A2H1'), ('LLDD', 'ID H2A2'),
        ('LLDLLL', 'ID H2A1H3'), ('LLDLL', 'ID H2A1H2'), ('LLDL', 'ID H2A1H1'),
    ],
    'UK': [('LLDDLLL', 'UK LLDDLLL')],
}


def _apply_template(text, template):
    out, cost = [], 0
    for ch, expect in zip(text, template):
        orig = ch
        if expect == 'L' and ch.isdigit():
            ch = DIGIT_TO_LETTER.get(ch, ch); cost += 1
        elif expect == 'D' and ch.isalpha():
            ch = LETTER_TO_DIGIT.get(ch, ch); cost += 1
        if ch != orig and ch in _FORBID_SUBS:
            cost += 2
        out.append(ch)
    return ''.join(out), cost


# ---------------------------------------------------------------------------
# D8 — Regex per-domain
# ---------------------------------------------------------------------------
PLATE_RE_ID = re.compile(r'^([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{0,3})$')
PLATE_RE_UK = re.compile(r'^([A-Z]{2})\s?(\d{2})\s?([A-Z]{3})$')


def validate_plate(text, domain='auto'):
    """Validasi teks terhadap regex domain. Return (lolos, domain_terpilih)."""
    t = re.sub(r'[^A-Z0-9 ]', '', str(text).upper()).strip()
    compact = t.replace(' ', '')
    if domain in ('auto', 'ID') and PLATE_RE_ID.match(compact):
        return True, 'ID'
    if domain in ('auto', 'UK') and PLATE_RE_UK.match(compact):
        return True, 'UK'
    return False, '-'


def correct_plate_text(raw, domain='auto'):
    """Koreksi teks OCR berdasarkan posisi karakter (D3).

    Returns: (teks_terkoreksi, nama_template, domain_terpilih)
    """
    t = re.sub(r'[^A-Z0-9]', '', str(raw).upper())
    if len(t) < 3:
        return t, '-', '-'
    candidates = []
    domains = [domain] if domain != 'auto' else ['UK', 'ID']
    for dom in domains:
        for tpl, label in TEMPLATES.get(dom, []):
            if len(tpl) != len(t):
                continue
            corrected, cost = _apply_template(t, tpl)
            dom_bias = 0 if (dom == 'UK' and len(t) == 7) else (1 if dom == 'ID' else 2)
            candidates.append((corrected, label, dom, cost, dom_bias))
    if not candidates:
        return t, '-', '-'

    def _sort_key(c):
        eff = c[3] - 1.5 if (c[2] == 'UK' and c[1] == 'UK LLDDLLL') else c[3]
        return (eff, c[4], c[1])

    best = min(candidates, key=_sort_key)
    corrected, label, dom = best[0], best[1], best[2]
    valid, _ = validate_plate(corrected, domain=dom)
    if not valid:
        for cand in sorted(candidates, key=_sort_key)[1:]:
            v2, _ = validate_plate(cand[0], domain=cand[2])
            if v2:
                return cand[0], cand[1], cand[2]
        return t, '-', '-'
    return corrected, label, dom


def format_plate(text, domain):
    """Format teks plat untuk tampilan (tambah spasi sesuai domain)."""
    t = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    if domain == 'UK' and len(t) == 7:
        return f'{t[:2]} {t[2:4]} {t[4:]}'
    if domain == 'ID':
        m = PLATE_RE_ID.match(t)
        if m:
            w, n, s = m.groups()
            return f'{w} {n}' + (f' {s}' if s else '')
    return t


# ---------------------------------------------------------------------------
# D4 + PRD v2 — Temporal voting + Locked-Plate Display
# ---------------------------------------------------------------------------
def _iou(a, b):
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


class PlateTracker:
    """Greedy IoU tracking + majority vote (D4) + locked display (PRD v2: V1–V3)."""

    def __init__(self, iou_thresh=0.25, max_gap=15, vote_min=3, min_conf=0.30,
                 lock_min=3, override_ratio=2.0, display_lock=True):
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.vote_min = vote_min
        self.min_conf = min_conf
        self.lock_min = lock_min
        self.override_ratio = override_ratio
        self.display_lock = display_lock
        self.tracks = []
        self._next_id = 0
        self.override_events = []

    def _get(self, track_id):
        for tr in self.tracks:
            if tr['track_id'] == track_id:
                return tr
        return None

    @staticmethod
    def _avg_conf(tr, plate):
        vals = [v[1] for v in tr['votes'] if v[0] == plate]
        return sum(vals) / len(vals) if vals else 0.0

    def best(self, track_id):
        """Teks yang HARUS ditampilkan untuk track (V1–V3). Return (teks, locked)."""
        tr = self._get(track_id)
        if tr is None or not tr['votes']:
            return '', False
        counter = Counter(v[0] for v in tr['votes'])
        top_plate, top_votes = counter.most_common(1)[0]
        if tr['locked_plate']:
            locked = tr['locked_plate']
            locked_votes = counter.get(locked, 0)
            if (top_plate != locked
                    and top_votes >= self.override_ratio * max(locked_votes, 1)
                    and self._avg_conf(tr, top_plate) > self._avg_conf(tr, locked)):
                self.override_events.append({'track_id': track_id, 'frame': tr['last_frame'],
                                             'from': locked, 'to': top_plate,
                                             'votes_from': locked_votes, 'votes_to': top_votes})
                tr['locked_plate'] = top_plate
                return top_plate, True
            return locked, True
        if top_votes >= self.lock_min:
            tr['locked_plate'] = top_plate
            return top_plate, True
        return top_plate, False

    def update(self, frame_idx, detections):
        active = [t for t in self.tracks if frame_idx - t['last_frame'] <= self.max_gap]
        for det in detections:
            best, best_iou = None, 0.0
            for tr in active:
                score = _iou(det['bbox'], tr['last_box'])
                if score > best_iou:
                    best, best_iou = tr, score
            if best is not None and best_iou >= self.iou_thresh:
                best['last_box'] = det['bbox']
                best['last_frame'] = frame_idx
                if det.get('plat_corrected') and det.get('ocr_conf', 0) >= self.min_conf:
                    best['votes'].append((det['plat_corrected'], det['ocr_conf'], frame_idx))
                det['track_id'] = best['track_id']
            else:
                tr = {'track_id': self._next_id, 'last_box': det['bbox'],
                      'last_frame': frame_idx, 'votes': [], 'locked_plate': None}
                self._next_id += 1
                if det.get('plat_corrected') and det.get('ocr_conf', 0) >= self.min_conf:
                    tr['votes'].append((det['plat_corrected'], det['ocr_conf'], frame_idx))
                self.tracks.append(tr)
                det['track_id'] = tr['track_id']
            det['vote_count'] = len([v for v in (best or self.tracks[-1])['votes']
                                     if v[0] == det.get('plat_corrected', '')])
            det['is_stable'] = det['vote_count'] >= self.vote_min
            if self.display_lock:
                display, locked = self.best(det['track_id'])
                det['plat_display'] = display if display else det.get('plat_corrected', '')
                det['display_locked'] = locked
            else:
                det['plat_display'] = det.get('plat_corrected', '')
                det['display_locked'] = False
        return detections

    def get_final_plates(self):
        results = []
        for tr in self.tracks:
            if not tr['votes']:
                continue
            counter = Counter(v[0] for v in tr['votes'])
            best_plate, best_count = counter.most_common(1)[0]
            avg_conf = sum(v[1] for v in tr['votes'] if v[0] == best_plate) / best_count
            results.append({
                'track_id': tr['track_id'],
                'plat_final': tr['locked_plate'] or best_plate,
                'locked': bool(tr['locked_plate']),
                'vote_count': best_count,
                'total_votes': len(tr['votes']),
                'stability': round(best_count / len(tr['votes']), 4),
                'avg_ocr_conf': round(avg_conf, 4),
            })
        return sorted(results, key=lambda x: -x['vote_count'])


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------
class ALPRPipeline:
    """Deteksi + OCR + D3/D8, siap dipakai backend maupun UI."""

    def __init__(self, config: ALPRConfig):
        self.cfg = config
        from ultralytics import YOLO
        import easyocr
        self.model = YOLO(config.model_path)
        if config.ocr_custom:
            self.reader = easyocr.Reader(list(config.ocr_langs), gpu=config.gpu,
                                         model_storage_directory=config.ocr_custom,
                                         download_enabled=False)
        else:
            self.reader = easyocr.Reader(list(config.ocr_langs), gpu=config.gpu)

    def new_tracker(self) -> PlateTracker:
        c = self.cfg
        return PlateTracker(iou_thresh=c.iou_track, max_gap=c.max_frame_gap,
                            vote_min=c.vote_min, min_conf=c.min_ocr_conf,
                            lock_min=c.lock_min, override_ratio=c.override_ratio,
                            display_lock=c.display_lock)

    def preprocess_plate(self, img_bgr, box):
        """Crop bbox + padding -> grayscale + CLAHE -> upscale (identik notebook)."""
        c = self.cfg
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * c.padding), int((y2 - y1) * c.padding)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        if c.upscale and c.upscale > 1:
            enh = cv2.resize(enh, None, fx=c.upscale, fy=c.upscale,
                             interpolation=cv2.INTER_CUBIC)
        return enh

    def detect_and_read(self, img_bgr, conf=None, tracker=None, frame_idx=0):
        """Deteksi + OCR + koreksi D3 untuk 1 frame/gambar.

        Jika tracker diberikan, deteksi di-enrich dengan track_id, plat_display,
        display_locked (locked display PRD v2).
        """
        c = self.cfg
        conf = c.conf_det if conf is None else conf
        res = self.model.predict(img_bgr, conf=conf, imgsz=c.img_size, verbose=False)[0]
        out = []
        for box, bconf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            box = [int(v) for v in box]
            small = (box[3] - box[1]) < c.min_height
            plate_img = None if small else self.preprocess_plate(img_bgr, box)
            text, oconf, valid = '', 0.0, False
            corrected, template, domain = '', '-', '-'
            if plate_img is not None:
                ocr = self.reader.readtext(plate_img, allowlist=c.allowlist)
                if ocr:
                    raw = ' '.join(t[1] for t in ocr)
                    oconf = float(min(t[2] for t in ocr))
                    text = raw.strip().upper()
                    if c.char_fix:
                        corrected, template, domain = correct_plate_text(raw, domain=c.domain_regex)
                    else:
                        corrected = text
                    valid, valid_domain = validate_plate(
                        corrected, domain=domain if domain != '-' else 'auto')
                    if valid:
                        domain = valid_domain
            det = {
                'bbox': box, 'det_conf': round(float(bconf), 4),
                'plat': text, 'plat_corrected': corrected, 'plat_valid': valid,
                'domain': domain, 'template': template,
                'ocr_conf': round(oconf, 4), 'skipped_small': bool(small),
            }
            if tracker is not None:
                det['frame'] = frame_idx
            out.append(det)
        if tracker is not None:
            tracker.update(frame_idx, out)
        return out

    @staticmethod
    def annotate(img_bgr, dets):
        """Gambar bbox + teks (V6): hijau+| = LOCKED, oranye+.. = sementara,
        merah = tidak valid/tidak terbaca."""
        img = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            shown = d.get('plat_display') or d.get('plat_corrected') or ''
            locked = d.get('display_locked', False)
            if shown:
                label = format_plate(shown, d['domain'])
                label += ' |' if locked else ' ..'
            else:
                label = f"plat? ({d['det_conf']:.2f})"
            if locked:
                color = (0, 200, 0)
            elif shown:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
        return img
