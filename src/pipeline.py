"""ALPR pipeline — deteksi YOLOv8 + PaddleOCR + koreksi D3 + tracking D4 + locked display.

Diekstrak dari notebooks/ocr_inference_paddleocr.ipynb (sel 10 & 12) — pipeline
final uji_7c. Semua logika identik dengan notebook; perbedaan hanya pada
pengemasan (class ALPRPipeline, parameter dari ALPRConfig, tanpa path Colab).

Pemakaian:
    from src.config import ALPRConfig
    from src.pipeline import ALPRPipeline

    pipe = ALPRPipeline(ALPRConfig())
    dets = pipe.detect_and_read(frame_bgr)              # gambar / frame tunggal
    tracker = pipe.new_tracker()                        # video: 1 tracker per video
    dets = pipe.detect_and_read(frame_bgr, tracker=tracker, frame_idx=i)
    finals = tracker.summarize()
    annotated = pipe.annotate(frame_bgr, dets)
"""

from __future__ import annotations

import re
from collections import Counter

# PENTING: torch harus diimpor SEBELUM paddle.
# paddle 2.6.x dan torch sama-sama membawa libiomp5md.dll (Intel OpenMP) versi
# berbeda — DLL yang termuat lebih dulu "menang". Bila paddle duluan, import
# torch kemudian gagal (WinError 127 shm.dll). Urutan ini aman untuk keduanya.
import torch  # noqa: F401

import cv2
import numpy as np

from .config import ALPRConfig

# ---------------------------------------------------------------------------
# D8: Regex per-domain
# ---------------------------------------------------------------------------
PLATE_RE_ID = re.compile(r'^([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{0,3})$')
PLATE_RE_UK = re.compile(r'^([A-Z]{2})\s?(\d{2})\s?([A-Z]{3})$')


def validate_plate(text, domain='auto'):
    t = re.sub(r'[^A-Z0-9 ]', '', str(text).upper()).strip()
    compact = t.replace(' ', '')
    if domain in ('auto', 'ID'):
        m = PLATE_RE_ID.match(compact)
        if m:
            return True, 'ID'
    if domain in ('auto', 'UK'):
        m = PLATE_RE_UK.match(compact)
        if m:
            return True, 'UK'
    return False, '-'


def format_plate(text, domain):
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
# D3: Character-position correction
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


def correct_plate_text(raw, domain='auto'):
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


# ---------------------------------------------------------------------------
# D4 + PRD v2: Temporal voting (PlateTracker)
# ---------------------------------------------------------------------------
class PlateTracker:
    def __init__(self, iou_thresh=0.20, max_gap=25, vote_min=3, min_conf=0.30,
                 lock_min=3, override_ratio=2.0):
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.vote_min = vote_min
        self.min_conf = min_conf
        self.lock_min = lock_min
        self.override_ratio = override_ratio
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

    def _iou(self, a, b):
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

    def best(self, track_id):
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
                self.override_events.append({
                    'track_id': track_id, 'frame': tr['last_frame'],
                    'from': locked, 'to': top_plate,
                    'votes_from': locked_votes, 'votes_to': top_votes
                })
                tr['locked_plate'] = top_plate
                return top_plate, True
            return locked, True

        if top_votes >= self.lock_min:
            tr['locked_plate'] = top_plate
            return top_plate, True

        if top_votes >= self.vote_min:
            return top_plate, False
        return '', False

    def update(self, detections, frame_idx):
        assigned_track_ids = set()
        for det in detections:
            box = det['box']
            plate = det['plat_corrected']
            conf = det['ocr_conf']

            best_tr, best_iou = None, 0.0
            for tr in self.tracks:
                if tr['track_id'] in assigned_track_ids:
                    continue
                if frame_idx - tr['last_frame'] > self.max_gap:
                    continue
                iou = self._iou(box, tr['box'])
                if iou > self.iou_thresh and iou > best_iou:
                    best_tr, best_iou = tr, iou

            if best_tr:
                best_tr['box'] = box
                best_tr['last_frame'] = frame_idx
                if plate and conf >= self.min_conf:
                    best_tr['votes'].append((plate, conf))
                det['track_id'] = best_tr['track_id']
                assigned_track_ids.add(best_tr['track_id'])
            else:
                new_id = self._next_id
                self._next_id += 1
                new_tr = {
                    'track_id': new_id, 'box': box, 'last_frame': frame_idx,
                    'votes': [(plate, conf)] if (plate and conf >= self.min_conf) else [],
                    'locked_plate': None
                }
                self.tracks.append(new_tr)
                det['track_id'] = new_id
                assigned_track_ids.add(new_id)

    def summarize(self):
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
    """Deteksi plat (YOLOv8) + OCR (PaddleOCR) — parameter sesuai ALPRConfig (uji_7c)."""

    def __init__(self, config: ALPRConfig | None = None):
        self.config = config or ALPRConfig()
        self._model = None
        self._ocr = None

    # ---- lazy load ----
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_path)
        return self._model

    @property
    def ocr(self):
        if self._ocr is None:
            import paddle
            from paddleocr import PaddleOCR
            cfg = self.config
            self._allowed = set(cfg.allowlist)
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=cfg.use_gpu_ocr and paddle.device.get_device().startswith('gpu'),
                det_limit_side_len=cfg.det_limit,
                rec_batch_num=cfg.rec_batch,
                show_log=False,
            )
        return self._ocr

    # ---- OCR ----
    def ocr_read(self, plate_img):
        res = self.ocr.ocr(plate_img, cls=False)
        if not res or res[0] is None:
            return '', 0.0
        texts, confs = [], []
        allowed = self._allowed
        for line in res[0]:
            txt, conf = line[1][0], float(line[1][1])
            txt = ''.join(ch for ch in txt.upper() if ch in allowed)
            if txt:
                texts.append(txt)
                confs.append(conf)
        if not texts:
            return '', 0.0
        return ' '.join(texts), min(confs)

    # ---- preprocessing crop plat ----
    def preprocess_plate(self, img_bgr, box):
        cfg = self.config
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * cfg.padding), int((y2 - y1) * cfg.padding)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cfg.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
        else:
            enh = gray
        if cfg.upscale and cfg.upscale > 1:
            enh = cv2.resize(enh, None, fx=cfg.upscale, fy=cfg.upscale,
                             interpolation=cv2.INTER_CUBIC)
        return enh

    # ---- deteksi + OCR per frame/gambar ----
    def detect_and_read(self, img_bgr, conf=None, tracker=None, frame_idx=0):
        cfg = self.config
        conf = cfg.conf_det if conf is None else conf
        res = self.model.predict(img_bgr, conf=conf, imgsz=cfg.img_size, verbose=False)[0]
        out = []
        for box, bconf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            box = [int(v) for v in box]
            if (box[3] - box[1]) < cfg.min_height:
                continue
            crop_enh = self.preprocess_plate(img_bgr, box)
            if crop_enh is None:
                continue
            raw_text, ocr_conf = self.ocr_read(crop_enh)
            corrected_text, label, domain = correct_plate_text(raw_text, domain=cfg.domain_regex)
            valid, _ = validate_plate(corrected_text, domain=domain)
            out.append({
                'box': box,
                'det_conf': float(bconf),
                'plat_raw': raw_text,
                'ocr_conf': float(ocr_conf),
                'plat_corrected': corrected_text,
                'template_label': label,
                'domain': domain,
                'plat_valid': valid,
                'track_id': None
            })
        if tracker is not None:
            tracker.update(out, frame_idx)
        return out

    # ---- tracker baru (1 per video) ----
    def new_tracker(self):
        cfg = self.config
        return PlateTracker(iou_thresh=cfg.iou_track, max_gap=cfg.max_frame_gap,
                            vote_min=cfg.vote_min, min_conf=cfg.min_ocr_conf,
                            lock_min=cfg.lock_min, override_ratio=cfg.override_ratio)

    # ---- anotasi ----
    def annotate(self, img_bgr, detections):
        img_out = img_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            plate = d['plat_corrected']
            ocr_conf = d['ocr_conf']
            valid = d['plat_valid']

            # Warna bbox berdasarkan status validasi plat
            if valid:
                color = (0, 255, 0)      # Hijau = Valid
            elif plate:
                color = (0, 165, 255)    # Oranye = Belum/tidak valid
            else:
                color = (0, 0, 255)      # Merah = Tidak ada teks

            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

            label = f"{plate} ({ocr_conf:.2f})" if plate else "No Text"
            cv2.putText(img_out, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_out
