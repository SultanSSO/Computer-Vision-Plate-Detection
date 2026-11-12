"""Post-processing ALPR — character-position correction (D3) + temporal voting (D4).

Acuan: docs/PRD_decision_plat.md §4 Fase D, §5 (schema JSONL v2).

Modul ini dipakai oleh `notebook/ocr_inference.ipynb` (uji 3) dan bisa di-copy
MLOps untuk REST API. Semua fungsi pure-Python (tanpa cv2/torch).

Schema JSONL v2 (backward compatible, tambahan field):
    plat_corrected, domain, track_id, vote_count, is_stable
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# D3 — Character-position correction
# ---------------------------------------------------------------------------
# Peta substitusi karakter visual yang sering tertukar (PRD_decision_plat.md §1.3)
LETTER_TO_DIGIT = {
    "O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6",
    "B": "8", "A": "4", "T": "7", "E": "3", "Q": "0", "D": "0",
}
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B",
}

# Karakter yang TIDAK boleh hasil substitusi (terlalu jarang di plat nyata)
_FORBID_SUBS = {"Q", "W", "V"}

# Template posisi: 'L' = huruf, 'D' = angka
TEMPLATES = {
    "ID": [  # Indonesia: 1-2 huruf + 1-4 angka + 0-3 huruf
        ("LDDDDLLL", "ID H1A4H3"), ("LDDDDLL", "ID H1A4H2"),
        ("LDDDDL", "ID H1A4H1"), ("LDDDD", "ID H1A4"),
        ("LDDDLLL", "ID H1A3H3"), ("LDDDLL", "ID H1A3H2"),
        ("LDDDL", "ID H1A3H1"), ("LDDD", "ID H1A3"),
        ("LDDLLL", "ID H1A2H3"), ("LDDLL", "ID H1A2H2"),
        ("LDDL", "ID H1A2H1"), ("LDD", "ID H1A2"),
        ("LDLLL", "ID H1A1H3"), ("LDLL", "ID H1A1H2"), ("LDL", "ID H1A1H1"),
        ("LLDDDDLLL", "ID H2A4H3"), ("LLDDDDLL", "ID H2A4H2"),
        ("LLDDDDL", "ID H2A4H1"), ("LLDDDD", "ID H2A4"),
        ("LLDDDLLL", "ID H2A3H3"), ("LLDDDLL", "ID H2A3H2"),
        ("LLDDDL", "ID H2A3H1"), ("LLDDD", "ID H2A3"),
        ("LLDDLLL", "ID H2A2H3"), ("LLDDLL", "ID H2A2H2"),
        ("LLDDL", "ID H2A2H1"), ("LLDD", "ID H2A2"),
        ("LLDLLL", "ID H2A1H3"), ("LLDLL", "ID H2A1H2"), ("LLDL", "ID H2A1H1"),
    ],
    "UK": [  # UK modern: 2 huruf + 2 angka + 3 huruf (2001-sekarang)
        ("LLDDLLL", "UK LLDDLLL"),
    ],
}


def _apply_template(text: str, template: str) -> tuple[str, int]:
    """Terapkan template posisi ke teks. Return (hasil, jumlah_substitusi).

    Substitusi yang menghasilkan karakter aneh (Q, W, V) dihukum +2 cost.
    """
    out, cost = [], 0
    for ch, expect in zip(text, template):
        orig = ch
        if expect == "L" and ch.isdigit():
            ch = DIGIT_TO_LETTER.get(ch, ch)
            cost += 1
        elif expect == "D" and ch.isalpha():
            ch = LETTER_TO_DIGIT.get(ch, ch)
            cost += 1
        if ch != orig and ch in _FORBID_SUBS:
            cost += 2
        out.append(ch)
    return "".join(out), cost


def correct_plate_text(raw: str, domain: str = "auto") -> tuple[str, str, str]:
    """Koreksi teks OCR berdasarkan posisi karakter (D3).

    Args:
        raw: teks mentah dari OCR.
        domain: 'ID', 'UK', atau 'auto' (coba semua template, pilih cost terendah).

    Returns:
        (teks_terkoreksi, nama_template, domain_terpilih)
    """
    t = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
    if len(t) < 3:
        return t, "-", "-"

    candidates = []
    domains = [domain] if domain != "auto" else ["UK", "ID"]
    for dom in domains:
        for tpl, label in TEMPLATES.get(dom, []):
            if len(tpl) != len(t):
                continue
            corrected, cost = _apply_template(t, tpl)
            # Prioritaskan domain UK untuk teks 7 karakter (video uji saat ini UK)
            # dan ID untuk lainnya. Tie-breaker: domain 'UK' diprioritaskan jika cost sama.
            dom_bias = 0 if (dom == "UK" and len(t) == 7) else (1 if dom == "ID" else 2)
            candidates.append((corrected, label, dom, cost, dom_bias))

    if not candidates:
        return t, "-", "-"

    # Urutkan: cost terendah, lalu domain bias, lalu nama template.
    # Tie-breaker tambahan: template UK LLDDLLL diprioritaskan untuk 7 karakter
    # (format UK lebih spesifik/ketat daripada ID yang fleksibel).
    def _sort_key(cand):
        corrected, label, dom, cost, dom_bias = cand
        # Template UK LLDDLLL: kurangi cost efektif 1.5 agar menang tie-break
        # (format UK lebih spesifik/ketat daripada ID yang fleksibel)
        eff_cost = cost - 1.5 if (dom == "UK" and label == "UK LLDDLLL") else cost
        return (eff_cost, dom_bias, label)

    best = min(candidates, key=_sort_key)
    corrected, label, dom = best[0], best[1], best[2]

    # Validasi: jika hasil koreksi tidak lolos regex domain-nya, coba kandidat lain.
    # Jika tidak ada yang valid, kembalikan teks ASLI (bukan hasil koreksi paksa).
    valid, _ = validate_plate(corrected, domain=dom)
    if not valid:
        for cand in sorted(candidates, key=_sort_key)[1:]:
            v2, _ = validate_plate(cand[0], domain=cand[2])
            if v2:
                return cand[0], cand[1], cand[2]
        # Tidak ada yang valid → kembalikan teks asli, tandai domain '-'
        return t, "-", "-"

    return corrected, label, dom


# ---------------------------------------------------------------------------
# D4 — Temporal voting (tracking berbasis IoU + majority vote)
# ---------------------------------------------------------------------------

def iou(a: list, b: list) -> float:
    """Intersection over Union untuk 2 bbox [x1,y1,x2,y2]."""
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
    """Tracking sederhana: greedy IoU + temporal voting per track.

    Setiap deteksi baru dicocokkan ke track aktif berdasarkan IoU.
    Teks final per track = majority vote dari kandidat yang lolos filter.

    PRD v2 (Locked-Plate Display): setelah vote cukup, teks track DIKUNCI
    (`locked_plate`) — tampilan tidak ikut bacaan frame terbaru. Kunci hanya
    berubah lewat override rule (V3): kandidat baru vote >= override_ratio x
    vote kandidat terkunci DAN avg confidence lebih tinggi.
    """

    def __init__(self, iou_thresh: float = 0.25, max_gap: int = 15,
                 vote_min: int = 3, min_conf: float = 0.30,
                 lock_min: int = 3, override_ratio: float = 2.0):
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.vote_min = vote_min
        self.min_conf = min_conf
        self.lock_min = lock_min            # V2 — vote minimum sebelum dikunci
        self.override_ratio = override_ratio  # V3 — rasio vote untuk mengganti kunci
        self.tracks: list[dict] = []
        self._next_id = 0
        self.override_events: list[dict] = []  # audit trail override (V3)

    def _new_track(self, det: dict) -> dict:
        tr = {
            "track_id": self._next_id,
            "last_box": det["bbox"],
            "last_frame": det["frame"],
            "votes": [],            # list of (plat_corrected, ocr_conf, frame)
            "locked_plate": None,   # V1 — teks terkunci (None = belum terkunci)
        }
        self._next_id += 1
        return tr

    def _get(self, track_id: int) -> dict | None:
        for tr in self.tracks:
            if tr["track_id"] == track_id:
                return tr
        return None

    @staticmethod
    def _avg_conf(tr: dict, plate: str) -> float:
        vals = [v[1] for v in tr["votes"] if v[0] == plate]
        return sum(vals) / len(vals) if vals else 0.0

    def best(self, track_id: int) -> tuple[str, bool]:
        """Teks yang HARUS ditampilkan untuk track (V1–V3). Return (teks, locked)."""
        tr = self._get(track_id)
        if tr is None or not tr["votes"]:
            return "", False
        counter = Counter(v[0] for v in tr["votes"])
        top_plate, top_votes = counter.most_common(1)[0]

        if tr["locked_plate"]:
            locked = tr["locked_plate"]
            locked_votes = counter.get(locked, 0)
            # V3 — override: kandidat lain menang telak + confidence lebih tinggi
            if (top_plate != locked
                    and top_votes >= self.override_ratio * max(locked_votes, 1)
                    and self._avg_conf(tr, top_plate) > self._avg_conf(tr, locked)):
                self.override_events.append({
                    "track_id": track_id, "frame": tr["last_frame"],
                    "from": locked, "to": top_plate,
                    "votes_from": locked_votes, "votes_to": top_votes,
                })
                tr["locked_plate"] = top_plate
                return top_plate, True
            return locked, True

        # V2 — lock rule: kunci setelah cukup bukti
        if top_votes >= self.lock_min:
            tr["locked_plate"] = top_plate
            return top_plate, True
        return top_plate, False  # kandidat sementara

    def update(self, frame_idx: int, detections: list[dict]) -> list[dict]:
        """Update tracker dengan deteksi baru. Return deteksi yang sudah di-enrich."""
        active = [t for t in self.tracks
                  if frame_idx - t["last_frame"] <= self.max_gap]

        for det in detections:
            best, best_iou = None, 0.0
            for tr in active:
                score = iou(det["bbox"], tr["last_box"])
                if score > best_iou:
                    best, best_iou = tr, score

            if best is not None and best_iou >= self.iou_thresh:
                best["last_box"] = det["bbox"]
                best["last_frame"] = frame_idx
                if det.get("plat_corrected") and det.get("ocr_conf", 0) >= self.min_conf:
                    best["votes"].append((det["plat_corrected"], det["ocr_conf"], frame_idx))
                det["track_id"] = best["track_id"]
            else:
                tr = self._new_track(det)
                if det.get("plat_corrected") and det.get("ocr_conf", 0) >= self.min_conf:
                    tr["votes"].append((det["plat_corrected"], det["ocr_conf"], frame_idx))
                self.tracks.append(tr)
                det["track_id"] = tr["track_id"]

            det["vote_count"] = len([v for v in (best or self.tracks[-1])["votes"]
                                     if v[0] == det.get("plat_corrected", "")])
            det["is_stable"] = det["vote_count"] >= self.vote_min

            # V1/V4 — teks tampilan terkunci (plat_display) + status kunci
            display, locked = self.best(det["track_id"])
            det["plat_display"] = display if display else det.get("plat_corrected", "")
            det["display_locked"] = locked

        return detections

    def get_final_plates(self) -> list[dict]:
        """Return hasil final per track: majority vote dari kandidat stabil."""
        results = []
        for tr in self.tracks:
            if not tr["votes"]:
                continue
            counter = Counter(v[0] for v in tr["votes"])
            best_plate, best_count = counter.most_common(1)[0]
            avg_conf = sum(v[1] for v in tr["votes"] if v[0] == best_plate) / best_count
            results.append({
                "track_id": tr["track_id"],
                "plat_final": best_plate,
                "vote_count": best_count,
                "total_votes": len(tr["votes"]),
                "stability": best_count / len(tr["votes"]),
                "avg_ocr_conf": round(avg_conf, 4),
                "n_frame": tr["last_frame"] - min(v[2] for v in tr["votes"]) + 1,
            })
        return sorted(results, key=lambda x: -x["vote_count"])


# ---------------------------------------------------------------------------
# Regex per-domain (D8)
# ---------------------------------------------------------------------------
PLATE_RE_ID = re.compile(r"^([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{0,3})$")
PLATE_RE_UK = re.compile(r"^([A-Z]{2})\s?(\d{2})\s?([A-Z]{3})$")


def validate_plate(text: str, domain: str = "auto") -> tuple[bool, str]:
    """Validasi teks terhadap regex domain. Return (lolos, domain_terpilih)."""
    t = re.sub(r"[^A-Z0-9 ]", "", str(text).upper()).strip()
    compact = t.replace(" ", "")

    if domain in ("auto", "ID"):
        m = PLATE_RE_ID.match(compact)
        if m:
            wilayah, nomor, seri = m.groups()
            formatted = f"{wilayah} {nomor}" + (f" {seri}" if seri else "")
            return True, "ID"

    if domain in ("auto", "UK"):
        m = PLATE_RE_UK.match(compact)
        if m:
            area, age, serial = m.groups()
            return True, "UK"

    return False, "-"


# ---------------------------------------------------------------------------
# Pipeline lengkap: OCR mentah → koreksi → validasi → tracking
# ---------------------------------------------------------------------------

def process_detection(raw_text: str, ocr_conf: float, domain: str = "auto") -> dict:
    """Proses 1 deteksi: koreksi posisi → validasi regex → siap tracking."""
    corrected, template, dom = correct_plate_text(raw_text, domain=domain)
    valid, valid_dom = validate_plate(corrected, domain=dom if dom != "-" else "auto")
    return {
        "plat": raw_text.strip().upper(),
        "plat_corrected": corrected,
        "plat_valid": valid,
        "domain": valid_dom if valid else "-",
        "template": template,
        "ocr_conf": round(float(ocr_conf), 4),
    }


def format_plate_for_display(text: str, domain: str) -> str:
    """Format teks plat untuk tampilan (tambah spasi sesuai domain)."""
    t = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    if domain == "UK" and len(t) == 7:
        return f"{t[:2]} {t[2:4]} {t[4:]}"
    if domain == "ID":
        m = PLATE_RE_ID.match(t)
        if m:
            w, n, s = m.groups()
            return f"{w} {n}" + (f" {s}" if s else "")
    return t
