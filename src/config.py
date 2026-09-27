"""Konfigurasi pipeline ALPR (deteksi YOLOv8 + PaddleOCR) — parameter final uji_7c.

Seluruh parameter di sini adalah nilai yang dipakai pada uji_7c
(output/uji_7c_paddleocr_sample_vidio_2.mp4/) — jangan diubah jika ingin
mereproduksi hasil tersebut.

Path didefinisikan relatif terhadap root workspace (folder yang memuat file ini).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Path (relatif root workspace)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "model", "best.pt")
VIDEO_PATH = os.path.join(ROOT, "test_video", "sample_vidio_2.mp4")
OUTPUT_DIR = os.path.join(ROOT, "output")

#: Nama run hasil acuan (uji_7c) — dipakai sebagai default nama folder output
RUN_NAME_UJI7C = "uji_7c_paddleocr_sample_vidio_2.mp4"

VIDEO_NOT_FOUND_MSG = (
    f"Video uji tidak ditemukan: {VIDEO_PATH}\n"
    "Video sample TIDAK disertakan di repo Git (privat).\n"
    "Dapatkan dari grup WhatsApp tim, lalu taruh di: test_video/sample_vidio_2.mp4"
)


@dataclass
class ALPRConfig:
    """Parameter pipeline — default = konfigurasi final uji_7c (PaddleOCR)."""

    # ---- Path ----
    model_path: str = MODEL_PATH

    # ---- OCR Engine ----
    ocr_engine: str = "paddleocr"
    use_gpu_ocr: bool = True          # fallback CPU otomatis bila GPU tidak ada

    # ---- Deteksi YOLO ----
    conf_det: float = 0.15
    img_size: int = 1024
    padding: float = 0.17
    min_height: int = 15

    # ---- OCR ----
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    min_ocr_conf: float = 0.30
    upscale: int = 3
    use_clahe: bool = False           # grayscale saja — PaddleOCR lebih suka citra natural
    rec_batch: int = 6                # rec_batch_num PaddleOCR
    det_limit: int = 960              # det_limit_side_len PaddleOCR
    conf_lock: float = 0.85           # threshold instant lock OCR

    # ---- Video ----
    frame_skip: int = 1
    capture_every_s: int = 7
    n_sample_img: int = 8
    seed: int = 42

    # ---- D3: koreksi posisi karakter ----
    char_fix: bool = True

    # ---- D4: temporal voting ----
    temporal: bool = True
    vote_min: int = 3
    iou_track: float = 0.03
    max_frame_gap: int = 55

    # ---- Locked-plate display (PRD v2) ----
    display_lock: bool = True
    lock_min: int = 3
    override_ratio: float = 2.5

    # ---- D8: domain regex ----
    domain_regex: str = "auto"        # 'auto' / 'ID' / 'UK'


def check_video(path: str = VIDEO_PATH) -> str:
    """Kembalikan path video bila ada; raise FileNotFoundError dengan pesan jelas bila tidak."""
    if not os.path.exists(path):
        raise FileNotFoundError(VIDEO_NOT_FOUND_MSG)
    return path
