# ALPR — YOLOv8 + EasyOCR (workspace uji_8)

Deteksi plat nomor kendaraan dan pembacaan teksnya dari video atau gambar.
Hasilnya berupa JSONL per deteksi dan video/gambar beranotasi. Backend dan
dashboard membaca hasil ini untuk ditampilkan.

Folder ini adalah workspace EasyOCR. Pipeline PaddleOCR (uji_7c) ada di root
branch `develop`, tidak di sini.

## Isi folder

| Path | Isi |
|---|---|
| `model/best.pt` | Bobot YOLOv8s (run `plat-yolov8s-r3`, 22,5 MB). File Git biasa, bukan LFS. |
| `model/baseline_metrics/` | Metrik tiap run training (JSON dan CSV). |
| `alpr_pipeline.py` | Modul pipeline: deteksi YOLO, OCR, koreksi karakter, tracking, locked display. |
| `scripts/plate_postprocess.py` | D3 (koreksi posisi karakter) dan D4 (temporal voting), pure Python. |
| `scripts/line_crossing.py` | Gerbang deteksi berbasis garis (opsional). |
| `scripts/extract_crops.py` | Ekstraksi crop plat dari log untuk pelabelan OCR. |
| `notebooks/03_inference_eval.ipynb` | Notebook inferensi dan evaluasi. Output sel sudah dihapus. |
| `output/uji_8_*/` | Hasil run: `predictions_txt/*.jsonl`, `annotated_images/`, `video_captures/`. |
| `docs/` | PRD dan catatan fase 1–4 (Bahasa Indonesia). |
| `data.yaml` | Konfigurasi dataset YOLO (path masih menunjuk Google Colab). |

Yang tidak ada di repo: dataset gambar, dan video mp4 (`test_video/`, `processed_video/`).
Dataset: https://www.kaggle.com/datasets/linkgish/indonesian-plate-number-from-multi-sources
Split 70/20/10 (train 525, val 150, test 75), seed 42.

## Mengunduh model

Model disimpan sebagai file Git biasa, jadi `git clone` cukup. Tidak perlu git-lfs.

```
git clone -b develop https://github.com/PMLD-13/alpr-ai.git
cd alpr-ai/easyocr-uji8
sha256sum model/best.pt   # harus bc35f721cfc02682a4bc04d6abd3b87992459b3acf52fef0f5e2634526aaaa45
```

Ukuran harus 22.524.330 byte. Kalau ukurannya sekitar 130 byte, itu pointer LFS
dari commit lama. Pakai commit terbaru di `develop`.

## Dependensi

Belum ada `requirements.txt` di folder ini. Yang dipakai dari kode:
`ultralytics` (YOLO), `easyocr`, `opencv-python`, `numpy`, dan `torch`.
Python 3.10 atau lebih baru.

## Pemakaian (pipeline)

```python
from alpr_pipeline import ALPRPipeline, ALPRConfig

pipe = ALPRPipeline(ALPRConfig(model_path="model/best.pt", gpu=False))

# gambar tunggal / frame
dets = pipe.detect_and_read(frame_bgr)

# video: satu tracker per video
tracker = pipe.new_tracker()
for i, frame in enumerate(frames):
    dets = pipe.detect_and_read(frame, tracker=tracker, frame_idx=i)
finals = tracker.get_final_plates()
annotated = pipe.annotate(frame, dets)
```

Parameter default (`ALPRConfig`):

| Parameter | Default | Fungsi |
|---|---|---|
| `conf_det` | 0.4 | Ambang confidence deteksi YOLO |
| `img_size` | 640 | Ukuran input YOLO |
| `padding` | 0.10 | Padding crop plat (10%) |
| `min_height` | 20 | Tinggi minimum bbox (px), lebih kecil di-skip |
| `min_ocr_conf` | 0.30 | Ambang confidence OCR |
| `upscale` | 3 | Faktor upscale crop sebelum OCR |
| `char_fix` | True | Koreksi posisi karakter (D3) |
| `temporal` | True | Voting antar frame (D4) |
| `vote_min` | 3 | Minimal vote untuk plat stabil |
| `iou_track` | 0.25 | Ambang IoU untuk tracking |
| `max_frame_gap` | 15 | Frame maksimal tanpa deteksi sebelum track diputus |
| `display_lock` | True | Kunci tampilan plat setelah stabil |
| `lock_min` | 3 | Minimal vote untuk mengunci tampilan |
| `override_ratio` | 2.0 | Hasil baru hanya menimpa kunci jika vote 2x lipat |
| `domain_regex` | `auto` | Validasi format plat: `auto`, `ID`, atau `UK` |

Pipeline per deteksi: YOLO → crop + padding → preprocessing (grayscale, CLAHE,
upscale) → EasyOCR (allowlist A–Z, 0–9) → koreksi D3 sesuai template format
plat → validasi regex domain → tracking IoU → voting D4 → locked display.

## Format JSONL (`predictions_txt/*.jsonl`)

Satu baris per deteksi per frame. Field:

| Field | Tipe | Keterangan |
|---|---|---|
| `frame` | int | Indeks frame |
| `timestamp_s` | float | Waktu dalam video (detik) |
| `file` | string | Nama video atau gambar sumber |
| `gate_active` | bool/null | Status gerbang line-crossing (null jika tidak dipakai) |
| `bbox` | [x1, y1, x2, y2] | Koordinat plat |
| `det_conf` | float | Confidence YOLO |
| `plat` | string | Teks mentah dari OCR |
| `plat_corrected` | string | Teks setelah koreksi D3 |
| `plat_valid` | bool | Lolos regex domain |
| `domain` | string | `ID`, `UK`, atau `-` |
| `template` | string | Template D3 yang dipakai, atau `-` |
| `ocr_conf` | float | Confidence OCR |
| `skipped_small` | bool | Bbox di bawah `min_height` |
| `track_id` | int | ID track |
| `vote_count` | int | Jumlah vote untuk teks ini |
| `is_stable` | bool | Sudah mencapai `vote_min` |
| `plat_display` | string | Teks yang ditampilkan (terkunci) |
| `display_locked` | bool | Tampilan sudah dikunci |

Contoh satu baris (disingkat):

```json
{"frame": 175, "timestamp_s": 5.83, "file": "demotol.mp4", "bbox": [163,160,208,172], "det_conf": 0.4285, "plat": "", "plat_corrected": "", "plat_valid": false, "domain": "-", "ocr_conf": 0.0, "skipped_small": true, "track_id": 0, "vote_count": 0, "is_stable": false, "plat_display": "", "display_locked": false}
```

Baris dengan `skipped_small: true` tidak dibaca OCR. Pakai `is_stable` atau
`display_locked` untuk memfilter hasil yang layak ditampilkan.

Jumlah baris per run: `uji_8_demoparking` 98, `uji_8_demopark` 208, `uji_8_demotol` 289.

## Metrik training (`model/baseline_metrics/`)

| Run | Base | Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| plat-yolov8n | yolov8n | 50 | 0,842 | 0,526 | 0,863 | 0,772 |
| plat-yolov8n-r2 | yolov8n | 100 | 0,811 | 0,486 | 0,836 | 0,712 |
| plat-yolov8s-r3 | yolov8s | 100 | 0,861 | 0,546 | 0,797 | 0,864 |

`best.pt` berasal dari run `plat-yolov8s-r3`. Ketiga run memakai dataset dan seed
yang sama (`seed 42`, imgsz 640, batch 16).

## Catatan dan batasan

- `data.yaml` masih menunjuk ke `/content/drive/MyDrive/PMLD_FT/...`. Ganti path
  itu sebelum training atau validasi di mesin lain.
- Angka di atas diambil dari file metrik yang tersimpan. Notebook tidak dijalankan
  ulang untuk folder ini.
- Keluaran JSONL belum dipetakan field per field ke webhook backend
  (`POST /api/detections`, field `tracker_id` dan `detections[]`). Ini langkah
  berikutnya.
- Akurasi OCR pada plat Indonesia dengan sudut samping masih terbatas. Hasil uji
  di `output/` ditujukan untuk demonstrasi alur, bukan untuk klaim akurasi.
