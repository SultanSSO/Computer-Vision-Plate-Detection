# Phase 3 — Setup Notes (OCR Inference Notebook)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 14 Sep 2026
**Status:** Phase 3 artefak SIAP — `ocr_inference.ipynb` di Drive, menunggu eksekusi di Colab

---

## 1. Konteks: hasil Phase 2 (final)

Target mAP50 ≥ 0,85 (PRD 01 §5) **tercapai** pada run ke-3:

| Run | Model | Epoch | mAP50 | Precision | Recall | Status |
| :-- | :-- | --: | --: | --: | --: | :-- |
| r1 | yolov8n | 50 | 0,8416 | 0,863 | 0,772 | baseline |
| r2 | yolov8n | 100 | 0,8111 | 0,836 | 0,712 | ⬇ overfitting (epoch dinaikkan, kapasitas mentok) |
| **r3** | **yolov8s** | 100 (early-stop @87, best @75) | **0,8609** ✅ | 0,797 | **0,864** | **PEMENANG** |

Pelajaran run 2→3: bottleneck bukan durasi training melainkan **kapasitas model**. Upgrade
yolov8n (3,2 jt parameter) → yolov8s (11,2 jt) menaikkan mAP50 melewati target dan recall
melonjak 0,71→0,86 (plat jauh lebih jarang terlewat) dengan trade-off precision turun tipis
(0,86→0,80) — untuk ALPR, plat terlewat lebih fatal daripada deteksi ganda.

**Artefak pemenang:** `MyDrive/PMLD/model/best.pt` & `last.pt` (22,5 MB, yolov8s r3).
Checkpoint berkala `epoch*.pt` di `runs/plat-yolov8s-r3/weights/` (~1,6 GB) sudah **dihapus**
untuk hemat storage — tersisa `best.pt`/`last.pt`.

---

## 2. Notebook `ocr_inference.ipynb` (20 sel, 9 sel kode)

Dibangun sesuai PRD 02 §1.1 (Phase 3), §3.5 (video pipeline), §5 (Langkah 4), dan PRD 01 §3.1/§4/§5.
Tersimpan di `MyDrive/PMLD/notebook/ocr_inference.ipynb` (md5 identik dengan lokal).

### Struktur sel

| Sel | Isi | Acuan PRD |
| :-- | :-- | :-- |
| 1 | Cek GPU (tidak di-assert — inferensi citra bisa di CPU, video lambat) | — |
| 2 | Mount Drive + path proyek + **guard**: `best.pt`, `dataset/images/test`, video wajib ada | §5 Langkah 1 |
| 3 | **Konfigurasi terpusat** | §3.5 |
| 4 | Install `ultralytics` + `easyocr` | — |
| 5 | **Pipeline inti** (lihat §3 di bawah) | 02 §1.1, 01 §2.1 |
| 6 | Uji 8 citra test → `annotated_images/` + `test_images_log.jsonl` | 01 §4, 02 §5 Langkah 4 |
| 7 | Evaluasi OCR (exact match + char accuracy) — target ≥80% | 01 §5 |
| 8 | Uji video 1 menit → `processed_video/` + log per frame | 02 §3.5, 01 §5 |
| 9 | Rekap plat paling sering terbaca di video | — |
| 10 | Ringkasan & serah terima ke MLOps | — |

### Parameter (sel 3)

| Parameter | Nilai | Fungsi |
| :-- | :-- | :-- |
| `CONF_DET` | 0.4 | Ambang confidence YOLO (selaras `CONF_TEST` notebook training) |
| `IMG_SIZE` | 640 | Samakan dengan training |
| `PADDING` | 0.10 | Crop plat dilebarkan 10% tiap sisi (bantu OCR baca tepi karakter) |
| `OCR_LANGS` | `['en']` | Plat Indonesia = huruf latin + angka |
| `ALLOWLIST` | `A–Z 0–9` | Batasi karakter yang dibaca EasyOCR |
| `MIN_OCR_CONF` | 0.30 | Hasil OCR di bawah ini ditandai `low_conf` |
| `FRAME_SKIP` | 3 | Proses 1 dari 3 frame; anotasi dipertahankan di frame antara (PRD 02 §3.5) |
| `N_SAMPLE_IMG` | 8 | Jumlah citra test yang diuji |
| `SEED` | 42 | Reproducibility |

---

## 3. Pipeline inti (sel 5) — self-contained untuk handover MLOps

Alur per deteksi (PRD 02 §1.1 Phase 3):

```
YOLOv8 best.pt → bbox [x1,y1,x2,y2]
   → crop (+padding 10%) → Grayscale + CLAHE          ← preprocess_plate()
   → EasyOCR (allowlist A-Z0-9) → teks mentah
   → post-processing regex plat Indonesia              ← clean_plate_text()
   → gambar bbox + teks di frame                       ← annotate()
```

* **`preprocess_plate(img, box, pad)`** — crop bbox dengan padding, lalu
  `cv2.cvtColor` (grayscale) + `cv2.createCLAHE(clipLimit=2.0)`.
* **`clean_plate_text(raw)`** — normalisasi uppercase, buang karakter non-alfanumerik,
  cocokkan ke pola plat Indonesia `^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$` (contoh `B 1234 XY`).
  Mengembalikan `(teks_terformat, lolos_regex)`.
* **`detect_and_read(img, conf)`** — gabungan deteksi→crop→OCR→regex; mengembalikan
  list dict `{bbox, det_conf, plat, plat_valid, ocr_conf}`.

Fungsi-fungsi ini sengaja ditulis self-contained agar MLOps tinggal menyalinnya saat
membungkus model menjadi REST API FastAPI (PRD 02 §4).

---

## 4. Keputusan desain: format `predictions_txt/` = JSON Lines

PRD 01 §4 hanya menyebut "log metadata hasil deteksi (.txt)" **tanpa mendefinisikan format**.
Diputuskan: **JSONL (1 baris JSON per deteksi)** — mudah di-parse stream oleh Backend Go
(webhook `/detections`, PRD 02 §4) dan tetap bisa dibaca manusia.

Schema per baris:

```json
{
  "file": "Cars00211.jpg",        // atau nama video
  "frame": 42,                    // hanya di log video
  "timestamp_s": 1.68,            // hanya di log video
  "bbox": [820, 510, 980, 560],
  "det_conf": 0.87,
  "plat": "B 1234 XYZ",
  "plat_valid": true,
  "ocr_conf": 0.92
}
```

⚠️ Format ini **usulan dari sisi AI Engineer** — sebaiknya disepakati dengan MLOps/Backend
sebelum integrasi API.

---

## 5. Deliverable Phase 3 (setelah notebook dijalankan)

```
MyDrive/PMLD/output/
├── annotated_images/                  ← citra test + bbox + teks OCR
├── predictions_txt/
│   ├── test_images_log.jsonl          ← 1 baris per deteksi citra
│   └── sample_video_1min_log.jsonl    ← deteksi per frame video
└── processed_video/
    └── sample_video_1min_annotated.mp4
```

## 6. Checklist target PRD 01 §5 (status per Phase 3 setup)

| Metrik | Target | Status |
| :-- | :-- | :-- |
| mAP50 | ≥ 85% | ✅ 0,8609 (Phase 2, r3) |
| OCR Text Accuracy | ≥ 80% | ⬜ lihat sel 7 setelah dijalankan |
| Video Testing | 100% selesai tanpa error | ⬜ lihat sel 8 |
| Output Completeness | visual + log tiap uji | ⬜ lihat folder `output/` |

---

## 7. Catatan & hal yang perlu dikonfirmasi

1. **Evaluasi OCR di sel 7 bersifat indikatif** — sebagian nama file test
   (`train018_jpg.rf.*.jpg`) bukan nomor plat asli, sehingga ground-truth otomatis dari
   nama file bisa underestimate. Untuk laporan, verifikasi manual beberapa sampel disarankan.
2. **Video uji 29,6 detik vs "1 menit" di PRD 02 §3.5** — file yang tersedia
   (`test_video/sample_video_1min.mp4`) durasinya 29,6 detik. Nama sudah mengikuti PRD,
   tapi durasi sebenarnya perlu dikonfirmasi ke tim.
3. **Video berat di CPU** — EasyOCR tanpa GPU lambat; runtime T4 disarankan (sel 1 hanya
   memperingatkan, tidak memaksa).
4. **Tuning cepat bila OCR < 80%**: naikkan `PADDING` (0,10→0,15), turunkan `MIN_OCR_CONF`,
   atau turunkan `CONF_DET` (0,4→0,3) agar plat kecil/samarnya ikut terdeteksi.

## 8. Cara menjalankan

1. Buka Colab → `MyDrive/PMLD/notebook/ocr_inference.ipynb`
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`
4. Pantau: sel 6 (citra beranotasi), sel 7 (akurasi OCR), sel 8 (video 100% tanpa error)
