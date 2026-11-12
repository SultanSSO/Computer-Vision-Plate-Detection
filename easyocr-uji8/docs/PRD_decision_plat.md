# PRD — Decision: Plat (Fine-Tuning & OCR Stabilization)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 21 Sep 2026
**Status:** Draft — menunggu persetujuan tim sebelum eksekusi
**Dokumen induk:** PRD 01 (Project Overview), PRD 02 (Technical Architecture), PHASE1–3 SETUP NOTES

---

## 0. Ringkasan Eksekutif

Pipeline ALPR end-to-end (YOLOv8 deteksi → preprocessing → EasyOCR → regex) **sudah berfungsi** dan
menghasilkan output lengkap (video beranotasi, capture frame, log JSONL). Namun kualitas **pembacaan
karakter OCR belum stabil**: pada uji terbaru 96% deteksi menghasilkan teks, tetapi mayoritas mengandung
kesalahan karakter (O↔0, I↔1, Z↔2) dan hanya 17% yang lolos format plat Indonesia.

Dokumen ini memutuskan **langkah selanjutnya**: fine-tuning model dari bobot terbaik saat ini dan
stabilisasi pembacaan OCR, dengan target kuantitatif yang terukur.

---

## 1. Kondisi Saat Ini (Baseline Terukur)

### 1.1 Riwayat & mutasi yang sudah dilakukan

| Fase | Hasil | Bukti |
| :-- | :-- | :-- |
| Phase 1 | Dataset 750 citra, split 70/20/10 (525/150/75), kelas `license_plate`, `nc: 1` | `dataset/`, `data.yaml`, `docs/phase1_dataset_report.json` |
| Phase 2 | 3 run training; **pemenang r3: YOLOv8s, mAP50 0,8609** (target ≥0,85 ✅) | `model/best.pt` (22,5 MB), `docs/PHASE2_SETUP_NOTES.md` |
| Phase 3 uji 1 | Video CCTV simpang (29,6 dtk): deteksi jalan, **OCR gagal** (plat terlalu kecil, hanya 106/644 ada teks, 17 valid) | `output/uji_1_sample_video_1min/` |
| Phase 3 uji 2 | Video baru (60 dtk, 1080p): **OCR jauh lebih baik** setelah upscale 3× | `output/uji_2_sample_video_1min_fix.mp4/` |

**Mutasi teknis pada pipeline (uji 1 → uji 2):**
1. `UPSCALE = 3` — crop plat diperbesar 3× (INTER_CUBIC) sebelum EasyOCR
2. `MIN_HEIGHT = 20` — bbox lebih pendek dari 20 px di-skip dari OCR (field `skipped_small` di log)
3. Output di-root per uji: `output/uji_<n>_<video_tag>/{annotated_images, predictions_txt, processed_video, video_captures}`
4. Capture frame beranotasi otomatis tiap 10 detik → `video_captures/`
5. **Rename video sumber** menjadi `sample_video_1min_fix.mp4` (ber-ekstensi) — semua path diarahkan ke sana

### 1.2 Baseline kualitas OCR (uji 2, video `sample_video_1min_fix.mp4`)

| Metrik | Nilai | Keterangan |
| :-- | --: | :-- |
| Total deteksi | 854 | 600 frame diproses (frame-skip 3) |
| Skip bbox kecil | 11 | `skipped_small` |
| Deteksi menghasilkan teks | 823 (**96,4%**) | ✅ cakupan OCR sangat baik |
| Lolos regex plat Indonesia | 143 (**16,7%**) | ❌ rendah |
| OCR Text Accuracy (target PRD 01 §5 ≥ 80%) | **belum terukur** | perlu ground truth |

**Contoh plat terbaca (top):** `EYGINBG`, `NL64067`, `1G65ZFX`, `EFIODZT`, `GXI50GJ`, `0G65ZFX`, `GJO6EPD`…

### 1.3 Diagnosis akar masalah

1. **Kesalahan substitusi karakter visual** — O↔0, I↔1, Z↔2, G↔6 (`GXI50GJ` seharusnya `GX15 OGJ`; `0G65ZFX` ↔ `OG65 ZFX`). EasyOCR belum di-fine-tune ke font plat kendaraan.
2. **Domain mismatch sementara** — video uji saat ini berisi **plat UK** (format `AB12 CDE`), sehingga regex plat Indonesia (`^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$`) otomatis menolak sebagian besar hasil. Angka 16,7% **bukan murni kesalahan OCR**.
3. **Tidak ada koreksi berbasis posisi karakter** — plat punya struktur posisi (huruf–angka–huruf) yang bisa dipakai memperbaiki substitusi (mis. posisi ke-2 pada `GXI50GJ` harus angka → `I`→`1`).
4. **Belum ada temporal voting** — kendaraan yang sama muncul di banyak frame; hasil OCR antar-frame belum digabung (terlihat dari varian `6JO6EPD`/`GJO6EPD`/`6J06EPD` untuk kendaraan yang sama).
5. **Detektor kadang longgar** — bbox menyertakan area non-plat, membingungkan recognizer.

---

## 2. Keputusan (Decision)

| # | Keputusan | Status |
| :- | :-- | :-- |
| D1 | **Fine-tune YOLOv8 detector** dari `best.pt` (r3) dengan data hasil kurasi video uji + augmentasi (blur, haze, low-light) — bukan training dari nol | Diusulkan |
| D2 | **Fine-tune recognizer EasyOCR** (custom recognition head) dengan dataset crop plat + ground truth teks | Diusulkan |
| D3 | **Character-position correction**: mapping koreksi per posisi menurut template plat (ID: H–A–H; UK: H–A–H) — O→0 di posisi angka, 0→O di posisi huruf, I→1, Z→2, dst. | Diusulkan |
| D4 | **Temporal voting**: kelompokkan deteksi per track kendaraan (bbox overlap antar-frame), pilih teks plat paling konsisten (majority vote) | Diusulkan |
| D5 | **Tidak ganti engine OCR** (tetap EasyOCR) — perubahan di model + post-processing, pipeline tidak berubah | Diputuskan |
| D6 | **Dataset training baru disimpan terpisah** di `dataset_ft/` — `dataset/` asli tidak diubah (imutabilitas data Phase 1) | Diputuskan |
| D7 | Format log JSONL dipertahankan, **ditambah field** `plat_corrected`, `track_id`, `vote_count` (backward compatible) | Diusulkan |
| D8 | Target regex final: **plat Indonesia**. Plat UK pada video uji dievaluasi dengan regex UK terpisah, tidak dicampur dalam metrik utama | Diputuskan |

---

## 3. Ruang Lingkup

### 3.1 Dalam ruang lingkup
- Kurasi dataset fine-tuning: crop plat dari video uji 1 & 2 + ground truth teks (manual/semi-otomatis)
- Augmentasi citra plat: blur, haze/kabut, low-light, rotasi kecil, noise — meniru kondisi video
- Fine-tune detector YOLOv8 (run baru `plat-yolov8s-ft-r4`)
- Fine-tune recognizer EasyOCR (custom checkpoint)
- Modul post-processing: character-position correction (D3) + temporal voting (D4)
- Evaluasi terukur terhadap ground truth (CER, plate accuracy) + perbandingan dengan baseline §1.2
- Update notebook `ocr_inference.ipynb` (uji 3) + dokumentasi hasil

### 3.2 Di luar ruang lingkup (tidak berubah)
- Struktur `dataset/` Phase 1 dan bobot `model/best.pt` (r3) — dibekukan sebagai baseline
- Format umum output `output/uji_<n>_*/` dan schema dasar JSONL
- Integrasi FastAPI / webhook Backend (Phase 4) — menunggu hasil fase ini
- Pelatihan ulang dari nol (scratch)

---

## 4. Rencana Kerja (4 Fase)

### Fase A — Dataset Fine-Tuning (target: 1–2 hari)
1. **Ekstraksi kandidat crop plat** dari log uji 1 & 2 (gunakan `bbox` + `frame` untuk memotong dari video sumber).
2. **Pelabelan ground truth** teks plat per crop (manual, dibantu pratinjau hasil OCR sebagai draft).
3. **Target jumlah:** minimal **300–500 crop berlabel** (semakin banyak semakin baik; kelas karakter seimbang A–Z, 0–9).
4. **Augmentasi** ×3–×5 per sampel: Gaussian/motion blur, haze, brightness/contrast rendah, rotasi ±5°, noise.
5. **Simpan ke `dataset_ft/`** dengan struktur YOLO untuk detector, dan `crops/ + labels.csv` untuk recognizer.
6. Verifikasi: tidak ada kebocoran split (crop dari video yang sama tidak tersebar di train & val).

### Fase B — Fine-Tune Detector YOLOv8 (target: 0,5 hari)
1. Inisialisasi dari `model/best.pt` (bukan `yolov8s.pt` pretrained umum).
2. Konfigurasi: `imgsz 640`, epoch 30–50, `patience 10`, lr lebih kecil dari training awal (mis. 0,001), augmentasi haze/blur aktif.
3. Run name: `plat-yolov8s-ft-r4`. Simpan `best.pt` → `model/best_ft.pt` (tidak menimpa r3).
4. **Gate kualitas:** mAP50 pada val `dataset_ft/` ≥ baseline r3 (0,8609). Jika lebih rendah, evaluasi ulang data sebelum lanjut.

### Fase C — Fine-Tune Recognizer EasyOCR (target: 1 hari)
1. Siapkan format training EasyOCR custom recognition (crop grayscale/CLAHE + label teks, sesuai pipeline inferensi — **konsistensi preprocessing wajib**).
2. Fine-tune dari checkpoint `english_g2` (bukan dari nol), charset `A–Z 0–9`, beberapa epoch dengan early stop.
3. Simpan checkpoint → `model/easyocr_ft/`. Notebook memuat custom recognizer via `recog_network` / model storage path.
4. **Gate kualitas:** CER pada val crops turun ≥ 30% relatif terhadap recognizer bawaan.

### Fase D — Stabilisasi OCR + Evaluasi (target: 1 hari)
1. **Character-position correction** (`correct_plate_text()`): template posisi; koreksi O↔0, I↔1, Z↔2, S↔5, B↔8, G↔6 sesuai posisi huruf/angka.
2. **Temporal voting** di pipeline video: track sederhana via IoU bbox antar-frame terproses; teks final per track = majority vote dari kandidat valid.
3. Jalankan **uji 3** (`sample_video_1min_fix.mp4`, folder `output/uji_3_*`) dengan konfigurasi baru.
4. Evaluasi terhadap ground truth subset → hitung **CER** dan **plate accuracy (exact match)**.
5. Tulis `docs/PHASE4_FINETUNE_NOTES.md` + perbarui tabel target.

---

## 5. Konfigurasi & Path (Referensi Implementasi)

```python
# ---- Path (sel 2 notebook) ----
VIDEO_PATH  = f'{PROJECT_DIR}/test_video/sample_video_1min_fix.mp4'  # SUDAH ber-ekstensi .mp4
VIDEO_TAG   = 'sample_video_1min_fix.mp4'
RUN_NAME    = f'uji_3_{VIDEO_TAG}'     # uji berikutnya = uji_3

# ---- Model hasil fine-tune ----
MODEL_PATH       = f'{PROJECT_DIR}/model/best_ft.pt'        # detector ft (Fase B)
OCR_CUSTOM_MODEL = f'{PROJECT_DIR}/model/easyocr_ft'        # recognizer ft (Fase C)

# ---- Parameter OCR (baseline uji 2, dipertahankan) ----
UPSCALE    = 3
MIN_HEIGHT = 20
PADDING    = 0.10
CONF_DET   = 0.4
FRAME_SKIP = 3

# ---- Baru (Fase D) ----
CHAR_FIX   = True    # character-position correction
TEMPORAL   = True    # voting antar-frame per track
VOTE_MIN   = 3       # minimal jumlah vote sebelum teks dianggap stabil
```

**Schema JSONL (v2, backward compatible):**
```json
{
  "frame": 300, "timestamp_s": 10.0, "file": "sample_video_1min_fix.mp4",
  "bbox": [x1, y1, x2, y2], "det_conf": 0.81,
  "plat": "GXI50GJ", "plat_valid": false, "ocr_conf": 0.72,
  "skipped_small": false,
  "plat_corrected": "GX15 OGJ", "track_id": 7, "vote_count": 5
}
```

---

## 6. Target & Kriteria Keberhasilan

| Metrik | Baseline uji 2 | Target uji 3 | Cara ukur |
| :-- | --: | --: | :-- |
| Deteksi menghasilkan teks | 96,4% | ≥ 95% (jaga) | log JSONL |
| CER (character error rate) | belum terukur | **≤ 20%** | vs ground truth subset |
| Plate accuracy (exact match) | belum terukur | **≥ 80%** ✅ target PRD 01 §5 | vs ground truth subset |
| Lolos regex (sesuai domain video) | 16,7% (regex ID pada video UK) | **≥ 70%** (regex UK untuk video fix) | log JSONL |
| Konsistensi antar-frame (teks stabil per track) | rendah (varian banyak) | ≥ 80% track punya 1 teks dominan | log + track_id |
| Video testing 100% tanpa error | ✅ | ✅ jaga | sel video |

> Catatan: metrik utama **plate accuracy vs ground truth**, bukan lolos regex — regex hanya indikator format,
> dan selama video uji berisi plat asing, keduanya harus dipisahkan (D8).

---

## 7. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| :-- | :-- | :-- |
| Data berlabel kurang (<300 crop) | fine-tune underfit, OCR tidak membaik | augmentasi agresif; tambah sumber (citra test, video uji 1); turunkan target epoch |
| Fine-tune detector menurunkan mAP50 | deteksi memburuk | simpan `best_ft.pt` terpisah; bandingkan dengan r3, pakai yang terbaik (rollback mudah) |
| Domain UK vs Indonesia | metrik regex membingungkan | evaluasi per-domain (D8); target akhir tetap plat Indonesia, UK hanya uji ketangguhan |
| Temporal voting salah menggabungkan track berbeda | plat tertukar antar kendaraan | threshold IoU konservatif; track baru jika IoU < ambang; simpan semua kandidat di log |
| EasyOCR custom training kompleks/gagal di Colab | Fase C molor | fallback: pertahankan recognizer bawaan + perkuat D3/D4 (koreksi posisi + voting) — tetap ada peningkatan |
| Waktu proses video naik (tracking + voting) | runtime memanjang | voting dilakukan pasca-passes pada log, bukan real-time; pipeline video tetap linear |

---

## 8. Struktur Workspace Setelah Fase Ini

```
MyDrive/PMLD/
├── dataset/                    ← BEKU (Phase 1, tidak diubah)
├── dataset_ft/                 ← BARU: data fine-tuning (crops + labels + augmentasi)
├── model/
│   ├── best.pt                 ← BEKU (r3 baseline)
│   ├── best_ft.pt              ← BARU (Fase B)
│   └── easyocr_ft/             ← BARU (Fase C)
├── notebook/
│   ├── yolo_training.ipynb     ← tambah blok fine-tune (atau notebook terpisah ft)
│   └── ocr_inference.ipynb     ← diperbarui (uji 3, D3 + D4)
├── output/
│   ├── uji_1_sample_video_1min/
│   ├── uji_2_sample_video_1min_fix.mp4/
│   └── uji_3_sample_video_1min_fix.mp4/   ← BARU hasil uji 3
└── docs/
    ├── PRD_decision_plat.md    ← dokumen ini
    └── PHASE4_FINETUNE_NOTES.md           ← BARU setelah eksekusi
```

---

## 9. Keputusan yang Dibutuhkan dari Tim

1. Setujui D1–D4 (fine-tune detector + recognizer + koreksi posisi + temporal voting)?
2. Siapa yang melabeli ground truth crop plat (target 300–500)? Estimasi 1–2 jam kerja manual.
3. Bolehkan menambah `dataset_ft/` di Drive (perkiraan < 200 MB setelah augmentasi)?
4. Untuk evaluasi akhir: gunakan video UK saat ini, atau rekam/unduh **video plat Indonesia** sebagai uji final (disarankan)?

---

*Disusun berdasarkan: hasil uji 1 & 2, PHASE1–3 SETUP NOTES, PRD 01 §5, PRD 02 §1.1/§3.5, dan mutasi pipeline (upscale 3×, MIN_HEIGHT, folder per-uji, rename video .mp4).*
