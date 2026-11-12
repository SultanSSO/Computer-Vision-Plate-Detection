# Fase A.1 — Hasil Ekstraksi Kandidat Crop Plat

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 21 Sep 2026
**Status:** Fase A.1 selesai — 128 crop berlabel-kosong siap untuk ground truth manual
**Acuan:** `docs/PRD_decision_plat.md` §4 Fase A, §5

---

## 1. Ringkasan Eksekusi

| Metrik | Nilai | Keterangan |
| :-- | --: | :-- |
| Crop total | **128** | dari 31 track kendaraan |
| Sumber uji 2 | 128 | 31 track, video `sample_video_1min_fix.mp4` (60 dtk, 1080p) |
| Sumber uji 1 | 0 | 0 track (semua blur total setelah filter kualitas) |
| ocr_conf median | 0,539 | crop referensi ≥ 0,35, crop pendukung ≥ 0,20 |
| Track dengan ocr_conf ≥ 0,80 | 12 | kandidat terbaik untuk evaluasi |

**Catatan uji 1:** Video CCTV simpang 29,6 dtk menghasilkan 0 crop setelah filter. Penyebab: plat terlalu kecil (< 24 px) dan blur gerak; meski 14 track terbentuk, tidak ada deteksi dengan ocr_conf ≥ 0,35. Ini konsisten dengan temuan Phase 3 uji 1 bahwa video tersebut tidak cocok untuk OCR.

---

## 2. Perubahan Kode dari Run Pertama

Run pertama (sebelum filter) menghasilkan **229 crop dari 123 track** — terlalu banyak noise. Perubahan pada `scripts/extract_crops.py`:

1. **`MIN_TRACK_LEN = 3`** — track < 3 frame dibuang (tidak bisa divalidasi antar-frame).
2. **`OCR_CONF_REF_MIN = 0.35`** — track yang crop paling tajamnya < 0,35 dibuang (blur total, tidak bisa dilabeli manusia).
3. **`OCR_CONF_MIN = 0.20`** — crop pendukung harus ≥ 0,20; jika referensi < 0,5, crop pendukung diperketat.
4. **Template label** diperjelas: `UK LLDDLLL` vs `ID H{a}A{b}H{c}` (sebelumnya hanya `LLDDLLL` dan `HAHHH`).

Hasil: 128 crop dari 31 track — lebih bersih, semua track punya minimal 1 crop yang bisa dibaca manusia.

---

## 3. Struktur Output `dataset_ft/`

```
dataset_ft/
├── crops/                    ← 128 file .jpg (grayscale + CLAHE + upscale 3×)
│   ├── uji_2_t000_f0000.jpg
│   ├── uji_2_t000_f0012.jpg
│   └── ...
├── labeling/
│   └── gallery.html          ← galeri per track untuk pelabelan manual
└── labels_draft.csv          ← metadata + kolom gt_text (KOSONG, diisi manual)
```

### Schema `labels_draft.csv`

| Kolom | Isi | Keterangan |
| :-- | :-- | :-- |
| `crop_id` | `uji_2_t001_f0012` | unik per crop |
| `file` | `crops/uji_2_t001_f0012.jpg` | path relatif dari `dataset_ft/` |
| `src` | `uji_2` | sumber video |
| `track_id` | `1` | ID track kendaraan |
| `frame` | `12` | nomor frame |
| `timestamp_s` | `0.4` | detik ke- |
| `bbox` | `1212 819 1297 840` | x1 y1 x2 y2 |
| `det_conf` | `0.6092` | confidence deteksi YOLO |
| `ocr_raw` | `HSISU` | teks mentah EasyOCR |
| `ocr_conf` | `0.6001` | confidence OCR |
| `plat_valid` | `False` | lolos regex ID (belum dipakai untuk UK) |
| `n_frame_track` | `23` | total frame deteksi dalam track |
| `n_variasi_ocr` | `20` | variasi teks OCR dalam track |
| `ocr_top` | `HSISU` | teks dominan (modus) |
| `usul_koreksi` | `H5ISU` | draft koreksi posisi (D3) |
| `template` | `ID H1A2H2` | template yang dipakai |
| **`gt_text`** | **(KOSONG)** | **← DIISI MANUAL** |
| `catatan` | | opsional: `blur`, `ganda`, `non-plat` |

---

## 4. Contoh Track & Draft OCR

| Track | Crop | ocr_conf max | OCR dominan | Usulan koreksi | Template |
| :-- | --: | --: | :-- | :-- | :-- |
| t001 | 6 | 0,85 | `GXI50GJ` | `GX15 OGJ` | ID H2A2H2 |
| t021 | 4 | 0,94 | `EYGINBG` | `EYG1NBG` | ID H3A1H3 |
| t040 | 6 | 0,96 | `EFIODZT` | `EF10DZT` | ID H2A2H3 |
| t047 | 6 | 0,68 | `AF65JKY` | `AF65JKY` | UK LLDDLLL |
| t050 | 6 | 0,94 | `6JO6EPD` | `GJO6EPD` | ID H1A2H3 |
| t068 | 6 | 0,86 | `1G65ZFX` | `IG65ZFX` | ID H1A2H3 |

Pola kesalahan yang terlihat:
- **O↔0, I↔1, Z↔2, G↔6, S↔5** — substitusi karakter visual (akar masalah §1.3 PRD).
- **Variasi antar-frame tinggi** — `6JO6EPD`/`GJO6EPD`/`6J06EPD` untuk kendaraan yang sama (butuh temporal voting D4).
- **Detektor longgar** — `ILHISZZCI` (9 karakter) kemungkinan bbox menangkap teks lain atau plat ganda.

---

## 5. Cara Melabeli Ground Truth

1. Buka `dataset_ft/labeling/gallery.html` di browser.
2. Untuk tiap track, perhatikan:
   - **OCR dominan** (teks kuning) — teks paling sering muncul.
   - **Usulan koreksi** (teks biru) — draft setelah substitusi posisi.
   - **Semua crop dalam track** — bandingkan antar-frame; kendaraan yang sama harus punya plat sama.
3. Isi kolom `gt_text` di `labels_draft.csv` dengan teks plat **persis** yang terlihat di crop paling tajam. Gunakan huruf besar, tanpa spasi.
4. Jika crop blur total dan tidak bisa dibaca: isi `gt_text` = `UNKNOWN`, isi `catatan` = `blur`.
5. Jika bbox bukan plat (mis. bagian bodi): isi `gt_text` = `NON_PLATE`, isi `catatan` = `bukan plat`.
6. Jika plat ganda/duplikat: isi `gt_text` = `GANDA`, isi `catatan` = `plat ganda`.

**Target:** minimal 300–500 crop berlabel. Saat ini 128 crop tersedia; jika perlu tambahan, ekstraksi bisa diulang dari log uji 1 dengan ambang lebih longgar, atau dari citra test dataset.

---

## 6. Langkah Selanjutnya (Fase A.2–A.6)

Setelah `gt_text` diisi:

| Langkah | Isi | Estimasi |
| :-- | :-- | :-- |
| **A.2** | Pelabelan manual ground truth (langkah ini) | 1–2 jam |
| **A.3** | Augmentasi ×3–×5: blur, haze, low-light, rotasi ±5°, noise | 0,5 jam |
| **A.4** | Split train/val (80/20) — pastikan tidak ada kebocoran per video | 0,5 jam |
| **A.5** | Format YOLO untuk detector + `crops/ + labels.csv` untuk recognizer | 0,5 jam |
| **A.6** | Verifikasi: tidak ada crop dari track yang sama di train & val | 0,5 jam |

Setelah Fase A selesai, lanjut ke **Fase B** (fine-tune detector YOLOv8) dan **Fase C** (fine-tune recognizer EasyOCR) di Colab.

---

## 7. File yang Dibuat/Diubah

| File | Aksi | Keterangan |
| :-- | :-- | :-- |
| `scripts/extract_crops.py` | **BARU** | Ekstraksi crop dari log JSONL → track → crop representatif |
| `dataset_ft/crops/*.jpg` | **BARU** | 128 crop plat (grayscale + CLAHE + upscale 3×) |
| `dataset_ft/labels_draft.csv` | **BARU** | Metadata + kolom `gt_text` kosong |
| `dataset_ft/labeling/gallery.html` | **BARU** | Galeri per track untuk pelabelan |
| `dataset_ft/` di Drive | **BELUM** | Menunggu persetujuan §9 PRD + pelabelan selesai |

---

*Disusun berdasarkan eksekusi `scripts/extract_crops.py` terhadap log uji 1 & uji 2, dengan filter kualitas blur (OCR_CONF_REF_MIN ≥ 0,35, MIN_TRACK_LEN ≥ 3).*
