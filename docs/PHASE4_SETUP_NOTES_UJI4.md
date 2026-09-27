# Phase 4 — Setup Notes (Uji 4: Locked-Plate Display)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 22 Sep 2026
**Status:** Notebook uji 4 SIAP dijalankan di Colab (Run all)
**Acuan:** `docs/PRD_decision_plat_v2.md` (V1–V7), `docs/PRD_decision_plat.md` (D3/D4/D7/D8)

---

## 1. Ringkasan Eksekusi

Notebook `ocr_inference.ipynb` diperbarui ke **uji 4** — implementasi penuh PRD v2 (Locked-Plate Display):

| Fitur PRD v2 | Status | Implementasi |
| :-- | :-- | :-- |
| **V1** Locked-Plate Display | ✅ | `tracker.best(track_id)` → `plat_display` per deteksi; `annotate()` menampilkan ini, bukan `plat_corrected` per-frame |
| **V2** Lock rule | ✅ | Track terkunci setelah `vote_count ≥ LOCK_MIN` (default 3); sebelum itu tampil kandidat sementara |
| **V3** Override rule | ✅ | Kunci berubah hanya jika kandidat baru ≥ `OVERRIDE_RATIO` (2×) vote + avg conf lebih tinggi; tercatat di `override_events` |
| **V4** Schema JSONL v2.1 | ✅ | Field `plat_display`, `display_locked` (backward compatible) |
| **V5** Video-only | ✅ | Uji citra (sel 13) tidak berubah |
| **V6** Indikator visual | ✅ | Hijau `\|` = LOCKED · oranye `..` = belum stabil · merah = tidak valid |
| **V7** Model tidak disentuh | ✅ | Murni post-processing; `best.pt`/parameter D3/D4 tidak berubah |

**Akar masalah yang diperbaiki** (dari analisis kode uji 3): `annotate()` menampilkan
`plat_corrected` per-frame; hasil voting D4 hanya dipakai di rekap sel 19 — tidak pernah
dialirkan ke tampilan video. Uji 4 menutup celah itu.

---

## 2. Verifikasi Lokal (sebelum Colab)

### 2.1 Unit test `scripts/plate_postprocess.py` — 5/5 LULUS

| Test | Skenario | Hasil |
| :-- | :-- | :-- |
| 1 | Lock setelah 3 vote; sebelumnya kandidat sementara | ✅ lock tepat di vote ke-3 |
| 2 | Setelah lock, 3 bacaan salah beruntun (`GX1SOGJ`, `GXI50GJ`, `6X15OGJ`) | ✅ **0 flip-flop** — tampilan tetap `GX15OGJ` |
| 3 | Kunci salah (`GJ06EPD`) + lawan 6 vote conf lebih tinggi | ✅ override ke `GJO6EPD`, tercatat di `override_events` |
| 4 | Deteksi tanpa teks (ocr_conf 0) pada track terkunci | ✅ tampilan tidak rusak |
| 5 | `get_final_plates()` konsisten dengan tampilan terkunci | ✅ |

### 2.2 Simulasi end-to-end (kode persis dari sel 11 notebook)

```
frame | ocr_frame   | plat_display | locked
  0   | GXI50GJ     | GX15OGJ      | False     ← kandidat sementara (D3 sudah koreksi)
  3   | GX15OGJ     | GX15OGJ      | False
  6   | GX15OGJ     | GX15OGJ      | True      ← LOCK di vote 3
 12   | GX1SOGJ     | GX15OGJ      | True      ← bacaan salah diabaikan
 15   | GXI50GJ     | GX15OGJ      | True      ← bacaan salah diabaikan
 21   | 6X15OGJ     | GX15OGJ      | True      ← bacaan salah diabaikan

flip-flop setelah lock: 0 (target PRD v2 §6 tercapai)
```

### 2.3 Syntax check notebook — 8/8 sel kode OK, JSON valid

---

## 3. Perubahan Notebook (uji 3 → uji 4)

| Sel | Perubahan |
| :-- | :-- |
| 1 | Header uji 4 + tabel V1–V7 + legenda indikator visual |
| 5 | `RUN_NAME = uji_4_sample_video_1min_fix.mp4` |
| 7 | Tambah config: `DISPLAY_LOCK=True`, `LOCK_MIN=3`, `OVERRIDE_RATIO=2.0` |
| 11 | `PlateTracker`: state `locked_plate`, metode `best()`, `override_events`; `update()` mengisi `plat_display` + `display_locked`; `annotate()` pakai `plat_display` + skema warna V6; `get_final_plates()` melaporkan status lock |
| 17 | Tracker dibuat dengan `lock_min` + `override_ratio`; laporan override events di akhir sel |
| 19 | **Metrik otomatis PRD v2 §6**: hitung transisi `plat_display` per track, pisahkan flip-flop setelah lock (target 0); rekap track LOCKED vs sementara |
| 20 | Ringkasan uji 4 + schema v2.1 + legenda tampilan |

---

## 4. Cara Menjalankan

1. Buka Colab → `MyDrive/PMLD/notebook/ocr_inference.ipynb`
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`
4. Yang perlu dipantau:
   - **Sel 17** — video selesai 100% + laporan override (bila ada)
   - **Sel 19** — metrik utama: **"Flip-flop SETELAH lock: 0" → TERCAPAI ✅**
   - Video hasil: teks plat tidak berubah-ubah lagi pada kendaraan yang sama;
     label hijau `\|` = terkunci, oranye `..` = masih mengumpulkan bukti

---

## 5. Output

```
MyDrive/PMLD/output/uji_4_sample_video_1min_fix.mp4/
├── annotated_images/
├── predictions_txt/
│   ├── test_images_log.jsonl                 ← schema v2
│   └── sample_video_1min_fix.mp4_log.jsonl   ← schema v2.1 (+plat_display, display_locked)
├── processed_video/
│   └── sample_video_1min_fix.mp4_annotated.mp4
└── video_captures/
```

---

## 6. Checklist Target PRD v2 §6

| Metrik | Target | Status |
| :-- | :-- | :-- |
| Flip-flop tampilan setelah lock | **0** | ✅ terverifikasi lokal (unit test + simulasi); konfirmasi Colab di sel 19 |
| Track terkunci = hasil voting akhir | 100% | ✅ `get_final_plates()` pakai `locked_plate` |
| Override tercatat (audit V3) | ada bila terjadi | ✅ `override_events` + laporan di sel 17 |
| CER/akurasi tidak menurun vs uji 3 | regresi check | ⬜ lihat sel 15 setelah Run all |
| Video 100% tanpa error | ✅ jaga | ⬜ lihat sel 17 |

---

## 7. Catatan & Batasan

1. **Metrik sel 15 (citra test) tidak berubah** — V5: locked display hanya untuk video.
   Regresi check CER/akurasi tetap dijalankan untuk memastikan tidak ada side effect.
2. **`LOCK_MIN = 3` dengan `FRAME_SKIP = 3`** berarti sebuah plat terkunci setelah ±3 frame
   terproses (~0,3 detik video 30 fps) — cukup cepat untuk demo, tetap bisa dinaikkan bila
   lock terlalu dini di video nyata.
3. **Override hanya saat kandidat baru ≥ 2× vote + conf lebih tinggi** — plat yang salah
   terkunci tetap bisa terkoreksi, tapi tidak gampang flip-flop.
4. **Jalur PRD v1 (fine-tune detector/recognizer) tetap terpisah** — menunggu pelabelan
   manual Fase A.2 (128 crop di `dataset_ft/labels_draft.csv`).

---

## 8. File yang Diubah

| File | Aksi | Sinkron Drive |
| :-- | :-- | :-- |
| `notebook/ocr_inference.ipynb` | Uji 4 (V1–V7) | ✅ md5 identik |
| `scripts/plate_postprocess.py` | `PlateTracker.best()` + lock/override | ✅ md5 identik |
| `docs/PRD_decision_plat_v2.md` | PRD (sudah ada) | ✅ |
| `docs/PHASE4_SETUP_NOTES_UJI4.md` | Dokumen ini | ✅ |

---

*Disusun berdasarkan implementasi PRD_decision_plat_v2.md di notebook uji 4, unit test lokal
(5/5 lulus), dan simulasi end-to-end dengan kode persis dari sel 11 notebook.*
