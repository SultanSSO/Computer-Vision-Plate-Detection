# Phase 4 — Setup Notes (Uji 3: Stabilisasi OCR)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 21 Sep 2026
**Status:** Notebook uji 3 SIAP dijalankan di Colab (Run all)
**Acuan:** `docs/PRD_decision_plat.md` §4 Fase D, §5, §6

---

## 1. Ringkasan Eksekusi

Notebook `ocr_inference.ipynb` telah diperbarui ke **uji 3** dengan implementasi penuh:

| Fitur PRD | Status | Implementasi |
| :-- | :-- | :-- |
| **D3** Character-position correction | ✅ | `correct_plate_text()` — template posisi ID/UK, substitusi O↔0, I↔1, Z↔2, S↔5, B↔8, G↔6 |
| **D4** Temporal voting | ✅ | `PlateTracker` — greedy IoU tracking + majority vote per track |
| **D7** Schema JSONL v2 | ✅ | Field baru: `plat_corrected`, `domain`, `template`, `track_id`, `vote_count`, `is_stable` |
| **D8** Regex per-domain | ✅ | `validate_plate()` — regex ID & UK terpisah, domain bias UK untuk 7 karakter |

**Fallback otomatis:** Jika `model/best_ft.pt` atau `model/easyocr_ft` belum ada, notebook otomatis menggunakan `best.pt` (r3) + EasyOCR bawaan. **Bisa langsung Run all tanpa menunggu Fase B/C.**

---

## 2. Perubahan dari Uji 2

### 2.1 Sel 1 (Header)
- Judul: Phase 4 — Uji 3
- Tabel fitur D3/D4/D7/D8
- Catatan fallback model

### 2.2 Sel 5 (Path)
- `RUN_NAME = f'uji_3_{VIDEO_TAG}'` → output ke `output/uji_3_sample_video_1min_fix.mp4/`
- `MODEL_PATH` otomatis pilih `best_ft.pt` jika ada, fallback `best.pt`
- `OCR_CUSTOM` otomatis pilih `easyocr_ft` jika ada, fallback bawaan

### 2.3 Sel 7 (Konfigurasi)
Parameter baru:
```python
CHAR_FIX     = True    # D3
TEMPORAL     = True    # D4
VOTE_MIN     = 3       # minimal vote untuk stabil
IOU_TRACK    = 0.25    # ambang tracking
MAX_FRAME_GAP = 15     # frame gap dalam track
DOMAIN_REGEX = 'auto'  # 'ID', 'UK', atau 'auto'
```

### 2.4 Sel 11 (Pipeline inti) — **paling banyak berubah**
Ditambahkan:
- **`correct_plate_text(raw, domain)`** — D3: coba semua template posisi, pilih yang cost terendah, validasi regex, fallback ke teks asli jika tidak valid
- **`validate_plate(text, domain)`** — D8: regex ID `^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$` + regex UK `^[A-Z]{2}\d{2}[A-Z]{3}$`
- **`PlateTracker`** — D4: tracking berbasis IoU + majority vote
- **`format_plate(text, domain)`** — tampilan dengan spasi sesuai domain
- **`detect_and_read()`** diperbarui: kembalikan `plat_corrected`, `domain`, `template`, `track_id`, `vote_count`, `is_stable`
- **`annotate()`** diperbarui: warna berbeda untuk valid (hijau), corrected (oranye), tidak valid (merah), tanda ✓ untuk stabil

### 2.5 Sel 15 (Evaluasi)
- Hitung **CER** (Character Error Rate) dengan Levenshtein distance
- Breakdown per domain (ID vs UK)
- Target: CER ≤ 20%, Char acc ≥ 80%

### 2.6 Sel 17 (Video)
- Inisialisasi `PlateTracker` sebelum loop
- `detect_and_read()` dipanggil dengan `tracker=tracker, frame_idx=idx`
- Log JSONL v2 dengan field tracking

### 2.7 Sel 19 (Rekap)
- Tampilkan plat raw vs corrected
- Tampilkan hasil temporal voting per track (track_id, plat_final, vote_count, stability)
- Breakdown per domain

---

## 3. Contoh Hasil D3 (dari testing lokal)

| Raw OCR | Corrected | Template | Domain | Valid |
| :-- | :-- | :-- | :-- | :-- |
| `GXI50GJ` | `GX15OGJ` | UK LLDDLLL | UK | ✅ |
| `EFIODZT` | `EF10DZT` | UK LLDDLLL | UK | ✅ |
| `6JO6EPD` | `GJ06EPD` | UK LLDDLLL | UK | ✅ |
| `1G65ZFX` | `IG65ZFX` | UK LLDDLLL | UK | ✅ |
| `AF65JKY` | `AF65JKY` | UK LLDDLLL | UK | ✅ |
| `0G65ZFX` | `OG65ZFX` | UK LLDDLLL | UK | ✅ |
| `B1234XY` | `B1234XY` | ID H1A4H2 | ID | ✅ |
| `NL64067` | `NL64067` | — | — | ❌ (tidak valid, dikembalikan asli) |

**Pola yang terlihat:**
- Substitusi I→1, O→0, 6→G, 0→O bekerja dengan benar sesuai posisi template
- Domain bias UK untuk 7 karakter efektif (video uji berisi plat UK)
- Fallback ke teks asli mencegah koreksi paksa yang menghasilkan format tidak valid

---

## 4. Contoh Hasil D4 (dari testing lokal)

```
Track #0: GX15OGJ (votes=4/5, stabilitas=80%, conf=0.86)
```

- Kendaraan yang sama muncul di 5 frame dengan variasi `GX15OGJ`/`GXI50GJ`
- Majority vote memilih `GX15OGJ` (4/5 = 80% stabilitas)
- `is_stable=True` karena `vote_count=4 >= VOTE_MIN=3`

---

## 5. Cara Menjalankan

1. Buka Colab → `MyDrive/PMLD/notebook/ocr_inference.ipynb`
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`
4. Pantau:
   - **Sel 6** — citra test beranotasi (warna: hijau=valid, oranye=corrected, merah=tidak valid)
   - **Sel 7** — CER & char accuracy (target CER ≤ 20%, acc ≥ 80%)
   - **Sel 8** — video 100% tanpa error
   - **Sel 9** — temporal voting: track stabil ≥ 80%

---

## 6. Output yang Diharapkan

```
MyDrive/PMLD/output/uji_3_sample_video_1min_fix.mp4/
├── annotated_images/                  ← citra test + bbox + teks (D3)
├── predictions_txt/
│   ├── test_images_log.jsonl          ← schema v2
│   └── sample_video_1min_fix.mp4_log.jsonl  ← +track_id, vote_count, is_stable
└── processed_video/
    └── sample_video_1min_fix.mp4_annotated.mp4
└── video_captures/
    └── *.jpg                          ← capture tiap 10 detik
```

---

## 7. Checklist Target PRD_decision_plat.md §6

| Metrik | Baseline uji 2 | Target uji 3 | Cara ukur | Status |
| :-- | --: | --: | :-- | :-- |
| Deteksi menghasilkan teks | 96,4% | ≥ 95% (jaga) | log JSONL | ⬜ |
| CER | belum terukur | ≤ 20% | sel 7 | ⬜ |
| Plate accuracy (exact match) | belum terukur | ≥ 80% | sel 7 | ⬜ |
| Lolos regex (per domain) | 16,7% (ID di video UK) | ≥ 70% (UK untuk video fix) | log JSONL | ⬜ |
| Konsistensi antar-frame | rendah | ≥ 80% track stabil | sel 9 | ⬜ |
| Video testing 100% tanpa error | ✅ | ✅ jaga | sel 8 | ⬜ |

---

## 8. File yang Diubah/Dibuat

| File | Aksi | Keterangan |
| :-- | :-- | :-- |
| `notebook/ocr_inference.ipynb` | **Diperbarui** | Uji 3 dengan D3+D4+D7+D8 |
| `scripts/plate_postprocess.py` | **BARU** | Modul post-processing (bisa di-copy MLOps) |
| `docs/PHASE4_SETUP_NOTES.md` | **BARU** | Dokumen ini |

---

## 9. Langkah Setelah Uji 3

| Kondisi | Aksi |
| :-- | :-- |
| CER ≤ 20% & accuracy ≥ 80% | ✅ Tulis `PHASE4_FINETUNE_NOTES.md`, handover ke MLOps |
| CER > 20% atau accuracy < 80% | Perkuat D3/D4, atau siapkan Fase B (fine-tune detector) & C (fine-tune recognizer) |
| Banyak track tidak stabil | Naikkan `VOTE_MIN`, perketat `MIN_OCR_CONF`, atau perbaiki `IOU_TRACK` |
| Domain UK terus dominan | Konfirmasi ke tim: perlu video plat Indonesia untuk evaluasi final? |

---

*Disusun berdasarkan implementasi D3+D4+D7+D8 di `notebook/ocr_inference.ipynb` (uji 3), testing lokal modul `scripts/plate_postprocess.py`, dan acuan `docs/PRD_decision_plat.md`.*
