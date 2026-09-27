# PRD — Decision: Plat v3 (Uji Domain Baru + Fallback PaddleOCR)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 22 Sep 2026
**Status:** Draft — rencana eksekusi, menunggu hasil uji 5
**Dokumen induk:** PRD_decision_plat.md (v1), PRD_decision_plat_v2.md (v2)
**Melanjutkan:** hasil uji 4 (Locked-Plate Display ✅, akurasi OCR masih kurang)

---

## 0. Ringkasan Eksekutif

Uji 4 berhasil menstabilkan **tampilan** (locked display, 0 flip-flop setelah lock), tetapi
**akurasi bacaan OCR belum memuaskan**. Hipotesis utama (dari analisis terukur): bottleneck
bukan detektor YOLO (mAP50 0,8609 ✅) melainkan **recognizer EasyOCR general-purpose** +
kualitas crop plat video (20–30 px).

Dokumen ini merencanakan **dua langkah berurutan dengan gerbang keputusan (decision gate)**:

1. **Uji 5** — ganti sampel video uji (domain baru) → ukur di mana kegagalan terjadi
   (deteksi vs OCR vs domain).
2. **Uji 6** — *jika* uji 5 menunjukkan recognizer sebagai penyebab utama → ganti engine OCR
   dari EasyOCR ke **PaddleOCR** (PP-OCRv4).

Prinsip: **keputusan berdasarkan angka dari log, bukan tebakan.** Setiap uji punya kriteria
lolos/gagal yang objektif sebelum lanjut ke langkah berikutnya.

---

## 1. Konteks & Baseline Terukur

### 1.1 Posisi saat ini (per uji 4)

| Komponen | Status terukur | Bukti |
| :-- | :-- | :-- |
| Deteksi YOLO (r3, yolov8s) | mAP50 0,8609 ✅; recall 0,864 | `model/best.pt`, PHASE2 notes |
| Cakupan OCR (uji 2, video UK) | 96,4% deteksi menghasilkan teks | log uji 2 |
| Akurasi karakter | Substitusi O↔0, I↔1, Z↔2, G↔6; bacaan 8–9 karakter pada plat 7 karakter | log uji 2–4 |
| Locked-Plate Display (PRD v2) | 0 flip-flop setelah lock ✅ | unit test + simulasi uji 4 |
| Fine-tune recognizer (PRD v1 Fase C) | ⏸️ menunggu pelabelan manual (Fase A.2, 128 crop) | `dataset_ft/labels_draft.csv` |

### 1.2 Kenapa PaddleOCR sebagai kandidat pengganti

| Aspek | EasyOCR `english_g2` | PaddleOCR PP-OCRv4 |
| :-- | :-- | :-- |
| Jenis model | General-purpose scene text | General-purpose, tapi recognizer lebih baru (SVTR) |
| Performa teks pendek terstruktur | Sedang (terbukti banyak substitusi di kasus kita) | Umumnya lebih akurat untuk teks pendek/plat |
| Integrasi | Sudah terpasang, pipeline stabil | `pip install paddleocr paddlepaddle`, ganti ~10 baris |
| Ukuran/model | ~100 MB | ~15–20 MB (model server/mobile tersedia) |
| GPU | Ya | Ya (CPU juga layak) |
| Charset A–Z 0–9 | Via `allowlist` | Via `rec_char_dict_path` / post-filter |
| Fine-tune custom | Didukung (Fase C v1) | Didukung (jika nanti dibutuhkan) |

**Catatan risiko:** PaddleOCR tidak otomatis menyelesaikan masalah — untuk font plat yang
sangat spesifik, fine-tune tetap jalur utama (PRD v1). PaddleOCR adalah **perbaikan cepat
berbiaya rendah** sebelum/sembari menunggu pelabelan.

---

## 2. Keputusan (Decision)

| # | Keputusan | Status |
| :- | :-- | :-- |
| W1 | **Uji 5 memakai video sampel baru** (menggantikan `sample_video_1min_fix.mp4`) untuk mengukur ketangguhan pipeline di domain berbeda | Diputuskan |
| W2 | **`DOMAIN_REGEX` wajib disesuaikan dengan isi video** — video plat Indonesia → `'ID'` (mematikan domain bias UK yang diperkenalkan uji 3 untuk video UK); video UK/campur → `'auto'` | Diputuskan |
| W3 | **Decision gate ke PaddleOCR berdasarkan metrik log uji 5** (§4), bukan kesan visual semata | Diputuskan |
| W4 | **PaddleOCR dipasang sebagai recognizer alternatif yang bisa diganti via 1 flag config** (`OCR_ENGINE = 'easyocr' | 'paddleocr'`) — pipeline deteksi, D3, D4, locked display (V1–V7) **tidak berubah** | Diusulkan |
| W5 | **EasyOCR tidak dihapus** — kedua engine tersedia; hasil uji dibandingkan berdampingan (A/B) pada video yang sama | Diputuskan |
| W6 | **Jika PaddleOCR menang**: ia menjadi default di notebook; jalur fine-tune recognizer (PRD v1 Fase C) **direvisi targetnya** dari EasyOCR ke PaddleOCR | Diusulkan |
| W7 | **Jika keduanya gagal pada metrik yang sama**: kesimpulan bergeser ke kualitas citra/deteksi → prioritaskan preprocessing+ (sharpening/binarisasi/super-resolution) atau Fase B, bukan ganti engine lagi | Diusulkan |
| W8 | Video uji lama (`sample_video_1min_fix.mp4`, UK) **dipertahankan sebagai regression set** — setiap perubahan engine tetap diuji ulang padanya | Diputuskan |

---

## 3. Rencana Kerja

### Fase U5 — Uji 5: Video Sampel Baru (target: 0,5 hari, besok)

1. **Upload** video baru ke `MyDrive/PMLD/test_video/` (nama jelas, mis. `sample_video_id_1.mp4`).
2. **Ubah konfigurasi** notebook (sel 5 & 7):
   ```python
   VIDEO_PATH   = f'{PROJECT_DIR}/test_video/<nama_video_baru>.mp4'
   VIDEO_TAG    = '<nama_video_baru>.mp4'
   RUN_NAME     = f'uji_5_{VIDEO_TAG}'
   DOMAIN_REGEX = 'ID'      # jika video berisi plat Indonesia — WAJIB (mematikan bias UK)
                            # 'auto' jika UK/campuran
   ```
3. **Run all** di Colab (T4).
4. **Analisis log** → isi tabel diagnosis §4.1.

### Fase U6 — Uji 6: PaddleOCR (target: 0,5–1 hari, jika gate §4.2 terpenuhi)

1. **Sel instalasi** tambahan: `pip install paddleocr paddlepaddle` (CPU) / `paddlepaddle-gpu`.
2. **Sel 11**: tambah wrapper recognizer terpadu:
   ```python
   OCR_ENGINE = 'paddleocr'   # 'easyocr' | 'paddleocr'

   def ocr_read(plate_img):
       """Return (raw_text, conf) — antarmuka sama untuk kedua engine."""
       if OCR_ENGINE == 'paddleocr':
           res = paddle_ocr.ocr(plate_img, cls=False)
           # normalisasi hasil → (teks, conf) seperti easyocr
       else:
           res = reader.readtext(plate_img, allowlist=ALLOWLIST)
           ...
   ```
3. **Post-filter charset**: buang karakter di luar `A–Z 0–9` sebelum masuk D3 (PaddleOCR
   kadang mengeluarkan karakter non-ASCII).
4. **A/B test pada video yang sama**:
   - Uji 6a: `OCR_ENGINE='easyocr'`, `RUN_NAME=uji_6a_easyocr_<tag>`
   - Uji 6b: `OCR_ENGINE='paddleocr'`, `RUN_NAME=uji_6b_paddleocr_<tag>`
5. **Bandingkan** metrik §4.2 → keputusan W6/W7.

### Di luar dua fase ini (tidak berubah)
- Fase A.2 pelabelan manual 128 crop (tetap berjalan terpisah — dibutuhkan apa pun engine-nya)
- PRD v2 (locked display) — sudah stabil, jangan disentuh
- Upgrade YOLO / fine-tune detector — **prioritas terakhir**, hanya jika §4.1 membuktikan deteksi yang gagal

---

## 4. Decision Gates (kriteria objektif)

### 4.1 Tabel diagnosis uji 5 — *di mana* kegagalan terjadi?

| Gejala di log uji 5 | Diagnosis | Tindakan |
| :-- | :-- | :-- |
| Deteksi sedikit / bbox meleset (`det_conf` rendah, banyak frame kosong) | **Detektor** gagal generalisasi | Fase B (fine-tune detector) naik prioritas; PaddleOCR percuma |
| Deteksi bagus, `plat` kosong/`ocr_conf` < 0,3 dominan | **Kualitas crop** (blur/kecil) | Preprocessing+ (sharpening, binarisasi, super-resolution) dulu |
| Deteksi bagus, teks ada, tapi substitusi karakter tinggi & exact match rendah | **Recognizer** lemah di domain ini | ✅ Gate terpenuhi → lanjut Fase U6 (PaddleOCR) |
| `plat` sering benar tapi `plat_corrected` merusaknya | **D3 over-correct** di domain baru | Periksa `DOMAIN_REGEX` (W2); revisi template/bias |
| Semua metrik bagus | Domain baru lebih mudah | Dokumentasikan; lanjut rencana semula (pelabelan + fine-tune) |

### 4.2 Metrik pembanding uji 6a vs 6b (video sama, engine beda)

| Metrik | Sumber | Menang jika |
| :-- | :-- | :-- |
| Deteksi menghasilkan teks | log JSONL | PaddleOCR ≥ EasyOCR |
| % lolos regex (domain sesuai video) | log JSONL | PaddleOCR lebih tinggi ≥ 10 poin |
| CER vs ground truth subset (manual, ±20 plat terbaca jelas) | hitung manual dari log + video | PaddleOCR CER turun ≥ 30% relatif |
| Jumlah koreksi D3 yang dibutuhkan (`plat ≠ plat_corrected`) | log JSONL | Lebih sedikit = recognizer lebih kuat |
| Flip-flop setelah lock (PRD v2) | log JSONL | Tetap 0 pada keduanya (regresi check) |
| Runtime per frame | output sel video | PaddleOCR tidak > 1,5× EasyOCR |

**Keputusan akhir:**
- PaddleOCR menang ≥ 3 metrik utama (regex, CER, koreksi D3) → **W6: jadikan default**
- Seri / EasyOCR menang → tetap EasyOCR, fokus ke fine-tune recognizer (PRD v1 Fase C)
- Keduanya sama-sama buruk → **W7**: masalah di kualitas citra, bukan engine

---

## 5. Konfigurasi Referensi (uji 5 & 6)

```python
# ---- Sel 5 (uji 5) ----
VIDEO_PATH   = f'{PROJECT_DIR}/test_video/<video_baru>.mp4'
VIDEO_TAG    = '<video_baru>.mp4'
RUN_NAME     = f'uji_5_{VIDEO_TAG}'

# ---- Sel 7 (uji 5) ----
DOMAIN_REGEX = 'ID'          # WAJIB untuk video plat Indonesia (matikan bias UK)
# DOMAIN_REGEX = 'auto'      # untuk video UK/campuran

# ---- Sel 7 (uji 6) ----
OCR_ENGINE   = 'paddleocr'   # BARU: 'easyocr' | 'paddleocr'
# sisanya TIDAK berubah: CHAR_FIX, TEMPORAL, DISPLAY_LOCK, LOCK_MIN, OVERRIDE_RATIO
```

**Schema log:** tetap v2.1 (tidak ada field baru; engine tercatat lewat `RUN_NAME`).

---

## 6. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| :-- | :-- | :-- |
| `DOMAIN_REGEX='ID'` lupa diset untuk video Indonesia | D3 memaksa template UK → hasil uji 5 menyesatkan | W2 + checklist di sel konfigurasi (print peringatan jika `'auto'` + banyak plat ID) |
| PaddleOCR berat diinstal di Colab (dependency paddle) | Waktu habis di setup | Pin versi; siapkan fallback CPU; estimasi instalasi < 5 menit |
| PaddleOCR membaca karakter non-latin/simbol | Noise masuk D3 | Post-filter charset `A–Z0-9` sebelum `correct_plate_text()` (U6 langkah 3) |
| Hasil PaddleOCR beda format (spasi, kapitalisasi) | D3/D4 salah baca | Normalisasi di wrapper `ocr_read()` — output wajib (teks uppercase tanpa spasi, conf) |
| Video baru platnya lebih kecil dari video UK | `MIN_HEIGHT=20` terlalu longgar/ketat | Evaluasi distribusi tinggi bbox di log uji 5; sesuaikan `MIN_HEIGHT` bila perlu |
| PaddleOCR menang di video baru tapi kalah di video UK | Keputusan engine tidak konsisten | W8: A/B test juga diulang pada video UK (regression set) sebelum final |
| Fine-tune EasyOCR (v1) jadi sia-sia jika pindah engine | Rencana ganda | W6: target fine-tune direvisi ke engine pemenang — data `dataset_ft/` tetap dipakai (format crop + label tidak berubah) |

---

## 7. Target & Kriteria Keberhasilan

| # | Target | Cara ukur |
| :- | :-- | :-- |
| T1 | Uji 5 selesai 100% tanpa error pada video baru | sel video |
| T2 | Diagnosis uji 5 terisi lengkap (tabel §4.1) — diketahui pasti deteksi/OCR/domain yang gagal | analisis log |
| T3 | Jika lanjut PaddleOCR: CER turun ≥ 30% relatif vs EasyOCR pada video yang sama | hitung manual subset |
| T4 | Jika lanjut PaddleOCR: % lolos regex naik ≥ 10 poin | log JSONL |
| T5 | Flip-flop setelah lock tetap 0 di semua uji (regresi PRD v2) | sel rekap |
| T6 | Keputusan engine final terdokumentasi (W6/W7) + revisi rencana fine-tune | update dokumen ini / PHASE notes |

---

## 8. Struktur Workspace Sesudah Fase Ini

```
MyDrive/PMLD/
├── test_video/
│   ├── sample_video_1min.mp4          ← uji 1 (CCTV simpang)
│   ├── sample_video_1min_fix.mp4      ← uji 2–4 (UK, regression set)
│   └── <video_baru>.mp4               ← BARU: uji 5
├── output/
│   ├── uji_4_sample_video_1min_fix.mp4/
│   ├── uji_5_<video_baru>/            ← BARU
│   ├── uji_6a_easyocr_<video_baru>/   ← BARU (jika gate terpenuhi)
│   └── uji_6b_paddleocr_<video_baru>/ ← BARU (jika gate terpenuhi)
├── notebook/ocr_inference.ipynb       ← +flag OCR_ENGINE + wrapper ocr_read
└── docs/
    ├── PRD_decision_plat_v3.md        ← dokumen ini
    └── PHASE4_SETUP_NOTES_UJI5.md     ← BARU setelah uji 5/6
```

---

## 9. Yang Dibutuhkan dari Tim

1. Video sampel baru sudah diputuskan isinya: plat **Indonesia** atau masih asing? (menentukan `DOMAIN_REGEX`)
2. Untuk CER (T3): siapa yang membaca ±20 plat jelas sebagai ground truth perbandingan? (±15 menit manual)
3. Jika PaddleOCR menang — setuju menjadikannya engine default dan merevisi target Fase C (W6)?

---

*Disusun berdasarkan: hasil uji 4 (locked display ✅, akurasi kurang), analisis bottleneck
(detektor ✅ vs recognizer ❌), baseline terukur uji 2 (96,4% cakupan, substitusi karakter tinggi),
dan keputusan user: ganti sampel video dulu, PaddleOCR sebagai fallback berikutnya.*
