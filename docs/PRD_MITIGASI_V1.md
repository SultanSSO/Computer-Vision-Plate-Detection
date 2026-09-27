# PRD — Mitigasi v1: Jalur Alternatif Darurat (Domain-Specific ALPR)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 23 Sep 2026
**Status:** Draft — menunggu persetujuan & eksekusi M1
**Dokumen induk:** `docs/PRD_01_Project_Overview.md`, `docs/PRD_decision_plat.md` (v1), `PRD_decision_plat_v2.md` (v2), `PRD_decision_plat_v3.md` (v3)
**Root workspace baru:** `MyDrive/PMLD/mitigasi/` *(folder "mitigasi" yang disiapkan user; mirror lokal: `plat-detection/mitigasi/`)*
**Video sumber domain:** `test_video/sample_vidio_1.mp4` — 29,6 detik, 1920×1080, 30 fps, 888 frame (terverifikasi via ffprobe)
**Domain plat:** **UK (Inggris)** — dikonfirmasi user: isi video adalah plat UK, sehingga `DOMAIN_REGEX = 'UK'` dan template D3 yang dipakai adalah `LLDDLLL` (2 huruf + 2 angka + 3 huruf)

---

## 0. Ringkasan Eksekutif

Pipeline utama (`model/best.pt` + EasyOCR) terbukti kuat di deteksi (mAP50 0,8609) tetapi
lemah di pembacaan teks — EasyOCR general-purpose banyak melakukan substitusi karakter
(O↔0, I↔1, Z↔2, G↔6) pada crop plat video. PRD v3 merencanakan perbaikan bertahap
(uji 5 → uji 6 PaddleOCR). **Dokumen ini adalah jalur alternatif darurat yang berjalan
PARALEL dan MANDIRI di folder terpisah**, dengan strategi berbeda:

1. **M1 — Dataset dari domain target:** ekstrak ±250–300 frame dari `sample_vidio_1.mp4`
   (video yang sama dengan yang akan diuji) → anotasi manual → fine-tune YOLOv8s dari
   checkpoint `best.pt` (transfer learning, bukan dari nol).
2. **M2 — Ganti recognizer:** EasyOCR → **PaddleOCR (PP-OCRv4)** di balik flag
   `OCR_ENGINE`, pipeline D3/D4/locked-display tidak disentuh.
3. **M3 — Deliverable final:** video beranotasi (bbox + teks plat) + log JSONL v2.1.

Prinsip: **model dilatih pada distribusi yang sama dengan data ujinya** (same-domain
training), mengeliminasi gap domain yang selama ini menjadi sumber kegagalan OCR.

> ⚠️ **Kesadaran metodologis:** training pada frame dari video yang sama dengan video uji
> berarti metrik akhir adalah **capaian in-domain**, bukan generalisasi. Ini disengaja
> (jalur darurat — targetnya demo/akurasi di kasus nyata ini), dan dinyatakan eksplisit
> di laporan akhir (§10). Untuk klaim generalisasi, regression set video UK lain
> (`sample_video_1min_fix.mp4`) tetap dipakai sebagai kontrol.

---

## 1. Latar Belakang & Justifikasi

### 1.1 Masalah yang diatasi

| Baseline terukur (pipeline utama) | Bukti |
| :-- | :-- |
| Deteksi YOLO r3 (yolov8s): mAP50 0,8609, recall 0,864 ✅ | `PHASE3_SETUP_NOTES.md` |
| Cakupan OCR 96,4%, tapi substitusi karakter tinggi | log uji 2 |
| Lolos regex domain hanya 16,7% (uji 2) | `PHASE4_SETUP_NOTES.md` §7 |
| Locked display stabil (0 flip-flop) ✅ tapi bacaan tetap salah | `PHASE4_SETUP_NOTES_UJI4.md` |

Kesimpulan analisis v3: bottleneck = **recognizer** + **kualitas/domain crop**, bukan detektor.
Tetapi detektor r3 juga belum pernah melihat domain `sample_vidio_1.mp4` secara spesifik.

### 1.2 Kenapa pendekatan ini layak dicoba

| Aspek | Jalur utama (v3) | Jalur mitigasi (dokumen ini) |
| :-- | :-- | :-- |
| Data training detektor | 750 gambar Kaggle (domain umum) | 250–300 frame dari video target (domain identik) |
| Recognizer | EasyOCR, PaddleOCR baru diuji setelah uji 5 | PaddleOCR langsung (langkah M2 sudah diputuskan) |
| Risiko merusak artefak utama | Ada (notebook yang sama) | **Nol** — folder terpisah, `model/best.pt` asli tidak disentuh |
| Waktu ke hasil | 2 fase berurutan + gate | 1 alur linear, bisa dikerjakan langsung |
| Biaya anotasi | 128 crop (recognizer, menunggu) | 250–300 frame (detektor, dikerjakan user) |

### 1.3 Yang TIDAK berubah (batasan eksplisit)

- Artefak utama di `MyDrive/PMLD/` (model, notebook, output uji 1–4) **tidak diubah/dihapus**.
- Modul `scripts/plate_postprocess.py` (D3 koreksi posisi, D4 tracking, locked display V1–V7)
  **dipakai apa adanya** — diimpor/disalin ke mitigasi, bukan ditulis ulang.
- Schema log tetap **JSONL v2.1** (konsisten dengan uji 4) + field tambahan engine OCR (§7.3).
- Locked-Plate Display tetap aktif di output video (regresi PRD v2 wajib: flip-flop = 0).

---

## 2. Tujuan & Kriteria Keberhasilan

| # | Tujuan | Metrik | Target | Cara ukur |
| :- | :-- | :-- | --: | :-- |
| G1 | Dataset domain video terbangun | jumlah frame teranotasi valid | 250–300 (min 200) | skrip verifikasi pasangan gambar-label 1:1 |
| G2 | Detektor fine-tuned mengungguli/menyamai baseline di domain video | mAP50 (val split dataset mitigasi) | ≥ 0,90 (baseline r3 = 0,8609 di domain Kaggle) | output training + `_metrics.json` |
| G3 | Pembacaan teks membaik | % lolos regex UK `^[A-Z]{2}\d{2}[A-Z]{3}$` | ≥ 70% (baseline uji 2: 16,7% dengan bias domain campur) | log JSONL video final |
| G4 | Kualitas karakter membaik | CER pada subset ground truth (±20 plat terbaca jelas, dibaca manual user) | turun ≥ 30% relatif vs log uji 2 | hitung manual dari log |
| G5 | Video final selesai diproses | completion | 100% (888 frame, tanpa error) | sel pipeline video |
| G6 | Stabilitas tampilan terjaga | flip-flop setelah lock | 0 (regresi PRD v2) | rekap tracker |
| G7 | Output lengkap | video beranotasi + JSONL + tabel plat final per track | 100% | cek folder output |

**Kriteria lanjut/berhenti (decision gate akhir):**
- G2 + G3 tercapai → jalur mitigasi **dinyatakan berhasil**; hasil dibandingkan dengan hasil
  uji 5/6 jalur utama (bila sudah ada) untuk keputusan engine/model final.
- G2 tercapai tapi G3 gagal → detektor domain bukan jawaban; masalah murni di recognizer →
  data anotasi tetap berguna untuk jalur utama (Fase B).
- Keduanya gagal → kesimpulan bergeser ke kualitas citra video (resolusi plat terlalu kecil);
  dokumentasikan, jangan ganti engine lagi tanpa preprocessing+.

---

## 3. Struktur Folder Mitigasi

Seluruh artefak baru berada di dalam `MyDrive/PMLD/mitigasi/` — tidak bercampur dengan
struktur utama:

```
MyDrive/PMLD/mitigasi/
├── docs/
│   └── PRD_MITIGASI_V1.md             ← dokumen ini
├── dataset_video/
│   ├── frames_raw/                    ← frame hasil ekstraksi (belum dipilih)
│   ├── images/{train,val,test}/       ← frame terpilih & teranotasi
│   ├── labels/{train,val,test}/       ← anotasi YOLO (manual via LabelImg/CVAT)
│   └── data_mitigasi.yaml             ← konfigurasi dataset (nc:1, license_plate)
├── scripts/
│   ├── extract_frames.py              ← M1.1: cache frame dari video (dedup + blur filter)
│   ├── verify_dataset.py              ← M1.3: cek pasangan 1:1 + validitas label YOLO
│   └── plate_postprocess.py           ← SALINAN dari scripts/ utama (D3/D4/lock, tidak diubah)
├── notebook/
│   ├── m1_finetune_yolov8s.ipynb      ← M1.4: fine-tune detector
│   └── m2_pipeline_paddleocr.ipynb    ← M2–M3: inferensi + PaddleOCR + video final
├── model/
│   └── best_mitigasi.pt               ← hasil fine-tune (BUKAN menimpa model/best.pt utama)
└── output/
    └── m3_final_video/                ← video beranotasi + JSONL + rekap per track
        ├── processed_video/sample_vidio_1_annotated.mp4
        ├── predictions_txt/sample_vidio_1_log.jsonl
        └── final_plates.csv           ← plat final per track (hasil locked voting)
```

**Aturan sinkronisasi:** sama dengan proyek utama — edit lokal di
`plat-detection/mitigasi/`, sinkron ke Drive, eksekusi di Colab. Setiap fase dicatat di
`docs/` mitigasi (lihat §9) dengan status md5 sinkron seperti PHASE notes utama.

---

## 4. Fase M1 — Dataset dari Video & Fine-Tune Detector

### 4.1 M1.1 Ekstraksi frame (`extract_frames.py`)

Mekanisme (parameter dapat disetel, default dalam kurung):

| Langkah | Aturan | Alasan |
| :-- | :-- | :-- |
| Sampling | 1 frame tiap N frame (`STRIDE=3` → ±296 kandidat dari 888 frame) | 250–300 target tercapai tanpa duplikasi berlebihan |
| Dedup visual | buang frame dengan MSE/perceptual-hash terlalu mirip frame tersimpan sebelumnya (ambang similarity) | frame berurutan nyaris identik → anotasi sia-sia |
| Blur filter | variance of Laplacian di bawah ambang → skip | frame blur tidak layak anotasi & melatih noise |
| Penamaan | `sv1_f<frame_idx:05d>.jpg` (mis. `sv1_f00123.jpg`) | traceable kembali ke timestamp video |

Output: `dataset_video/frames_raw/` ±250–300 gambar + `frames_raw/_manifest.csv`
(frame_idx, timestamp_s, blur_score, kept/rejected).

### 4.2 M1.2 Anotasi manual (oleh user)

- Tool bebas (disarankan **LabelImg** mode YOLO, atau CVAT) — output label format YOLO
  `class cx cy w h` ternormalisasi.
- **Kelas tunggal:** `license_plate` (konsisten dengan `data.yaml` utama).
- Pedoman anotasi:
  1. Box rapat ke tepi plat (bukan bumper), semua plat yang terlihat dianotasi walau kecil —
     tandai plat < 20 px tinggi tetap dianotasi (model belajar keberadaannya; OCR-lah yang skip).
  2. Plat terpotong tepi frame tetap dianotasi pada bagian yang terlihat.
  3. Jangan anotasi teks non-plat (stiker, tulisan bodi kendaraan).
- **Target: 250–300 frame teranotasi** (minimum 200 untuk lanjut ke M1.4).

### 4.3 M1.3 Verifikasi & split (`verify_dataset.py`)

- Cek pasangan gambar-label 1:1, 0 mismatch; label valid (5 kolom, nilai 0–1, w/h > 0).
- Split deterministik **70/20/10, seed 42** → `images|labels/{train,valtest}`.
- Generate `data_mitigasi.yaml` (path absolut Colab, `nc: 1`, `names: ['license_plate']`).

### 4.4 M1.4 Fine-tune YOLOv8s (`m1_finetune_yolov8s.ipynb`)

| Parameter | Nilai | Alasan |
| :-- | :-- | :-- |
| Base weights | `MyDrive/PMLD/model/best.pt` (r3, yolov8s) | transfer learning — detektor sudah paham "plat", tinggi adaptasi domain |
| Epochs | 50, `patience=10` (early stop) | dataset kecil → cepat overfit; early stop wajib |
| IMG_SIZE | 640 (sama dengan training utama) | konsistensi pipeline inferensi |
| BATCH | 16 | T4 GPU aman |
| `save_period` | 5 → checkpoint ke `mitigasi/model/` | anti sesi Colab putus (risiko PRD 01) |
| SEED | 42 | reprodusibilitas |

Evaluasi di akhir: mAP50 / mAP50-95 / P / R pada **val split dataset mitigasi** + simpan
`_metrics.json` + salin bobot terbaik ke `model/best_mitigasi.pt`.

---

## 5. Fase M2 — PaddleOCR sebagai Recognizer

### 5.1 Integrasi (flag tunggal, mengikuti keputusan W4 PRD v3)

```python
OCR_ENGINE = 'paddleocr'   # 'paddleocr' | 'easyocr' (fallback tetap ada)
```

- Instalasi di notebook: `pip install paddleocr paddlepaddle` (pin versi; fallback CPU
  bila GPU build bermasalah — estimasi < 5 menit, mitigasi risiko v3 §6).
- Wrapper `ocr_read(plate_img) -> (teks_upper_tanpa_spasi, conf)` — antarmuka identik untuk
  kedua engine, sehingga D3/D4/lock tidak tahu engine apa di belakangnya.
- **Post-filter charset:** buang karakter di luar `A–Z 0–9` sebelum masuk
  `correct_plate_text()` (D3) — PaddleOCR kadang mengeluarkan non-ASCII.
- Preprocessing crop **tidak berubah**: crop + padding 10% → grayscale → CLAHE(2.0) →
  upscale 3× → `MIN_HEIGHT=20` (sama dengan uji 2 agar perbandingan adil).

### 5.2 DOMAIN_REGEX — sudah ditetapkan: `'UK'`

User telah mengonfirmasi bahwa `sample_vidio_1.mp4` berisi **plat UK**, sehingga
(keputusan W2 v3) konfigurasi domain dikunci:

- `DOMAIN_REGEX = 'UK'` — regex `^[A-Z]{2}\d{2}[A-Z]{3}$`, template D3 `LLDDLLL`.
- Bias domain UK untuk teks 7 karakter di `correct_plate_text()` kini **konsisten dengan
  isi video sebenarnya** (di uji 2 bias UK aktif lewat `'auto'` — di sini eksplisit).
- Checklist di sel konfigurasi tetap ada: **print peringatan** bila mayoritas kandidat
  justru cocok template ID (indikasi video/setting tertukar).

---

## 6. Fase M3 — Pipeline Video Final & Output

Alur per frame (menggunakan komponen yang sudah teruji — bukan kode baru):

```
frame (tiap 1 dari 3, FRAME_SKIP=3)
  → YOLO best_mitigasi.pt (conf ≥ 0,35)
  → crop + padding 10% → grayscale + CLAHE → upscale 3×
  → ocr_read() [PaddleOCR] → post-filter A–Z0–9
  → correct_plate_text() (D3, domain UK — §5.2)
  → PlateTracker.update() (D4: IoU 0,25, vote, lock_min=3, override_ratio=2.0)
  → annotate(): bbox + plat_display (hijau|=locked / oranye..=sementara / merah=invalid)
  → tulis baris JSONL v2.1
  → frame antara (skip) memakai anotasi terakhir (sesuai PRD 02 §3.5)
```

Output di `output/m3_final_video/`:
1. `processed_video/sample_vidio_1_annotated.mp4` — video 888 frame beranotasi penuh.
2. `predictions_txt/sample_vidio_1_log.jsonl` — 1 baris per deteksi (schema §7.3).
3. `final_plates.csv` — rekap per track: `track_id, plat_final, locked, vote_count,
   avg_conf, override_events` (hasil `get_final_plates()`).

---

## 7. Spesifikasi Teknis

### 7.1 Lingkungan eksekusi

| Komponen | Spesifikasi |
| :-- | :-- |
| Runtime | Google Colab, GPU T4 (`Runtime → Change runtime type → T4`) |
| Root | `PROJECT_DIR = /content/drive/MyDrive/PMLD/mitigasi` |
| Dependensi inti | `ultralytics`, `paddleocr`, `paddlepaddle`, `opencv-python`, `easyocr` (fallback saja) |
| Model dasar | `MyDrive/PMLD/model/best.pt` (read-only — tidak pernah ditimpa) |

### 7.2 Konfigurasi terpusat (sel konfigurasi notebook M2/M3)

```python
MODEL_PATH   = f'{PROJECT_DIR}/model/best_mitigasi.pt'
OCR_ENGINE   = 'paddleocr'
DOMAIN_REGEX = 'UK'            # DITETAPKAN: sample_vidio_1.mp4 berisi plat UK (§5.2)
CONF_DET     = 0.35
PADDING      = 0.10
UPSCALE      = 3
MIN_HEIGHT   = 20
FRAME_SKIP   = 3
CHAR_FIX     = True            # D3
TEMPORAL     = True            # D4
DISPLAY_LOCK = True            # PRD v2
LOCK_MIN     = 3
OVERRIDE_RATIO = 2.0
SEED         = 42
```

### 7.3 Schema log (JSONL v2.1 + 1 field)

```json
{
  "file": "sample_vidio_1.mp4",
  "frame": 42,
  "timestamp_s": 1.4,
  "bbox": [820, 510, 980, 560],
  "det_conf": 0.87,
  "plat": "GXI50GJ",
  "plat_corrected": "GX15OGJ",
  "plat_display": "GX15OGJ",
  "display_locked": true,
  "domain": "UK",
  "template": "UK LLDDLLL",
  "track_id": 3,
  "vote_count": 5,
  "ocr_conf": 0.91,
  "ocr_engine": "paddleocr"
}
```

Field `ocr_engine` satu-satunya tambahan — backward compatible dengan parser uji 4.

---

## 8. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| :-- | :-- | :-- |
| Plat di video terlalu kecil/blur → anotasi & OCR sama-sama sulit | G3/G4 gagal | Blur filter M1.1; hitung distribusi tinggi bbox setelah anotasi — bila median < 20 px, laporkan sebagai batasan data, bukan kegagalan model |
| Dataset 250 frame terlalu kecil → overfit fine-tune | mAP val tinggi tapi video jelek | Early stop + `patience=10`; augmentasi default Ultralytics aktif; transfer learning dari best.pt (bukan from-scratch) |
| Frame mirip semua (kendaraan sama berulang) → dataset tidak beragam | generalisasi intra-video rendah | Dedup M1.1 + stride; bila video memang repetitif, tambah STRIDE dan terima jumlah lebih sedikit tapi beragam |
| PaddleOCR berat diinstal / konflik dependensi di Colab | waktu habis di setup | Pin versi; fallback CPU; jika tetap gagal → fallback EasyOCR via flag (G2 tetap bisa diukur terpisah dari G3) |
| PaddleOCR keluarkan karakter non-latin | noise masuk D3 | post-filter charset wajib (§5.1) |
| `DOMAIN_REGEX` salah set | D3 over-correct, hasil menyesatkan | Domain sudah dikunci `'UK'` (§5.2, dikonfirmasi user); checklist peringatan otomatis tetap aktif bila kandidat ID mendominasi |
| Sesi Colab putus saat training | kehilangan progress | `save_period=5` ke Drive (mitigasi standar PRD 01 §6) |
| Hasil in-domain disalahartikan sebagai generalisasi | klaim berlebihan | disclaimer §0 + §10; regression set video UK wajib ikut diuji bila model ini diusulkan jadi default |
| Waktu anotasi manual molor (target 1–2 hari) | seluruh jadwal mundur | minimum viable: 200 frame boleh lanjut M1.4; sisanya menyusul sebagai tambahan data |

---

## 9. Rencana Eksekusi & Estimasi

| # | Fase | Aktivitas | Output | Estimasi | Penanggung jawab |
| :- | :-- | :-- | :-- | :-- | :-- |
| 1 | M1.1 | Jalankan `extract_frames.py` di Colab | `frames_raw/` + manifest | 0,5 jam | AI Engineer |
| 2 | M1.2 | Anotasi manual 250–300 frame | label YOLO | 1–2 hari | **User** |
| 3 | M1.3 | Verifikasi + split + `data_mitigasi.yaml` | dataset final | 0,5 jam | AI Engineer |
| 4 | M1.4 | Fine-tune YOLOv8s (Colab T4) | `best_mitigasi.pt` + `_metrics.json` | 1–2 jam | AI Engineer |
| 5 | M2 | Notebook pipeline + wrapper PaddleOCR | `m2_pipeline_paddleocr.ipynb` | 0,5 hari | AI Engineer |
| 6 | M3 | Run video final + evaluasi (G3–G7) | video + JSONL + CSV + rekap | 1 jam | AI Engineer |
| 7 | — | Tulis `docs/MITIGASI_V1_NOTES.md` (hasil vs target, keputusan gate §2) | catatan fase | 0,5 hari | AI Engineer |

**Total estimasi: 2–4 hari** (didominasi anotasi manual).

---

## 10. Batasan & Disclaimer

1. **Metrik bersifat in-domain.** Dataset training berasal dari video yang sama dengan video
   uji → angka G2–G4 mengukur keberhasilan adaptasi domain, bukan generalisasi ke video lain.
   Klaim generalisasi hanya sah lewat regression set (`sample_video_1min_fix.mp4`) dan/atau
   video baru.
2. **Ground truth CER (G4) dibaca manual** (±20 plat terbaca jelas) oleh user — subjektivitas
   diminimalkan dengan hanya memilih plat yang "terbaca jelas" di video.
3. Jalur ini **bukan pengganti** rencana jalur utama (v3: uji 5/6 + fine-tune recognizer
   Fase C); keduanya berjalan paralel dan hasil akhirnya dibandingkan sebelum handover ke MLOps.
4. `best_mitigasi.pt` tidak otomatis menjadi deliverable `model/best.pt` — keputusan itu
   hanya melalui perbandingan terdokumentasi (gate §2 + regression test).

---

## 11. Yang Dibutuhkan / Dikonfirmasi dari User

| # | Pertanyaan | Status |
| :- | :-- | :-- |
| 1 | Isi plat di `sample_vidio_1.mp4`: Indonesia, UK, atau campuran? | ✅ **TERJAWAB: plat UK** → `DOMAIN_REGEX='UK'` (§5.2) |
| 2 | Tool anotasi pilihan: LabelImg (lokal) atau CVAT (web)? | ⬜ Sebelum M1.2 |
| 3 | Setuju target anotasi 250–300 dengan minimum 200 untuk lanjut? | ⬜ Sekarang |
| 4 | Siapa membaca ±20 plat ground truth untuk CER (G4)? (default: user, ±15 menit) | ⬜ Saat M3 |
| 5 | Nama folder Drive: `mitigasi` | ✅ **TERJAWAB** — sesuai struktur §3 |

---

*Disusun berdasarkan: baseline terukur uji 2–4 (detektor ✅, recognizer ❌), PRD_decision_plat
v1–v3, mekanisme kerja pipeline existing (D3/D4/locked display di `scripts/plate_postprocess.py`),
dan keputusan user: dataset dari frame `sample_vidio_1.mp4` (250–300 frame, anotasi manual,
plat UK) + fine-tune YOLOv8s + PaddleOCR + output video bbox & JSON.*
