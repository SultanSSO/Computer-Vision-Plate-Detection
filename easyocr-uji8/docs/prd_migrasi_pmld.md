# PRD — Migrasi Workspace PMLD: Fine-Tuning Deteksi Plat Nomor Indonesia (Angle Samping)

| | |
| :-- | :-- |
| **Versi** | 1.0 |
| **Status** | Draft |
| **Referensi** | `kajian_migrasi_pmld.md` |
| **Workspace sumber** | Google Drive `PMLD/` (lama) |
| **Workspace target** | Google Drive `MyDrive/PMLD_FT/` (baru, ~0,5 GB) |

---

## 1. Ringkasan

Proyek PMLD (Proyek Mandiri Lintas Disiplin) bertujuan membangun sistem computer vision untuk deteksi dan pembacaan teks plat nomor kendaraan (ALPR). Model YOLO saat ini sudah mencapai performa baik pada video uji plat luar negeri dengan kamera frontal (mAP50 0,8609), tetapi **gagal pada video plat Indonesia yang diambil dari sudut samping/miring** — deteksi halusinasi (false positive) dan hasil OCR banyak error.

Dokumen ini mendefinisikan kebutuhan untuk **migrasi workspace ke struktur baru yang rapi dan ringan**, serta **rencana fine-tuning** agar model akurat pada domain plat Indonesia angle samping, dengan seluruh hasil eksperimen dapat dilihat langsung di Google Colab tanpa perlu download.

## 2. Latar Belakang & Masalah

| # | Masalah | Bukti |
| :-- | :-- | :-- |
| P1 | Model kesulitan membaca plat Indonesia dari angle samping; sering halusinasi (objek non-plat ikut terdeteksi) dan OCR error | Uji pada video baru `plat_indo_samping` |
| P2 | Hasil training/fine-tuning tidak bisa dinilai langsung di Colab; harus download dulu, proses ribet | Alur kerja saat ini |
| P3 | Inferensi memakan ~11 GB VRAM — berat untuk dijalankan lokal oleh tim ML/backend | Pengukuran pada runtime T4 Colab |
| P4 | Workspace lama berantakan dan terlalu besar; campur aduk input, output histori, cache, dan artefak | Struktur `PMLD/` saat ini |

## 3. Tujuan & Success Metrics

| # | Tujuan | Ukuran Keberhasilan |
| :-- | :-- | :-- |
| G1 | Model akurat mendeteksi & membaca plat Indonesia dari angle miring | Peningkatan metrik vs `baseline_metrics/` (keputusan berdasarkan angka dari log, bukan tebakan); OCR teks valid pada uji `plat_indo_samping` |
| G2 | Hasil training/inferensi tampil inline di Colab | 100% hasil uji (video, gambar bbox, tabel metrik) dirender via matplotlib — download hanya saat hasil sudah final |
| G3 | Workspace baru rapi, ringan, terstruktur | Total ≤ ~0,5 GB; hanya berisi input eksperimen; output mulai bersih dari `uji_8` |
| G4 | Inferensi lebih hemat VRAM | Konsumsi VRAM turun signifikan dari 11 GB (via downscale, frame skipping, OCR selektif) |

## 4. Ruang Lingkup

### In-Scope
1. Migrasi item terpilih dari `PMLD/` → `MyDrive/PMLD_FT/` sesuai tabel keputusan (§6).
2. Adaptasi notebook training → `02_yolo_finetune.ipynb` (mulai dari `best.pt`, checkpoint ke Drive, run-name baru).
3. Adaptasi notebook inferensi → `03_inference_eval.ipynb` (render inline + hemat VRAM).
4. Notebook baru `01_extract_frames.ipynb` untuk ekstraksi & sampling frame video baru (jalur mitigasi same-domain training).
5. Modul baru `scripts/line_crossing.py` — Line Crossing Detection: deteksi plat hanya aktif saat kendaraan melewati garis yang ditetapkan (on/off via flag config).
6. Video uji baru `test_video/plat_indo_samping.mp4` (upload user).

### Out-of-Scope
- `model/last.pt`, `runs/`, `output/` lama (uji 1–7c), `__pycache__`, cache label — ditinggal di workspace lama sebagai arsip.
- `notebook/gradio_ui.ipynb` (UI demo tidak relevan untuk eksperimen).
- `test_video/sample_vidio_2.mp4` (137 MB; tetap aman di workspace lama).
- Perubahan arsitektur model (tetap YOLOv8s fine-tune, bukan training dari nol).

## 5. Kebutuhan Fungsional

| ID | Kebutuhan | Prioritas |
| :-- | :-- | :-- |
| FR-1 | Sistem dapat memuat dataset lama (750 gambar: 525 train / 150 val / 75 test) via `data.yaml` yang path-nya mengarah ke `PMLD_FT/dataset/` | Must |
| FR-2 | `01_extract_frames.ipynb` mengekstrak & menyampling frame dari video baru ke `dataset/dataset_ft/` untuk dianotasi | Must |
| FR-3 | `02_yolo_finetune.ipynb` melakukan fine-tune dari `model/best.pt` dengan data gabungan (dataset lama + `dataset_ft`), batch size kecil, checkpoint tersimpan inkremental ke Drive | Must |
| FR-4 | `03_inference_eval.ipynb` membaca video langsung dari Drive, memproses dengan pipeline `alpr_pipeline.py`, dan menulis hasil ke `output/uji_N/` di Drive | Must |
| FR-5 | Semua hasil (video beranotasi, gambar bbox, tabel metrik) dirender inline di Colab — evaluasi tanpa download | Must |
| FR-6 | Mode hemat VRAM pada inferensi: downscale resolusi frame, frame skipping, OCR hanya pada frame terpilih, `imgsz` disesuaikan | Must |
| FR-7 | Line Crossing Detection (`line_crossing.py`): deteksi plat aktif hanya saat melewati garis vertikal/horizontal yang ditetapkan, berhenti di garis akhir; dapat on/off via flag config | Should |
| FR-8 | Voting D4 + locked display (`plate_postprocess.py`) tetap dipakai sebagai aturan tampilan hasil OCR | Must |
| FR-9 | `extract_crops.py` menghasilkan capture gambar bbox sebagai bagian deliverable | Must |
| FR-10 | Setiap uji baru (`output/uji_N/`) mencantumkan perbandingan metrik terhadap `model/baseline_metrics/` | Should |

## 6. Struktur Folder Target

**Root baru:** `MyDrive/PMLD_FT/` — estimasi total ~0,5 GB.

```
MyDrive/PMLD_FT/
├── data.yaml
├── dataset/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset_ft/
├── model/
│   ├── best.pt
│   └── baseline_metrics/
├── notebook/
│   ├── 01_extract_frames.ipynb
│   ├── 02_yolo_finetune.ipynb
│   ├── 03_inference_eval.ipynb
│   └── alpr_pipeline.py
├── scripts/
│   ├── plate_postprocess.py
│   ├── extract_crops.py
│   └── line_crossing.py
├── test_video/
│   ├── plat_indo_samping.mp4      (diupload user, belum ada di Drive)
│   └── sample_vidio_1.mp4
├── docs/
└── output/                        (kosong — terisi mulai uji_8)
```

### Keputusan Copy / Tidak

| Item | Keputusan | Alasan |
| :-- | :-- | :-- |
| `data.yaml` | COPY + sesuaikan path | Wajib training; path lama menunjuk ke folder PMLD lama |
| `dataset/images/` (750 gambar) | COPY | Basis transfer learning; ~0,3 GB |
| `dataset/labels/` | COPY, kecuali `*.cache` | Cache lama berisiko mismatch; YOLO membuat ulang otomatis |
| `dataset/dataset_ft/` | BARU | Output `01_extract_frames.ipynb` untuk anotasi domain baru |
| `model/best.pt` | COPY | Checkpoint awal fine-tune (yolov8s-r3, mAP50 0,8609) |
| `model/last.pt` | TIDAK | Redundan; tidak dipakai resume |
| `model/*_metrics.json, results.csv` | COPY (ke `baseline_metrics/`) | Pembanding sebelum vs sesudah fine-tune |
| `model/*_confusion_matrix.png, results.png` | TIDAK | Masih bisa dilihat di workspace lama |
| `notebook/yolo_training.ipynb` | COPY → adaptasi jadi `02_yolo_finetune.ipynb` | Template training |
| `notebook/ocr_inference*.ipynb` | COPY → adaptasi jadi `03_inference_eval.ipynb` | Pipeline inferensi video |
| `notebook/gradio_ui.ipynb` | TIDAK | Tidak relevan untuk eksperimen |
| `notebook/alpr_pipeline.py` | COPY | Inti logika deteksi+OCR; ditambah integrasi line-crossing |
| `scripts/plate_postprocess.py` | COPY | Voting D4 + locked display (unit test 5/5 lulus) |
| `scripts/extract_crops.py` | COPY | Deliverable capture bbox |
| `scripts/line_crossing.py` | BARU | Gerbang deteksi via garis (saran user) |
| `test_video/plat_indo_samping.mp4` | BARU (upload user) | Video uji domain target |
| `test_video/sample_vidio_1.mp4` | COPY (18,5 MB) | Kontrol regresi |
| `test_video/sample_vidio_2.mp4` | TIDAK (137 MB) | Hanya relevan untuk histori uji 7 |
| `docs/` | COPY | Konteks keputusan; < 10 MB |
| `runs/`, `output/` lama, `__pycache__/` | TIDAK | Regenerable / arsip di workspace lama |

### Korelasi Antar Folder

1. **`data.yaml` ↔ `dataset/` ↔ `model/best.pt`** — triplet wajib training; salah satu hilang/path salah → training gagal di verifikasi awal.
2. **`test_video/` → `notebook/03` → `output/`** — video diproses langsung dari Drive, hasil ditulis ke Drive, dinilai inline.
3. **`notebook/01` → `dataset/dataset_ft/` → `notebook/02`** — alur mitigasi same-domain: ekstraksi frame → anotasi → fine-tune data gabungan.
4. **`scripts/` → `notebook/`** — pipeline memanggil `plate_postprocess.py` dan `line_crossing.py`; tanpa scripts, inferensi rusak walau model ada.
5. **`model/baseline_metrics/` → `output/uji_N/`** — setiap uji baru dibandingkan dengan baseline lama.

## 7. Kebutuhan Non-Fungsional

| ID | Kebutuhan |
| :-- | :-- |
| NFR-1 | Total workspace baru ≤ ~0,5 GB |
| NFR-2 | Inferensi berjalan di GPU T4 Colab dengan VRAM < 11 GB (target turun signifikan via mode hemat) |
| NFR-3 | Checkpoint training tersimpan inkremental ke Drive agar aman dari diskoneksi runtime Colab |
| NFR-4 | Line Crossing Detection tidak merusak mode deteksi normal (toggle via flag, default mengikuti konfigurasi eksperimen) |
| NFR-5 | Workspace lama tidak dihapus/diubah — tetap utuh sebagai arsip hasil uji 1–7c |

## 8. Deliverable

Setiap hasil uji (`output/uji_N/`) wajib berisi — mengikuti pola `PMLD/output` saat ini:

1. **Video hasil deteksi** beranotasi.
2. **Capture picture** hasil deteksi dengan bbox yang akurat.
3. **File TXT** hasil pembacaan teks plat.
4. **File JSON** validasi keakuratan pembacaan.

File-file di atas hanya di-download setelah hasil dinilai bagus secara inline di Colab.

## 9. Rencana Eksekusi (Tahapan)

| Fase | Aktivitas | Output |
| :-- | :-- | :-- |
| 1. Migrasi | Buat struktur `PMLD_FT/`, copy item sesuai tabel keputusan, sesuaikan path `data.yaml`, upload `plat_indo_samping.mp4` | Workspace baru siap pakai |
| 2. Verifikasi awal | Load `data.yaml` + `best.pt`, jalankan inferensi baseline pada `sample_vidio_1.mp4` (kontrol regresi) dan `plat_indo_samping.mp4` | Baseline tercatat di `output/uji_8/` |
| 3. Ekstraksi frame | Jalankan `01_extract_frames.ipynb` → sampling frame video baru → anotasi | `dataset/dataset_ft/` terisi |
| 4. Fine-tuning | Jalankan `02_yolo_finetune.ipynb` (data gabungan, batch kecil, checkpoint inkremental) | Model baru + log metrik inline |
| 5. Evaluasi | Jalankan `03_inference_eval.ipynb` pada kedua video uji; bandingkan vs `baseline_metrics/`; iterasi parameter bila belum sesuai | `output/uji_N/` dengan perbandingan angka |
| 6. Finalisasi | Download deliverable (video, capture bbox, txt, json) setelah hasil memenuhi G1 | Deliverable final untuk tim ML/backend |

## 10. Acceptance Criteria

- [ ] Training Colab berjalan tanpa error path (verifikasi triplet `data.yaml`/`dataset/`/`best.pt` lolos).
- [ ] Hasil fine-tune terlihat inline di Colab tanpa download.
- [ ] Deteksi & OCR pada `plat_indo_samping.mp4` membaik terukur dibanding baseline (angka dari log).
- [ ] Tidak ada regresi pada `sample_vidio_1.mp4`.
- [ ] Ukuran workspace `PMLD_FT/` ≤ ~0,5 GB (di luar `output/` yang tumbuh saat eksperimen).
- [ ] Deliverable uji final berisi 4 komponen: video, capture bbox, txt, json validasi.
- [ ] Fitur line crossing dapat diaktifkan/nonaktifkan tanpa mengubah kode pipeline.

## 11. Risiko & Mitigasi

| ID | Risiko | Mitigasi |
| :-- | :-- | :-- |
| R1 | Frame hasil ekstraksi belum cukup representatif → fine-tune tidak membaik | Sampling bervariasi (variasi angle, pencahayaan, jarak); tambah iterasi anotasi |
| R2 | VRAM tetap jebol saat inferensi video panjang | Mode hemat wajib default: downscale + frame skipping + OCR selektif; uji bertahap durasi pendek → panjang |
| R3 | Runtime Colab diskonek saat training panjang | Checkpoint inkremental ke Drive + resume dari checkpoint terakhir |
| R4 | Path `data.yaml` salah arah ke workspace lama | Verifikasi awal wajib (Fase 2) sebelum training |
| R5 | Line crossing mengubah perilaku deteksi tak terduga | Toggle via flag config; diuji terpisah sebelum digabung ke pipeline utama |
| R6 | Anotasi manual memakan waktu | Sampling frame cerdas (dedup frame mirip), prioritas frame dengan plat terlihat |

## 12. Catatan

- Prinsip kerja: **keputusan berdasarkan angka dari log, bukan tebakan** — setiap klaim "membaik" harus didukung perbandingan metrik.
- Workspace lama `PMLD/` tetap menjadi arsip utuh dan sumber rujukan histori uji 1–7c.
- Goals utama: **model dapat mendeteksi plat nomor dengan akurat dari video dengan angle kamera sedikit miring.**
