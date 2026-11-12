# Phase 2 — Setup Notes (YOLOv8 Training Notebook)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 14 Sep 2026
**Status:** Phase 2 artefak SIAP — Drive sudah bisa ditraining

---

## 1. Perubahan keputusan: dataset sekarang FISIK di Drive

Phase 1 §3 memutuskan dataset **tidak** disimpan di Drive (Colab unduh Kaggle saat runtime).
Keputusan itu **diubah** pada 14 Sep 2026 atas arahan pemilik proyek: dataset yang sudah
tersedia di folder lama `gdrive:plat-deteksi/dataset` (750 gambar siap-pakai) dipindahkan
ke `gdrive:PMLD/dataset` dan **diselaraskan ke konteks PRD**.

Alasan perubahan:
* Training bisa langsung jalan tanpa menunggu unduh 1,4 GB tiap sesi Colab.
* Dataset sudah terverifikasi (pasangan gambar-label 1:1, label YOLO valid).

Mode runtime tetap tersedia di notebook lewat `DATASET_MODE = 'runtime'`.

## 2. Penyelarasan dataset ke PRD (yang dilakukan)

| # | Aspek | Kondisi dataset lama | Setelah diselaraskan | Acuan |
| :-- | :--- | :--- | :--- | :--- |
| 1 | Split | train/val saja (80/20) | **train/val/test 70/20/10** | PRD 02 §2.2 |
| 2 | Nama kelas | `plate_number` | **`license_plate`** | PRD 02 §3.2 |
| 3 | `path` | `.../plat-deteksi/dataset` | **`.../PMLD/dataset`** | PRD 02 §2.1 |
| 4 | `nc` | tidak ada | **`nc: 1`** | PRD 02 §3.2 |
| 5 | `data.yaml` | di dalam `dataset/` | **di root `MyDrive/PMLD/`** | PRD 02 §2.2 |
| 6 | Video uji | `docs/vidio_plate.mp4` | **`test_video/sample_video_1min.mp4`** | PRD 02 §3.5 |
| 7 | README lama | menyatakan dataset kosong | **dihapus** (sudah tidak benar) | — |

Pembagian split test dibuat **deterministik** (seed 42): 75 pasang dipindah dari train,
sehingga 750 gambar terbagi tepat 525 / 150 / 75.

## 3. Kondisi dataset di `MyDrive/PMLD/dataset` (terverifikasi)

| Split | Gambar | Label | Pasangan 1:1 |
| :--- | ---: | ---: | :--- |
| train | 525 (70%) | 525 | ✅ |
| val | 150 (20%) | 150 | ✅ |
| test | 75 (10%) | 75 | ✅ |
| **total** | **750** | **750** | ✅ |

Verifikasi yang dijalankan: setiap split punya jumlah gambar = jumlah label, dan
himpunan nama file gambar identik dengan himpunan nama file label (0 mismatch).

## 4. Notebook `yolo_training.ipynb` (25 sel, 12 sel kode)

| Sel | Isi |
| :--- | :--- |
| 1–3 | Cek GPU (`assert` CUDA tersedia) + `nvidia-smi` |
| 4–5 | Mount Drive; definisikan `PROJECT_DIR=/content/drive/MyDrive/PMLD`, `MODEL_DIR`, `RUNS_DIR`, `DATA_YAML` |
| 6–7 | **Konfigurasi terpusat**: `RUN_NAME`, `BASE_MODEL`, `EPOCHS`, `IMG_SIZE`, `BATCH`, `SAVE_PERIOD`, `PATIENCE`, `SEED`, **`DATASET_MODE`** |
| 8–9 | Install Ultralytics |
| 10–11 | Siapkan dataset — **mode `drive`**: verifikasi struktur; **mode `runtime`**: jalankan `prepare_dataset.py` |
| 12–13 | Verifikasi dataset (mode-aware) + tampilkan `data.yaml` |
| 14 | Pastikan `MyDrive/PMLD/data.yaml` (deliverable) ada & sesuai |
| 15–16 | **Training** — `save_period=5` → checkpoint berkala ke Drive |
| 17–18 | Salin `best.pt` & `last.pt` → `MyDrive/PMLD/model/` |
| 19–20 | Evaluasi mAP50 / mAP50-95 / Precision / Recall + cek target ≥0.85 + simpan `_metrics.json` |
| 21–22 | Visualisasi `results.png`, `PR_curve.png`, `confusion_matrix.png` |
| 23–24 | Tes prediksi cepat pada 4 sampel val |

### `DATASET_MODE`

* **`'drive'` (default)** — dataset fisik di `MyDrive/PMLD/dataset`, `data.yaml` dari root PMLD.
  Ini yang dipakai sekarang. Tidak ada unduhan saat runtime.
* **`'runtime'`** — Colab unduh Kaggle lalu bangkitkan dataset ke `/content/dataset`
  (mode Phase 1, tetap tersedia sebagai cadangan).

## 5. Kepatuhan notebook terhadap PRD 02 §3.3

| Spesifikasi PRD | Implementasi |
| :--- | :--- |
| Base model `yolov8n.pt` / `yolov8s.pt` | `BASE_MODEL` (default `yolov8n.pt`) |
| Input resolution 640×640 | `IMG_SIZE = 640` |
| Epochs 50–100 | `EPOCHS = 50` |
| Batch size 16 / 32 | `BATCH = 16` |
| Checkpoint `save_period=5` → `MyDrive/PMLD/model/` | `save_period=SAVE_PERIOD` |
| Kelas `license_plate`, `nc: 1` | di `data.yaml` |
| Deliverable `model/best.pt` & `model/last.pt` | sel 17–18 menyalin keduanya |

## 6. Hasil uji skrip di lokal (bukti, bukan klaim)

`prepare_dataset.py` (mode runtime) diuji dengan zip yang sudah ada, subset 120 gambar:

| Split | Gambar | Label |
| :--- | ---: | ---: |
| train | 84 | 84 |
| val | 24 | 24 |
| test | 12 | 12 |
| **total** | **120** | 120 (199 bounding box) |

* 199 baris label YOLO valid (5 kolom, nilai 0–1, w/h > 0) → **0 tidak valid**.
* Irisan nama antar-split = **0**; irisan **hash MD5 konten gambar** antar-split = **0**.
* Waktu eksekusi ~5 detik.

Unduhan Kaggle tanpa kredensial juga diuji nyata: `HTTP 206`, `Content-Range: bytes 0-1023/1464859619`
(1,46 GB) → endpoint terbuka, mode runtime tetap bisa dipakai.

## 7. Yang masih menunggu

* **Eksekusi training di Colab** — notebook siap, belum pernah dijalankan (tidak ada GPU lokal).
* **Konfirmasi durasi video uji** ke tim: file yang ada 29,6 detik, PRD 02 §3.5 meminta 1 menit.
* `notebook/ocr_inference.ipynb` (Phase 3) — belum dibuat.

## 8. Cara menjalankan

1. Buka Colab → `MyDrive/PMLD/notebook/yolo_training.ipynb`
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all` (mode default `DATASET_MODE='drive'` — langsung pakai dataset di Drive)
4. Pantau sel 19: target `mAP50 ≥ 0.85`
