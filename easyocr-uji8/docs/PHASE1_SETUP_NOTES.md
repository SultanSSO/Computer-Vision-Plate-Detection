# Phase 1 — Setup Notes (Data Acquisition & Environment)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 14 Sep 2026 | **Akun Drive:** sultannaufalalfaiq2006@mail.ugm.ac.id
**Status:** Phase 1 selesai kecuali video uji 1 menit (menunggu file dari tim)

---

## 1. Yang sudah dibuat di `MyDrive/PMLD`

Struktur folder sesuai PRD 02 §2.2 sudah dibuat lengkap dan terverifikasi:

```
PMLD/
├── docs/                                  ← PRD_01, PRD_02, catatan ini
├── dataset/{images,labels}/{train,val,test}/   ← sengaja KOSONG (lihat §3)
├── model/                                 ← best.pt & last.pt nanti dari Colab
├── output/{annotated_images,predictions_txt,processed_video}/
├── notebook/                              ← prepare_dataset.py (bootstrap dataset)
├── test_video/                            ← menunggu sample_video_1min.mp4
└── data.yaml
```

## 2. Hasil pengecekan dataset (Langkah A PRD: cek lokal/Drive dulu)

Zip Kaggle mentah **sudah ada di komputer lokal** — tidak perlu unduh ulang 1,46 GB:

| Bagian dataset | Isi | Ukuran |
| :--- | :--- | :--- |
| `plate_detection_dataset` | 1.383 gambar + `annotations.json` (COCO, 2.128 bbox, 1 kelas) | 316,8 MB |
| `plate_text_dataset` | 1.863 crop plat + `label.csv` (bahan uji OCR) | 1.163,1 MB |

Metadata dataset target (dicek tanpa kredensial): judul *Indonesian Vehicle License Plate Dataset*,
1.479,9 MB, `isPrivate=false`, update terakhir 2025-08-21.

## 3. Keputusan: dataset TIDAK diunggah ke Drive (Colab unduh langsung)

Diuji dengan permintaan HTTP nyata, bukan asumsi:

* Unduh publik **tanpa kredensial**: `HTTP 200` (uji `uciml/iris`, 3.687 byte terunduh).
* Dataset target tanpa kredensial: `HTTP 206` pada range request → endpoint terbuka.
* Artinya Colab bisa `wget/urllib` langsung dari Kaggle: jalur datacenter, tidak memakai
  bandwidth pribadi, tidak menghabiskan kuota Drive, dan tidak ada risiko upload 2.741 file
  kecil yang putus di tengah.

Konsekuensi: isi `PMLD/dataset/` di Drive dibiarkan kosong dan dataset dibangkitkan **saat runtime
Colab** ke `/content/dataset` (baca dari SSD lokal Colab = training lebih cepat daripada baca Drive FUSE).

**Cara pakai di Colab (2 perintah):**

```bash
%cd /content/drive/MyDrive/PMLD/notebook
!python prepare_dataset.py --workdir /content/dataset-mentah --out /content/dataset --download
```

Alternatif mode Drive (kalau tim tetap mau dataset fisik di Drive) — cukup ganti `--out`:

```bash
!python prepare_dataset.py --workdir /content/dataset-mentah \
    --out /content/drive/MyDrive/PMLD/dataset --download
```

## 4. Hasil uji skrip di lokal (bukti, bukan klaim)

`prepare_dataset.py` dijalankan lokal memakai zip yang sudah ada, keluaran `C:\Users\Sultan\plat-deteksi\dataset-pmld`:

| Split | Gambar | Label |
| :--- | ---: | ---: |
| train | 959 | 959 |
| val | 274 | 274 |
| test | 137 | 137 |
| **total** | **1.370** | 1.370 (2.124 bounding box) |

Verifikasi:
* Semua 1.370 label berformat YOLO valid (`kelas cx cy w h`, nilai 0–1, w/h > 0).
* 284 gambar punya lebih dari satu plat (mendukung kebutuhan multi-objek F-02).
* **Irisan antar-split = 0** (tidak ada kebocoran data train→val/test).
* 1.369 label hasil skrip **identik nilai per nilai** dengan dataset yang sudah dipakai sebelumnya →
  konversi COCO→YOLO terbukti tidak bergeser. Laporan mesin: `docs/phase1_dataset_report.json`.

## 5. Penyimpangan dari PRD (disengaja, perlu diketahui)

1. **Dataset tidak fisik di Drive** — PRD 02 §3.2 memakai `path: /content/drive/MyDrive/PMLD/dataset/...`.
   Karena Colab mengunduh sendiri, `path` di `data.yaml` mengarah ke `/content/dataset`
   (mode Drive tetap bisa dengan `--out` seperti di atas).
2. **Split test 70/20/10 dibangkitkan skrip** dengan seed tetap `42` (PRD hanya menyebut struktur
   folder train/val/test, tanpa proporsi). Proporsi diubah lewat argumen `--ratios`.
3. **Nama kelas `license_plate`** mengikuti PRD 02 §3.2 (dataset lama memakai `plate_number`;
   hanya nama di `data.yaml`, file label tidak berubah karena class id tetap `0`).
4. **Skrip disimpan di `notebook/prepare_dataset.py`** (PRD hanya menyebut dua notebook; skrip ini
   pendamping agar notebook tetap self-bootstrapping dan bisa dijalankan ulang kapan saja).

## 6. Yang masih menunggu

* `test_video/sample_video_1min.mp4` — file video uji dari tim; Phase 4 tidak bisa jalan tanpa ini.
* `notebook/yolo_training.ipynb` (Phase 2) dan `notebook/ocr_inference.ipynb` (Phase 3).
