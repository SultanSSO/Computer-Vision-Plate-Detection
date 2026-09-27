# 🚗 Indonesian License Plate Recognition (ALPR) — YOLOv8 + PaddleOCR

Workspace **PMLD — plat-detection**: deteksi plat nomor Indonesia dengan **YOLOv8**,
ekstraksi teks dengan **PaddleOCR**, koreksi posisi karakter (D3), tracking antar-frame
dengan temporal voting (D4), dan tampilan terkunci anti flip-flop (locked display).

Workspace ini **sudah pernah dijalankan** — hasil run final (**uji_7c**) tersedia di
[`output/uji_7c_paddleocr_sample_vidio_2.mp4/`](output/uji_7c_paddleocr_sample_vidio_2.mp4/).

---

## 📦 Instalasi

> **Prasyarat:** Python **3.10 atau 3.12** (direkomendasikan 3.12).
> ⚠️ **Jangan Python 3.11** — kombinasi `paddlepaddle` + `torch` di 3.11 Windows
> bentrok DLL (`shm.dll` WinError 127). Python 3.13 belum didukung paddlepaddle.
>
> **Catatan teknis:** `paddlepaddle` 2.6.x dan `torch` membawa `libiomp5md.dll`
> (Intel OpenMP) versi berbeda. Kode di repo ini selalu mengimpor **torch sebelum
> paddle** (lihat `src/pipeline.py`) — jangan ubah urutan itu bila mengedit.

```bash
git clone <repo-url>
cd plat-detection
python -m venv .venv          # pastikan python = 3.10 / 3.12
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU (opsional):** ganti `paddlepaddle` dengan `paddlepaddle-gpu` sesuai versi CUDA
> ([panduan](https://www.paddlepaddle.org.cn/install/quick)). Default = CPU, tetap jalan (lebih lambat).
> Saat run pertama, PaddleOCR mengunduh model recognizer (~15 MB) — perlu internet.

## 📹 Video Sample (WAJIB untuk mode video)

Video uji **tidak disertakan di repo ini** (privat). Dapatkan dari **grup WhatsApp tim**,
lalu taruh di path persis berikut:

```
test_video/sample_vidio_2.mp4
```

Tanpa file ini, mode video berhenti dengan pesan error yang menjelaskan hal di atas.

## 🚀 Cara Menjalankan

```bash
# Mode video (default: test_video/sample_vidio_2.mp4)
python -m src.cli

# Video lain / nama run sendiri (tidak menimpa hasil uji_7c)
python -m src.cli --input test_video/video_anda.mp4 --run-name uji_saya

# Mode gambar (file tunggal atau folder)
python -m src.cli --input foto_plat.jpg
python -m src.cli --input folder_gambar/ --run-name uji_gambar_saya
```

Output tiap run masuk ke `output/<run_name>/`:

```
output/<run_name>/
├── annotated_images/   ← gambar beranotasi (mode gambar)
├── predictions_txt/    ← log JSONL per deteksi
├── processed_video/    ← video beranotasi (mode video)
└── video_captures/     ← capture frame tiap 7 detik (mode video)
```

Semua parameter pipeline (threshold, dsb.) ada di **`src/config.py`** —
nilai default = konfigurasi final uji_7c.

## 📓 Notebook

Versi notebook (sel demi sel, cocok untuk eksplorasi) tersedia dan bisa dijalankan
langsung di Jupyter lokal:

```bash
jupyter notebook notebooks/ocr_inference_paddleocr.ipynb
```

## 📊 Hasil Run Final — uji_7c

Run referensi pada `sample_vidio_2.mp4` (1080p @30fps, 54 detik, PaddleOCR CPU/GPU):

- **2.405 deteksi** tercatat (`predictions_txt/sample_vidio_2.mp4_log.jsonl`)
- **75 track kendaraan**, 56 di antaranya menghasilkan bacaan plat
- Contoh plat final terbaik (vote/stabilitas tinggi): `LN15ZZC` (122/126),
  `NL64OGX` (105/112), `EF10DZT` (101/110), `BP63LYH` (73/73)

Bukti hasil (gambar beranotasi + log + capture video) ikut di-commit di
`output/uji_7c_paddleocr_sample_vidio_2.mp4/`. Video hasil (`processed_video/`)
tidak di-commit karena berukuran besar — jalankan ulang untuk mereproduksinya.

> Run baru Anda otomatis memakai folder `output/<run_name>/` sendiri dan
> **tidak menimpa** hasil uji_7c.

## 🧠 Pipeline

```
frame → YOLOv8s (best.pt) → crop + padding 17% → grayscale → upscale 3×
      → PaddleOCR (PP-OCRv4) → filter charset A–Z0–9 → koreksi D3 (template posisi)
      → validasi regex domain (ID/UK) → tracking IoU + temporal voting (D4)
      → locked display (vote ≥ LOCK_MIN, override ≥ 2.5×)
```

## 📁 Struktur

```
plat-detection/
├── src/            ← kode utama: config.py, pipeline.py, cli.py
├── model/best.pt   ← bobot YOLOv8s final (Git LFS)
├── notebooks/      ← ocr_inference_paddleocr.ipynb (jalan di lokal)
├── scripts/        ← utilitas dataset (extract_crops.py)
├── docs/           ← PRD & catatan fase (keputusan desain)
├── test_video/     ← taruh video sample di sini (dari WhatsApp)
└── output/         ← hasil run (uji_7c + run baru Anda)
```

## 📚 Dokumentasi

Lihat `docs/` untuk keputusan desain lengkap:
`PRD_01_Project_Overview.md`, `PRD_decision_plat_v3.md` (W4/W5: PaddleOCR),
`PHASE4_SETUP_NOTES_UJI4.md`, `PRD_MITIGASI_V1.md`.
