# PRD 02: Technical Architecture & Infrastructure
## Indonesian License Plate Recognition System (YOLOv8 + OCR)

---

## 1. Technical System Architecture & End-to-End Workflow

Arsitektur teknis proyek ini dibagi menjadi dua bagian utama: **Alur Kerja Internal AI Engineer** (fokus utama pekerjaan Anda) dan **Alur Serah Terima Antar-Tim (Inter-Team Handover Flow)**.

### 1.1 Alur Kerja Internal AI Engineer (Internal AI Pipeline Flow)

```text
  [ START: Inisialisasi Environment Google Colab ]
                         │
                         ▼
  [ Phase 1: Data Acquisition & Check ]
   ├── Cek ketersediaan dataset di Drive: `MyDrive/PMLD/dataset`
   ├── Jika ADA  ──► Gunakan dataset lokal
   └── Jika TDK  ──► Auto-download Kaggle Dataset via API ke `MyDrive/PMLD/dataset`
                         │
                         ▼
  [ Phase 2: YOLOv8 Training & Fine-Tuning ]
   ├── Jalankan `MyDrive/PMLD/notebook/yolo_training.ipynb`
   ├── Baca konfigurasi dari `MyDrive/PMLD/data.yaml`
   ├── Melatih model (yolov8n.pt/yolov8s.pt) selama 50-100 Epochs
   └── Simpan bobot terbaik ke `MyDrive/PMLD/model/best.pt`
                         │
                         ▼
  [ Phase 3: OCR Pipeline Development ]
   ├── Jalankan `MyDrive/PMLD/notebook/ocr_inference.ipynb`
   ├── Load model `MyDrive/PMLD/model/best.pt`
   ├── Prediksi Bounding Box Plat Nomor: [x1, y1, x2, y2]
   ├── Crop Region Plat Nomor ──► Preprocessing (Grayscale + CLAHE)
   └── Ekstraksi Teks dengan OCR (EasyOCR/PaddleOCR) ──► Post-Processing Regex
                         │
                         ▼
  [ Phase 4: Model Testing & Validation ]
   ├── Uji Citra Tes   ──► Simpan ke `MyDrive/PMLD/output/annotated_images/` & `.txt`
   └── Uji Video 1-Min ──► Simpan ke `MyDrive/PMLD/output/processed_video/`
                         │
                         ▼
  [ END: Artefak Siap Diserahterimakan ke MLOps ]
```

### 1.2 Alur Serah Terima Antar-Tim (Inter-Team Integration Flow)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. AI ENGINEER (Role Anda)                                                             │
│    • Melatih YOLOv8 & Membangun Pipeline OCR di Google Colab.                          │
│    • Menghasilkan `best.pt`, `yolo_training.ipynb`, dan `ocr_inference.ipynb` di PMLD. │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (Handover best.pt & skrip OCR)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MLOPS                                                                               │
│    • Mengambil `best.pt` & skrip OCR dari PMLD.                                        │
│    • Membungkus model menjadi REST API menggunakan FastAPI.                            │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (Handover Service API)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. DEVOPS                                                                              │
│    • Membuarkan Dockerfile ringan untuk FastAPI (MLOps), Go (BE), dan React (FE).     │
│    • Mengatur Docker Compose, Port Mapping, & Resource Limits.                         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌────────────────────────────────────────┐   ┌────────────────────────────────────────┐
│ 4. BACKEND (Go)                        │   │ 5. FRONTEND (React.js)                 │
│    • Menerima data deteksi via POST    │   │    • Menampilkan dashboard statistik.   │
│      webhook `/detections`.            │   │    • Menampilkan tabel hasil deteksi,  │
│    • Manajemen DB CCTV & Auth JWT.     │   │      filter, dan live streaming.    │
└────────────────────────────────────────┘   └────────────────────────────────────────┘
```

---

## 2. Infrastructure & Storage Specifications (MyDrive/PMLD)

### 2.1 Computing Infrastructure
* **Execution Platform:** Google Colab (Cloud Compute GPU T4 / V100).
* **Storage Root Path:** Seluruh file proyek wajib berada di dalam direktori `MyDrive/PMLD`.

### 2.2 Standardized Directory Structure (`MyDrive/PMLD`)
Seluruh folder dan file proyek AI Engineer terpusat 100% di dalam direktori **PMLD**:

```text
MyDrive/PMLD/
├── docs/                                   # Folder Dokumen PRD
│   ├── PRD_01_Project_Overview.md
│   └── PRD_02_Technical_Architecture_Infrastructure.md
│
├── dataset/                                # Folder Dataset Plat Nomor
│   ├── images/
│   │   ├── train/                          # Gambar untuk Training
│   │   ├── val/                            # Gambar untuk Validasi
│   │   └── test/                           # Gambar untuk Testing
│   └── labels/
│       ├── train/                          # Label YOLO (.txt) Training
│       ├── val/                            # Label YOLO (.txt) Validasi
│       └── test/                           # Label YOLO (.txt) Testing
│
├── test_video/                             # Folder Video Pengujian
│   └── sample_video_1min.mp4               # File Video 1 Menit untuk Uji Inferensi
│
├── model/                                  # Folder Penyimpanan Weights Model
│   ├── last.pt                             # Checkpoint Pelatihan Terakhir
│   └── best.pt                             # Model Terbaik Hasil Training YOLOv8
│
├── output/                                 # Folder Luaran/Hasil Prediksi
│   ├── annotated_images/                   # Gambar Prediksi (+ Bounding Box & Label)
│   ├── predictions_txt/                    # Log Teks Hasil Prediksi (.txt)
│   └── processed_video/                    # Video Render 1 Menit (+ Bounding Box & OCR)
│
├── notebook/                               # Folder Jupyter Notebook Colab
│   ├── yolo_training.ipynb                 # Notebook Pelatihan & Fine-Tuning YOLOv8
│   └── ocr_inference.ipynb                 # Notebook Integrasi OCR & Testing Video/Gambar
│
└── data.yaml                               # File Konfigurasi Dataset YOLOv8
```

---

## 3. Machine Learning & OCR Technical Specifications

### 3.1 Dataset Search & Acquisition Protocol
1. **Target Dataset Kaggle:** `https://www.kaggle.com/datasets/linkgish/indonesian-plate-number-from-multi-sources`.
2. **Prosedur Otomatis AI Agent:**
   * **Langkah A (Cek Storage):** AI Agent memeriksa apakah `MyDrive/PMLD/dataset/images` dan `labels` sudah terisi data.
   * **Langkah B (Pengunduhan Otomatis):** Jika folder kosong/belum ada, jalankan skrip pengunduhan via Kaggle API di Google Colab yang mengekstrak data secara otomatis langsung ke `MyDrive/PMLD/dataset/`.

### 3.2 Dataset Configuration (`MyDrive/PMLD/data.yaml`)
```yaml
train: /content/drive/MyDrive/PMLD/dataset/images/train
val: /content/drive/MyDrive/PMLD/dataset/images/val
test: /content/drive/MyDrive/PMLD/dataset/images/test

nc: 1
names: ['license_plate']
```

### 3.3 YOLOv8 Fine-Tuning Specifications (`yolo_training.ipynb`)
* **Base Model:** `yolov8n.pt` (Nano) atau `yolov8s.pt` (Small).
* **Input Resolution:** $640 	imes 640$ pixels.
* **Epochs:** 50 – 100 Epochs (disesuaikan kestabilan loss).
* **Batch Size:** 16 / 32.
* **Checkpointing:** Opsi `save_period=5` (otomatis menyelaraskan simpanan bobot ke `MyDrive/PMLD/model/`).

### 3.4 OCR & Preprocessing Pipeline Specifications (`ocr_inference.ipynb`)
* **OCR Library:** EasyOCR (Primary) / PaddleOCR.
* **Alur Deteksi & Pembacaan Teks:**
  1. **Bounding Box Detection:** Dapatkan koordinat $[x1, y1, x2, y2]$ dari prediksi `best.pt`.
  2. **Image Cropping:** Potong citra plat nomor dari gambar/frame asli.
  3. **Preprocessing:**
     * Ubah citra ke Grayscale.
     * Terapkan CLAHE (*Contrast Limited Adaptive Histogram Equalization*) / Thresholding agar karakter kontras.
  4. **Text Extraction:** Salurkan citra yang sudah diproses ke modul OCR.
  5. **Text Post-processing:** Pembersihan noise dengan pola Regex Plat Indonesia (Contoh: `^[A-Z]{1,2}\s[0-9]{1,4}\s[A-Z]{1,3}$`).

### 3.5 Video Testing Pipeline Specifications
* **Input Video:** `MyDrive/PMLD/test_video/sample_video_1min.mp4`.
* **Proses Inferensi Video:**
  1. Baca video frame demi frame menggunakan OpenCV.
  2. Eksekusi deteksi YOLOv8 `best.pt` & pembacaan OCR pada setiap frame (bisa diterapkan *frame-skipping* misal tiap 2-3 frame untuk efisiensi).
  3. Gambar *bounding box* dan teks plat hasil OCR di atas frame.
  4. Simpan video beranotasi ke `MyDrive/PMLD/output/processed_video/sample_video_1min_annotated.mp4`.

---

## 4. Inter-Team Interface Boundaries

Tabel batas tanggung jawab dan titik integrasi untuk kejelasan tugas:

| Peran Tim | Batas Tanggung Jawab Utama | Titik Integrasi dengan AI Engineer |
| :--- | :--- | :--- |
| **AI Engineer (Anda)** | Fine-tuning YOLOv8, integrasi OCR, uji coba citra & video 1 menit di Google Colab. | Menghasilkan `best.pt` & skrip inferensi di `MyDrive/PMLD`. |
| **MLOps** | Membangun REST API FastAPI, video processing stream service, integrasi model ke BE. | Menggunakan `best.pt` & logika OCR dari AI Engineer untuk dibuatkan REST API. |
| **DevOps** | Membuat Dockerfile, Docker Compose, Port Mapping, & batas RAM/CPU. | Mengemas service FastAPI (MLOps), Go (BE), dan React (FE). |
| **Backend (Go)** | Auth JWT, DB Migration, Manajemen CCTV, Webhook Endpoint `/detections`. | Menerima data JSON hasil deteksi dari API MLOps. |
| **Frontend (React)** | Slicing UI Dashboard, Tabel Hasil Deteksi, Filter, & Live Preview. | Menampilkan data deteksi dari Backend ke interface pengguna. |
| **UI/UX** | Design Figma & Prototyping tampilan. | Menjadi panduan desain visual untuk Frontend. |

---

## 5. AI Agent Technical Execution Protocol

AI Agent wajib mengeksekusi dan mendampingi AI Engineer mengikuti tahapan teknis berikut:

### Langkah 1: Setup Environment & Drive Mount
* Mount Google Drive ke Google Colab (`/content/drive`).
* Pastikan dan buat struktur folder di `MyDrive/PMLD/{docs,dataset,model,output,notebook,test_video}`.

### Langkah 2: Smart Dataset Acquisition
* Cek folder `MyDrive/PMLD/dataset/images`.
* Jika kosong, unduh dataset Kaggle `linkgish/indonesian-plate-number-from-multi-sources` secara otomatis via skrip dan ekstrak langsung ke `MyDrive/PMLD/dataset/`.
* Buat/perbarui file `MyDrive/PMLD/data.yaml`.

### Langkah 3: Eksekusi Training YOLOv8 (`yolo_training.ipynb`)
* Jalankan script pelatihan YOLOv8 dengan *checkpointing* berkala di `MyDrive/PMLD/model/`.
* Evaluasi mAP50, Precision, dan Recall.
* Pastikan `best.pt` tersimpan dengan sempurna di `MyDrive/PMLD/model/best.pt`.

### Langkah 4: Pipeline OCR & Video Testing (`ocr_inference.ipynb`)
* Jalankan notebook inferensi mengimpor `best.pt`.
* Uji pada sampel citra di `dataset/images/test`.
* Uji pada file video 1 menit di `test_video/`.
* Simpan hasil citra/video beranotasi ke `output/` serta simpan log teks di `output/predictions_txt/`.
