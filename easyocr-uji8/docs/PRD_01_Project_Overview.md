# PRD 01: Project Overview & Requirements
## Indonesian License Plate Recognition System (YOLOv8 + OCR)

---

## 1. Document Overview & Project Context
* **Project Name:** Indonesian License Plate Detection & Text Extraction System (ALPR - Automatic License Plate Recognition)
* **Target Audience:** AI Agent & AI Engineer
* **Document Version:** v2.0 (Updated with Team Context & PMLD Drive Architecture)
* **Date:** September 13, 2026
* **Status:** Active Project Document (Part 1 of 2)

### 1.1 Context & Ecosystem Position
Proyek ini merupakan bagian dari sistem Automatic License Plate Recognition (ALPR) berbasis microservices yang dikembangkan oleh tim lintas peran (AI Engineer, MLOps, Backend, Frontend, DevOps, UI). 

Fokus utama dokumen ini adalah mengatur cakupan kerja **AI Engineer**. Tugas utama AI Engineer adalah melakukan eksplorasi dataset, melatih (*fine-tuning*) model deteksi objek **YOLOv8**, mengintegrasikan *best model* dengan modul **OCR (Optical Character Recognition)**, serta menguji performanya pada dataset gambar dan video pengujian 1 menit. Hasil pekerjaan ini (*notebook* & *model weights*) nantinya akan diserahterimakan kepada MLOps untuk diintegrasikan ke dalam REST API (FastAPI) dan dikontainerisasi oleh DevOps.

---

## 2. Roles & Responsibilities

### 2.1 Primary Focus: AI Engineer
Sebagai AI Engineer, tanggung jawab penuh mencakup:
1. **Mencari & Mengelola Dataset:** Mengidentifikasi dataset plat nomor Indonesia teranotasi dari Kaggle (`linkgish/indonesian-plate-number-from-multi-sources`), memprioritaskan pencarian di penyimpanan lokal/Drive terlebih dahulu sebelum melakukan pengunduhan otomatis via skrip.
2. **Fine-Tuning Model YOLOv8:** Melatih dan melakukan *hyperparameter tuning* pada YOLOv8 di Google Colab untuk menghasilkan *best model* (`best.pt`) dengan akurasi lokalisasi tinggi.
3. **Integrasi Model dengan OCR:** Menghubungkan koordinat *bounding box* hasil deteksi YOLOv8 dengan pemrosesan awal citra (*grayscale*, *thresholding*, CLAHE) dan mesin OCR untuk mengekstrak teks plat nomor.
4. **Uji Coba & Evaluasi Model:** Menguji pipeline inferensi menggunakan sampel citra dan file **video pengujian berdurasi ~1 menit** yang disediakan di Google Drive.

### 2.2 Supporting Role: AI Agent
* Bertindak sebagai asisten teknis yang memandu AI Engineer langkah demi langkah (*step-by-step assistant*).
* Menyediakan skrip otomatisasi untuk memeriksa dataset lokal, mengunduh data via Kaggle API jika belum ada, mengatur struktur folder di Drive, serta mengoptimalkan skrip notebook training dan OCR.

### 2.3 Context Inter-Team Handover (Batas Tanggung Jawab)
* **MLOps:** Menerima `best.pt` & skrip OCR dari AI Engineer untuk dibangun menjadi REST API berbasis FastAPI.
* **DevOps:** Menerima service dari MLOps, BE, dan FE untuk dikontainerisasi (Dockerfile/Docker Compose).
* **Backend (Go) & Frontend (React):** Mengonsumsi API hasil deteksi dan menyajikan UI visualisasi.

---

## 3. Scope of Work (Cakupan Kerja AI Engineer)

### 3.1 In-Scope (Wajib Dikerjakan)
1. **Environment Setup & Drive Integration:** Seluruh pengerjaan berbasis **Google Colab** dengan root directory Google Drive di `MyDrive/PMLD`.
2. **Dataset Management Pipeline:**
   * Pencarian dataset lokal/Drive terlebih dahulu.
   * Pengunduhan otomatis dataset Kaggle `linkgish/indonesian-plate-number-from-multi-sources` langsung ke `MyDrive/PMLD/dataset` jika data belum ada.
   * Pembagian dataset (*train/val/test split*) dan verifikasi anotasi YOLO.
3. **YOLOv8 Training & Fine-Tuning:**
   * Melatih model YOLOv8 (yolov8n/yolov8s) pada folder `notebook/yolo_training.ipynb`.
   * Melakukan eksperimen epoch dan hyperparameter untuk mencapai mAP50 optimal.
   * Menyimpan *best weights* (`best.pt`) ke `MyDrive/PMLD/model/`.
4. **OCR Pipeline Integration:**
   * Pengembangan pipeline inferensi pada `notebook/ocr_inference.ipynb`.
   * Pemotongan region plat nomor (*crop*), perbaikan citra (CLAHE/Binarization), dan ekstraksi teks dengan OCR.
5. **Testing & Output Generation:**
   * Pengujian pada citra tes dan **video uji 1 menit** (`MyDrive/PMLD/test_video/`).
   * Menghasilkan citra/video beranotasi *bounding box* + label teks dan file teks hasil prediksi `.txt`.

### 3.2 Out-of-Scope (Bukan Tugas AI Engineer)
1. Pembuatan REST API / FastAPI (Tugas MLOps).
2. Konfigurasi Docker / Containerization & Port Mapping (Tugas DevOps).
3. Pembuatan Database Go / Webhook / WebSocket (Tugas Backend).
4. Slicing UI & React Integration (Tugas Frontend & UI Design).

---

## 4. Output & Deliverable Specifications

Sebagai AI Engineer, luaran (*deliverables*) yang harus diselesaikan di `MyDrive/PMLD` adalah:

1. **Jalur Notebook (`notebook/`):**
   * `notebook/yolo_training.ipynb`: File Jupyter Notebook untuk pelatihan dan evaluasi model YOLOv8.
   * `notebook/ocr_inference.ipynb`: File Jupyter Notebook yang memuat model `best.pt` terintegrasi dengan OCR untuk pengujian citra & video.
2. **Model Weights (`model/`):**
   * `model/best.pt`: Bobot model terbaik hasil pelatihan YOLOv8.
3. **File Konfigurasi (`data.yaml`):**
   * File `data.yaml` yang tersetel presisi merujuk pada path dataset di Drive.
4. **Hasil Prediksi (`output/`):**
   * `output/annotated_images/`: Gambar tes dengan *bounding box* dan teks plat.
   * `output/predictions_txt/`: File `.txt` memuat metadata hasil deteksi.
   * `output/processed_video/`: Video hasil inferensi 1 menit beranotasi *bounding box* dan teks plat.

---

## 5. Success Metrics & Performance Criteria

| Metrik | Target Keberhasilan | Catatan |
| :--- | :--- | :--- |
| **Bounding Box Precision (mAP50)** | $\ge 85\%$ | Akurasi lokalisasi plat nomor oleh YOLOv8 |
| **OCR Text Accuracy** | $\ge 80\%$ | Persentase pembacaan karakter (angka & huruf) yang benar |
| **Video Testing Completion** | $100\%$ | Pipeline di `ocr_inference.ipynb` mampu memproses video 1 menit tanpa error |
| **Output Completeness** | $100\%$ | Setiap pengujian menghasilkan visual beranotasi dan file log `.txt` |

---

## 6. Risk Management & Mitigations

* **Risiko 1: Dataset Belum Terunduh / Path Tidak Ditemukan**
  * *Mitigasi:* AI Agent mengeksekusi skrip pemeriksaan keberadaan dataset di `MyDrive/PMLD/dataset`. Jika kosong, skrip Kaggle API langsung dijalankan untuk mengunduh otomatis tanpa bantuan manual.
* **Risiko 2: Sesi Google Colab Terputus Saat Training**
  * *Mitigasi:* Konfigurasi `save_period=5` agar *checkpoint* pelatihan berkala langsung tersimpan di `MyDrive/PMLD/model/`.
* **Risiko 3: Pembacaan OCR Kurang Akurat pada Video**
  * *Mitigasi:* Terapkan teknik pemrosesan citra (CLAHE & Grayscale) serta *frame skipping/smoothing* pada video untuk menjaga akurasi dan kecepatan.
