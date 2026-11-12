# Overview :
- PMLD merupakan proyek mandiri lintas disiplin yang diberikan oleh dosen pembimbingku sebagai syarat matakuliah dan sks.

- proyek yang kupilih adalah computer vision untuk mendeteksi teks plat nomor kendaraan dengan akurat.

- proyek ini dikerjakan oleh beberapa oang dalam bentuk kelompok. role yang aku ambil adalah AI engineer yang bertugas untuk, melatih model yolo hingga mendapatkan prediksi terbaik. role AI engineer ini tentu juga kan berkolaborasi dengan role yang lain, seperti beckand dan machine learning. hasil dari prediksiku nanti akan dikirimkan mereka dan akan diproses lebih lanjut.

# Alasan migrasi :
- sebenarnya aku sudah membuat workspace dan mendapatkan best model di drive : https://drive.google.com/drive/folders/1Ev4nRcrHNUZs_XQFicXkPYZY9TmUU_wb di dalam projek ini vidio test yang digunakan itu vidio test dari plat british luar negeri dengan angel kamera fokus dari depan dan plat nomer terlihat jelas sehingga hasilnya sudah bagus. namun ketika diberi vidio test yang baru dengan plat indonesia dan angel pengambilan gambarnya itu dari samping, model kesusahan untuk membaca.

- saat model diberikan vidio baru yang berplat indonesia dan angel kameranya dari samping, hasil deteksinya sering halu, objek yang seharusnya tidak dideteksi malah terdeteksi, hasil pembacaan ocr teksnya juga banyak yang eror.

- aku ingin saat proses training dan lain lainya selesai, hasilnya itu dapat dilihat di google colabnya langsung, jadi untuk melihat hasil fine tunning atau triningnya itu tidak perlu mendownload hasilnya terlebih dahulu karena prosesnya ribet.

- saat model di running itu memakan 11 gb, sehingga lumayan memberatkan. ini mungkin terjadi karena aku memprosesnya kan menggunakan T4 dari colab, dan temanku yang berperan sebagai machine learning dan backand jadi menjalankannya di lokal.

- workspace terlihat berantakan dan mungkin bisa dikatakan terlalu besar menurut saya pribadi.

# Struktur rencana folder migrasi : (Bagian yang harus di isi oleh hermes agent)

**Root baru:** `MyDrive/PMLD_FT/` — estimasi total ~0,5 GB (workspace lama jauh lebih besar).

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

## Rincian: apa yang dicopy, apa yang tidak

| Item | Keputusan | Alasan |
| :-- | :-- | :-- |
| `data.yaml` | COPY + sesuaikan path | Wajib untuk training; path dataset lama menunjuk ke folder PMLD lama, harus diarahkan ke PMLD_FT |
| `dataset/images/` (750 gambar: 525 train, 150 val, 75 test) | COPY | Basis transfer learning; ukuran kecil (~0,3 GB) |
| `dataset/labels/` | COPY, kecuali `*.cache` | Cache lama bisa menyebabkan mismatch kalau dataset berubah; YOLO membuat cache baru otomatis |
| `dataset/dataset_ft/` | BARU | Output `01_extract_frames.ipynb`: frame dari video baru untuk dianotasi (jalur mitigasi M1 — same-domain training) |
| `model/best.pt` | COPY | Checkpoint awal fine-tune (yolov8s-r3, mAP50 0,8609) |
| `model/last.pt` | TIDAK | Redundan dengan best.pt; tidak dipakai resume |
| `model/*_metrics.json, results.csv` | COPY (ke `baseline_metrics/`) | Pembanding angka sebelum vs sesudah fine-tune |
| `model/*_confusion_matrix.png, results.png` | TIDAK | Bisa dilihat kapan pun di workspace lama; bukan input eksperimen |
| `notebook/yolo_training.ipynb` | COPY → adaptasi jadi `02_yolo_finetune.ipynb` | Template training; diadaptasi: mulai dari best.pt, checkpoint disimpan ke Drive, epoch & run-name baru |
| `notebook/ocr_inference*.ipynb` | COPY → adaptasi jadi `03_inference_eval.ipynb` | Pipeline inferensi video; diadaptasi: hasil render inline di Colab + hemat VRAM |
| `notebook/gradio_ui.ipynb` | TIDAK | UI demo, tidak relevan untuk eksperimen fine-tune |
| `notebook/alpr_pipeline.py` | COPY | Inti logika deteksi+OCR; ditambah integrasi line-crossing |
| `scripts/plate_postprocess.py` | COPY | Voting D4 + locked display (unit test 5/5 lulus) — aturan tampilan tetap dipakai |
| `scripts/extract_crops.py` | COPY | Membuat capture gambar bbox — bagian dari deliverable yang diminta |
| `scripts/line_crossing.py` | BARU | Modul Line Crossing Detection (saran user): deteksi plat hanya aktif saat melewati garis yang ditetapkan; bisa on/off via flag config |
| `test_video/plat_indo_samping.mp4` | BARU (upload user) | Video uji baru berplat Indonesia, angle samping — belum ada di Drive |
| `test_video/sample_vidio_1.mp4` | COPY (18,5 MB) | Kontrol regresi — pembanding agar klaim "membaik" terukur |
| `test_video/sample_vidio_2.mp4` | TIDAK (137 MB) | Video terbesar; hanya relevan untuk histori uji 7 — tetap aman di workspace lama |
| `docs/` | COPY | PRD + phase notes = konteks keputusan; ukurannya < 10 MB |
| `runs/` | TIDAK | Artefak training lama; regenerasi otomatis saat training baru |
| `output/` lama (13 folder uji 1–7c) | TIDAK | Histori hasil — tetap utuh sebagai arsip di workspace PMLD lama; workspace baru mulai bersih dari uji_8 |
| `__pycache__/` | TIDAK | Sampah build, dibuat ulang otomatis |

## Penjelasan korelasi antar folder

1. **data.yaml ↔ dataset/ ↔ model/best.pt** — triplet wajib training. `data.yaml` menunjuk lokasi `dataset/`; `best.pt` adalah titik awal transfer learning. Salah satu tidak dicopy atau path tidak disesuaikan → training Colab gagal di verifikasi awal.

2. **test_video/ → notebook/03 → output/** — video uji masuk `test_video/`, diproses `03_inference_eval.ipynb` (baca langsung dari Drive), hasilnya ditulis ke `output/uji_N/` juga langsung ke Drive. Karena hasil tampil inline di notebook (matplotlib), evaluasi tidak butuh download — hanya download saat hasil sudah bagus.

3. **notebook/01 → dataset/dataset_ft/ → notebook/02** — alur mitigasi M1: `01_extract_frames.ipynb` mengekstrak + menyampling frame dari video baru → user anotasi → `02_yolo_finetune.ipynb` melatih dari `best.pt` dengan data gabungan (dataset lama + dataset_ft). Ini yang mengarahkan model ke domain plat Indonesia angle samping.

4. **scripts/ → notebook/** — `alpr_pipeline.py` memanggil `plate_postprocess.py` (aturan lock/override tampilan) dan `line_crossing.py` (gerbang deteksi). `extract_crops.py` dipakai notebook saat menulis deliverable. Tanpa folder scripts, pipeline inferensi rusak walau model ada.

5. **model/baseline_metrics/ → output/uji_N** — setiap uji baru mencantumkan perbandingan terhadap baseline lama, sesuai prinsip PRD: "keputusan berdasarkan angka dari log, bukan tebakan".

## Jawaban atas poin masalah di bagian Alasan Migrasi

- **Hasil dilihat langsung di Colab**: `02` dan `03` menulis checkpoint/log ke `MyDrive/PMLD_FT/` secara inkremental dan merender hasil (video, gambar bbox, tabel metrik) inline — tidak perlu download untuk menilai.
- **Beban 11 GB VRAM**: `03_inference_eval.ipynb` dirancang hemat — downscale resolusi frame, frame skipping, OCR hanya di frame terpilih (bukan tiap frame), `imgsz` disesuaikan. Fine-tune juga memakai batch size kecil.
- **Workspace berantakan/besar**: workspace baru hanya berisi input eksperimen (~0,5 GB). Semua output regenerable (runs/, output/ lama, cache, last.pt) ditinggal di workspace lama yang tetap jadi arsip utuh hasil uji 1–7c.
- **Deliverable output**: `output/uji_N/` memakai pola yang sama dengan PMLD/output sekarang: video beranotasi + folder capture gambar (bbox) + file txt + json validasi.

# Output yang diharapkan :
- output yang diharapkan adalah workspace yang rapi dan hasil trainingnya bisa langsung di load dan dilihat di google colab. shingga tidak perlu download hasilnya, ketika masih belum sesuai tinggal sesuaikan paramater dan running ulang, ketika sudah bagus baru di download.

- model dapat membaca plat indonesia dengan akurat di sudut kamera yang miring.

- saat proses selesai di trining atau di tunning dan hasilnya sesuai, yang di download itu ada vidio hasil deteksinya, capture picture yang berhasil dideteksi dan hasil pembacaan teksnya akurat, file txt dan json yang menunjukan ke validtanya (kurang lebih outputnya persis yang ada saat ini di drive PMLD/output)

- rencana migrasi ini diharapkan dapat membuatkan workspace yang lebih terstruktur jelas, rapi dan ringan, tidak terlalu kompleks dan rumit karena outputnya hanya diminta vidio hasil deteksi, capture picture hasil deteksi yang sudah ada bboxnya yang akurat, txt dan json validasinya.

-intinya goals utamanya model dapat mendeteksi plat nomor dengan akurat dari vidio angel sedikit miring 

# Notes tambahan
- mungkin saran dariku, untuk migrasi nya bisa ditambahkan modul yang memberikan garis vertikal atau horizontal pada sistem deteksinya. sehingga model hanya mendeteksi plat ketika sudah melewati salah satu garis yang sudah di tetapkan dan berhenti di garis yang sudah di tetapkan juga. atau istilahnya adalah Line Crossing Detection.