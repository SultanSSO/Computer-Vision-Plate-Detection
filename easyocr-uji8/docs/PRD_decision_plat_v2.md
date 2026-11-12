# PRD — Decision: Plat v2 (Stabilisasi Tampilan / Locked-Plate Display)

**Proyek:** PMLD — Indonesian License Plate Recognition (ALPR)
**Peran:** AI Engineer | **Tanggal:** 22 Sep 2026
**Status:** Draft — menunggu persetujuan tim sebelum eksekusi
**Dokumen induk:** PRD_decision_plat.md (v1), PHASE4_SETUP_NOTES.md
**Melanjutkan:** hasil uji 3 (D3 character-position correction + D4 temporal voting)

---

## 0. Ringkasan Eksekutif

Uji 3 menunjukkan D3+D4 **sudah memperbaiki kualitas pembacaan** — namun timbul satu masalah
UX baru: **teks plat di video beranotasi masih berubah-ubah antar-frame (flip-flop)**, bahkan
ketika OCR sudah benar. Contoh: track yang sama menampilkan `GX15OGJ` → `GXI50GJ` → `GX15OGJ`
pada frame berurutan.

Penyebab sudah teridentifikasi di kode (bukan di model): fungsi `annotate()` menampilkan
`plat_corrected` **per-frame**, sedangkan hasil temporal voting (D4) hanya dipakai untuk
**rekap pasca-proses** di sel rekap — tidak pernah dipakai untuk tampilan video.

Dokumen ini memutuskan mekanisme **Locked-Plate Display**: setelah sebuah track mengumpulkan
cukup bukti, teks yang ditampilkan untuk track itu **dikunci (locked)** dan tidak berubah lagi
sampai track berakhir — kecuali ada bukti kuat yang memaksa koreksi.

---

## 1. Kondisi Saat Ini (Baseline Terukur)

### 1.1 Diagnosis akar masalah (dari kode uji 3)

**Sel 11 — `annotate()` (tampilan per-frame):**
```python
label = format_plate(d['plat_corrected'], d['domain'])   # ← hasil frame INI saja
if d.get('is_stable'):
    label += ' ✓'                                         # ← hanya penanda, teks tetap per-frame
```

**Sel 17 — loop video:**
```python
last_dets = detect_and_read(frame, tracker=tracker, frame_idx=idx)
...
annotated = annotate(frame, last_dets)   # ← pakai deteksi mentah frame ini
```

**Sel 19 — rekap (pasca-proses):**
```python
final = tracker.get_final_plates()   # ← voting D4 baru dipakai DI SINI, bukan di video
```

**Kesimpulan:** hasil voting yang sudah benar tidak pernah "dialirkan balik" ke tampilan.
`vote_count` dan `is_stable` sudah ada di setiap deteksi, tetapi hanya jadi hiasan (tanda ✓).

### 1.2 Gejala yang dilaporkan

| Gejala | Frekuensi | Dampak |
| :-- | :-- | :-- |
| Teks plat berganti antar-frame pada kendaraan yang sama | Sering | Video demo terlihat tidak andal; sulit diverifikasi mata |
| Tampilan menampilkan hasil salah setelah sebelumnya benar | Kadang | Kepercayaan terhadap hasil turun ("tadi benar kok berubah?") |
| Tanda ✓ muncul-hilang mengikuti `vote_count` per-frame | Sering | Indikator stabilitas tidak konsisten dengan teks |

### 1.3 Yang sudah berfungsi (jangan dirusak)

- D3 character-position correction (`GXI50GJ → GX15OGJ`) ✅
- D4 tracking + majority vote (rekap per-track di sel 19) ✅
- Schema JSONL v2 (`track_id`, `vote_count`, `is_stable`) ✅
- Akurasi deteksi & cakupan OCR (uji 2: 96,4% deteksi menghasilkan teks) ✅

---

## 2. Keputusan (Decision)

| # | Keputusan | Status |
| :- | :-- | :-- |
| V1 | **Locked-Plate Display**: teks yang ditampilkan per track = teks hasil voting track itu (`tracker.best(track_id)`), **bukan** `plat_corrected` per-frame. Teks per-frame tetap dicatat di log apa adanya | Diusulkan |
| V2 | **Aturan kunci (lock rule)**: sebuah track menampilkan teks terkunci setelah `vote_count ≥ VOTE_MIN` (default 3). Sebelum terkunci, tampilkan kandidat terbaik sementara dengan penanda visual berbeda (mis. `…` atau warna pudar) | Diusulkan |
| V3 | **Override rule (anti-stuck)**: teks terkunci BOLEH berubah hanya jika kandidat baru mengumpulkan `vote_count` ≥ 2× kandidat terkunci **dan** `avg_ocr_conf` kandidat baru lebih tinggi. Mencegah plat salah terkunci selamanya | Diusulkan |
| V4 | **Log tidak berubah**: log JSONL tetap mencatat `plat`, `plat_corrected` per-frame (audit trail). Ditambah 2 field: `plat_display` (teks yang ditampilkan) dan `display_locked` (bool) — backward compatible (v2.1) | Diusulkan |
| V5 | **Berlaku untuk video saja**: uji citra (sel 13) tetap menampilkan hasil per-citra — tidak ada konsep track di citra tunggal | Diputuskan |
| V6 | **Indikator visual diganti**: tanda ✓ diganti status eksplisit — `LOCKED` (teks terkunci, hijau), `…` (belum stabil, oranye), tetap merah untuk tidak valid | Diusulkan |
| V7 | **Tidak mengubah model/training**: murni perubahan post-processing & tampilan. `best.pt`/`best_ft.pt`, parameter D3/D4 tidak disentuh | Diputuskan |

---

## 3. Ruang Lingkup

### 3.1 Dalam ruang lingkup
- Modifikasi `PlateTracker` (sel 11): tambah metode `best(track_id)` → teks terkunci per track + logika override (V3)
- Modifikasi `detect_and_read()` / loop video (sel 17): enrich deteksi dengan `plat_display` + `display_locked`
- Modifikasi `annotate()` (sel 11): tampilkan `plat_display` + indikator status (V6)
- Ekstensi schema JSONL → v2.1: field `plat_display`, `display_locked`
- Uji 4: jalankan ulang notebook pada `sample_video_1min_fix.mp4` → `output/uji_4_sample_video_1min_fix.mp4/`
- Update dokumentasi (PHASE4 notes + tabel target)

### 3.2 Di luar ruang lingkup (tidak berubah)
- Model detector (`best.pt` / `best_ft.pt`) dan recognizer EasyOCR
- Parameter D3 (`CHAR_FIX`, template) dan D4 (`IOU_TRACK`, `MAX_FRAME_GAP`, `VOTE_MIN`)
- Pipeline citra test (sel 13) & evaluasi OCR (sel 15)
- Fase A.2 pelabelan manual, Fase B/C fine-tuning — tetap jalur terpisah
- Integrasi FastAPI / Backend (Phase 5)

---

## 4. Desain Teknis

### 4.1 Alur per-frame (sesudah)

```
frame idx (kelipatan FRAME_SKIP)
  → detect_and_read() → deteksi mentah {plat, plat_corrected, ocr_conf, ...}
  → tracker.update(idx, dets)          ← voting D4 (tidak berubah)
  → tracker.best(track_id) per deteksi ← BARU (V1–V3)
        ├─ track belum terkunci → plat_display = kandidat sementara, display_locked=False
        └─ track terkunci       → plat_display = teks terkunci,   display_locked=True
  → annotate(frame, dets) menampilkan plat_display
  → log JSONL v2.1: semua field lama + plat_display + display_locked
```

### 4.2 Logika `PlateTracker.best()` (V2 + V3)

```python
def best(self, track_id):
    """Teks yang harus ditampilkan untuk track. Return (teks, locked: bool)."""
    tr = self._get(track_id)
    if not tr['votes']:
        return '', False
    counter = Counter(v[0] for v in tr['votes'])
    top_plate, top_votes = counter.most_common(1)[0]

    if tr.get('locked_plate'):
        locked = tr['locked_plate']
        locked_votes = counter.get(locked, 0)
        # V3 — override: kandidat lain menang telak + confidence lebih tinggi
        if top_plate != locked and top_votes >= 2 * max(locked_votes, 1):
            if self._avg_conf(tr, top_plate) > self._avg_conf(tr, locked):
                tr['locked_plate'] = top_plate
                return top_plate, True
        return locked, True

    # V2 — lock rule: kunci setelah cukup bukti
    if top_votes >= self.vote_min:
        tr['locked_plate'] = top_plate
        return top_plate, True
    return top_plate, False   # kandidat sementara
```

### 4.3 Schema JSONL v2.1 (backward compatible)

```json
{
  "frame": 300, "timestamp_s": 10.0, "file": "sample_video_1min_fix.mp4",
  "bbox": [1207, 785, 1290, 807], "det_conf": 0.81,
  "plat": "GXI50GJ", "plat_corrected": "GX15OGJ",
  "plat_valid": true, "domain": "UK", "template": "UK LLDDLLL",
  "ocr_conf": 0.72, "skipped_small": false,
  "track_id": 7, "vote_count": 5, "is_stable": true,
  "plat_display": "GX15OGJ", "display_locked": true
}
```

> Catatan: pada contoh di atas, frame itu membaca `GXI50GJ` (mentah), D3 mengoreksi ke
> `GX15OGJ`, dan karena track #7 sudah terkunci pada `GX15OGJ`, tampilan tetap `GX15OGJ`.
> Jika frame itu membaca salah (mis. `6X15OGJ`), `plat_display` **tetap** `GX15OGJ` — inilah
> perilaku yang diminta.

### 4.4 Indikator visual (V6)

| Status | Warna bbox | Sufiks label | Arti |
| :-- | :-- | :-- | :-- |
| `display_locked=True`, valid | Hijau | `▮` (atau tanpa sufiks) | Teks terkunci — tidak akan berubah |
| `display_locked=False`, ada kandidat | Oranye | `…` | Belum stabil — kandidat sementara |
| Tidak ada teks / tidak valid | Merah | — | Deteksi tanpa pembacaan |

---

## 5. Konfigurasi (tambahan di sel 7)

```python
# ---- Locked-Plate Display (PRD v2) ----
DISPLAY_LOCK   = True    # V1: tampilkan teks hasil voting, bukan per-frame
LOCK_MIN       = 3       # V2: vote minimum sebelum teks dikunci (= VOTE_MIN)
OVERRIDE_RATIO = 2.0     # V3: kandidat baru harus 2x vote untuk mengganti kunci
RUN_NAME       = f'uji_4_{VIDEO_TAG}'   # uji berikutnya = uji_4
```

---

## 6. Target & Kriteria Keberhasilan

| Metrik | Baseline uji 3 | Target uji 4 | Cara ukur |
| :-- | --: | --: | :-- |
| Flip-flop tampilan (pergantian teks per track setelah lock) | sering (tiap frame bisa beda) | **0** setelah `display_locked=True` | hitung transisi `plat_display` per track dari log |
| Track terkunci dengan teks = hasil voting akhir | — | **100%** | bandingkan `plat_display` saat locked vs `get_final_plates()` |
| Override benar (lock salah → terkoreksi) | — | tercatat di log bila terjadi | field perubahan `locked_plate` (opsional: log event) |
| Akurasi/CER (sel 15) | hasil uji 3 | **tidak menurun** | sel evaluasi (regresi check) |
| Video 100% tanpa error | ✅ | ✅ jaga | sel video |

> Metrik utama kasus ini: **nol pergantian teks tampilan setelah lock**, tanpa regresi akurasi.

---

## 7. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| :-- | :-- | :-- |
| Lock terlalu dini pada teks salah (OCR salah konsisten) | Teks salah tampil terus | V3 override rule; naikkan `LOCK_MIN` bila sering terjadi; track salah tetap bisa diverifikasi dari log per-frame |
| Override terlalu agresif | Flip-flop kembali | `OVERRIDE_RATIO ≥ 2` + syarat confidence; override dicatat untuk evaluasi |
| Track pecah (IoU gagal) → kunci baru tiap pecahan | Plat sama tampil beda di dua track | Sudah dimitigasi `MAX_FRAME_GAP=15`; bila masih pecah, evaluasi `IOU_TRACK` (di luar PRD ini) |
| Pengguna bingung teks tampilan ≠ teks log per-frame | Salah tafsir hasil | Dokumentasi: `plat` = bacaan frame itu, `plat_display` = yang ditampilkan; keduanya ada di log |
| Runtime bertambah | Loop video melambat | `best()` O(votes per track), dijalankan hanya pada frame terproses (1/3) — dampak diperkirakan < 1% |

---

## 8. Rencana Kerja

| Langkah | Isi | Estimasi |
| :-- | :-- | :-- |
| 1 | Tambah `PlateTracker.best()` + state `locked_plate` + override (V2, V3) di `scripts/plate_postprocess.py` + unit test lokal | 0,5 jam |
| 2 | Update notebook: sel 7 (config), sel 11 (`best()` inline + `annotate`), sel 17 (enrich `plat_display`), sel 19 (metrik flip-flop) | 0,5 jam |
| 3 | Sinkron ke Drive → **uji 4** di Colab (T4, Run all) | ~20–40 menit runtime |
| 4 | Verifikasi: 0 flip-flop setelah lock, tidak ada regresi CER/akurasi | 0,5 jam |
| 5 | Update `PHASE4_SETUP_NOTES.md` + status tabel target | 0,5 jam |

---

## 9. Keputusan yang Dibutuhkan dari Tim

1. Setujui V1–V4 (locked display + override rule + schema v2.1)?
2. Ambang lock `LOCK_MIN = 3` dan `OVERRIDE_RATIO = 2.0` — cukup konservatif?
3. Indikator visual V6 (hijau=LOCKED / oranye=… / merah=invalid) — disetujui untuk demo?
4. Setelah uji 4 lolos, apakah hasil ini yang dipakai untuk **demo/dashboard**, atau menunggu video plat Indonesia (pertanyaan terbuka dari PRD v1 §9)?

---

## 10. Relasi dengan PRD v1

| Jalur | Status | Dokumen |
| :-- | :-- | :-- |
| Peningkatan model (Fase A dataset → B detector → C recognizer) | Menunggu pelabelan manual (A.2) | PRD_decision_plat.md (v1) |
| **Stabilisasi tampilan (PRD ini)** | Siap dieksekusi — tidak tergantung pelabelan | **PRD_decision_plat_v2.md** |

Kedua jalur independen: PRD v2 bisa dikerjakan dan diuji **sekarang** tanpa menunggu v1,
karena tidak menyentuh model maupun dataset.

---

*Disusun berdasarkan: laporan pengguna ("ketika OCR sudah benar kadang berubah lagi"), analisis kode uji 3
(`annotate()` menampilkan `plat_corrected` per-frame; voting D4 hanya dipakai di rekap sel 19),
PRD_decision_plat.md §2 D4/D7, dan PHASE4_SETUP_NOTES.md.*
