"""
line_crossing.py — Line Crossing Detection (PRD migrasi FR-7)

Gerbang deteksi berbasis garis: pipeline deteksi/OCR penuh hanya aktif
ketika objek (plat/kendaraan) MELEWATI garis aktivasi yang ditetapkan,
dan berhenti ketika melewati garis deaktivasi (opsional).

Tujuan:
- Mengurangi beban komputasi (OCR tidak dijalankan tiap frame — hemat VRAM).
- Membatasi deteksi hanya pada area/zona yang relevan (misal: plat baru
  terbaca jelas setelah kendaraan masuk zona tertentu).

Dipakai oleh: notebook/03_inference_eval.ipynb (flag LINE_CROSSING).
Dapat on/off via konfigurasi tanpa mengubah kode pipeline (NFR-4).

Contoh pakai:
    gate = LineCrossingGate(orientation='vertical', activate_at=0.35,
                            deactivate_at=0.85, frame_size=(1280, 720))
    active = gate.update([(cx1, cy1), (cx2, cy2)])   # pusat bbox (piksel)
    frame  = gate.draw(frame)                        # render garis + status
"""

import cv2


class LineCrossingGate:
    """Gerbang line-crossing stateless-per-frame (berbasis pusat bbox).

    Parameter
    ---------
    orientation : 'vertical' | 'horizontal'
        'vertical'   = garis tegak; posisi diukur relatif thd LEBAR frame (sumbu x).
        'horizontal' = garis mendatar; posisi relatif thd TINGGI frame (sumbu y).
    activate_at : float (0–1)
        Posisi garis aktivasi. Gerbang TERBUKA saat ada pusat bbox
        melewati garis ini.
    deactivate_at : float (0–1) | None
        Posisi garis berhenti. Gerbang TERTUTUP saat ada pusat bbox
        melewati garis ini. None = tanpa garis berhenti (gerbang tetap
        terbuka setelah aktif).
    frame_size : tuple (w, h)
        Ukuran frame piksel — untuk menggambar garis & konversi posisi.
    """

    def __init__(self, orientation='vertical', activate_at=0.5,
                 deactivate_at=None, frame_size=None):
        assert orientation in ('vertical', 'horizontal'), \
            "orientation harus 'vertical' atau 'horizontal'"
        assert 0.0 <= activate_at <= 1.0, "activate_at harus dalam rentang 0–1"
        if deactivate_at is not None:
            assert 0.0 <= deactivate_at <= 1.0, "deactivate_at harus dalam rentang 0–1"
        self.orientation = orientation
        self.activate_at = float(activate_at)
        self.deactivate_at = None if deactivate_at is None else float(deactivate_at)
        self.frame_size = frame_size
        self.active = False
        self._prev_rel = []          # posisi relatif pusat bbox frame sebelumnya
        self.cross_in_count = 0      # statistik: berapa kali gerbang terbuka
        self.cross_out_count = 0     # statistik: berapa kali gerbang tertutup

    # ---------------- internal ----------------
    def _rel_pos(self, center):
        """Posisi relatif (0–1) sebuah pusat bbox pada sumbu garis."""
        w, h = self.frame_size if self.frame_size else (1, 1)
        if self.orientation == 'vertical':
            return center[0] / max(w, 1)
        return center[1] / max(h, 1)

    @staticmethod
    def _crossed(prev, cur, line):
        """True bila segmen prev→cur memotong posisi garis (sisi manapun)."""
        return (prev - line) * (cur - line) < 0

    # ---------------- API ----------------
    def update(self, centers):
        """Perbarui status gerbang dari pusat-pusat bbox frame ini.

        centers : list of (cx, cy) dalam piksel.
        return  : bool — True bila gerbang aktif (deteksi penuh boleh jalan).
        """
        cur = [self._rel_pos(c) for c in centers]

        # Deteksi perpotongan garis antara frame sebelumnya dan sekarang
        for p in self._prev_rel:
            for q in cur:
                if self._crossed(p, q, self.activate_at):
                    if not self.active:
                        self.cross_in_count += 1
                    self.active = True
                if self.deactivate_at is not None and self._crossed(p, q, self.deactivate_at):
                    if self.active:
                        self.cross_out_count += 1
                    self.active = False

        self._prev_rel = cur
        return self.active

    def draw(self, frame):
        """Gambar garis aktivasi (hijau), garis berhenti (merah), dan status."""
        h, w = frame.shape[:2]
        if self.frame_size is None:
            self.frame_size = (w, h)

        if self.orientation == 'vertical':
            ax = int(self.activate_at * w)
            cv2.line(frame, (ax, 0), (ax, h), (0, 255, 0), 2)
            cv2.putText(frame, 'START', (ax + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if self.deactivate_at is not None:
                dx = int(self.deactivate_at * w)
                cv2.line(frame, (dx, 0), (dx, h), (0, 0, 255), 2)
                cv2.putText(frame, 'STOP', (dx + 5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            ay = int(self.activate_at * h)
            cv2.line(frame, (0, ay), (w, ay), (0, 255, 0), 2)
            cv2.putText(frame, 'START', (10, ay - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if self.deactivate_at is not None:
                dy = int(self.deactivate_at * h)
                cv2.line(frame, (0, dy), (w, dy), (0, 0, 255), 2)
                cv2.putText(frame, 'STOP', (10, dy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status = 'GATE: AKTIF' if self.active else 'GATE: standby'
        color = (0, 255, 0) if self.active else (128, 128, 128)
        cv2.putText(frame, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def stats(self):
        """Statistik penggunaan gerbang (untuk log uji)."""
        return {
            'orientation': self.orientation,
            'activate_at': self.activate_at,
            'deactivate_at': self.deactivate_at,
            'cross_in': self.cross_in_count,
            'cross_out': self.cross_out_count,
            'active': self.active,
        }
