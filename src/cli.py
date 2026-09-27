"""CLI ALPR — uji gambar atau video dengan pipeline PaddleOCR (parameter uji_7c).

Contoh:
    # Video default (test_video/sample_vidio_2.mp4 — dari grup WhatsApp)
    python -m src.cli

    # Video lain / nama run khusus
    python -m src.cli --input test_video/sample_vidio_2.mp4 --run-name uji_saya

    # Gambar tunggal atau folder berisi gambar
    python -m src.cli --input foto.jpg
    python -m src.cli --input folder_gambar/

Struktur output per run (di bawah output/<run_name>/):
    annotated_images/   — gambar beranotasi (mode gambar)
    predictions_txt/    — log JSONL per deteksi
    processed_video/    — video beranotasi (mode video)
    video_captures/     — capture frame tiap CAPTURE_EVERY_S detik (mode video)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch  # noqa: F401  — harus sebelum paddle (konflik libiomp5md.dll, lihat pipeline.py)
import cv2

from .config import (ALPRConfig, OUTPUT_DIR, RUN_NAME_UJI7C, VIDEO_PATH,
                     check_video)
from .pipeline import ALPRPipeline

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VID_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def make_run_dirs(run_dir: str) -> dict:
    dirs = {
        'img': os.path.join(run_dir, 'annotated_images'),
        'txt': os.path.join(run_dir, 'predictions_txt'),
        'video': os.path.join(run_dir, 'processed_video'),
        'caps': os.path.join(run_dir, 'video_captures'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Mode gambar (replikasi sel 12 notebook)
# ---------------------------------------------------------------------------
def run_images(pipe: ALPRPipeline, paths: list[str], run_dir: str) -> None:
    dirs = make_run_dirs(run_dir)
    log_path = os.path.join(dirs['txt'], 'test_images_log.jsonl')
    rows, n_valid = [], 0

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f'  ⚠️  Tidak bisa membaca: {p}')
            continue
        dets = pipe.detect_and_read(img)
        for d in dets:
            rows.append({'file': os.path.basename(p), **d})
            n_valid += int(d['plat_valid'] and d['ocr_conf'] >= pipe.config.min_ocr_conf)
        out_img = os.path.join(
            dirs['img'], os.path.splitext(os.path.basename(p))[0] + '_annotated.jpg')
        cv2.imwrite(out_img, pipe.annotate(img, dets))
        print(f'  ✅ {os.path.basename(p)} → {len(dets)} deteksi')

    with open(log_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f'\n✅ {len(paths)} gambar dianotasi → {dirs["img"]}')
    print(f'✅ {len(rows)} deteksi dicatat → {log_path}')
    print(f'📊 Plat valid & OCR conf >= {pipe.config.min_ocr_conf}: {n_valid}/{len(rows)}')


# ---------------------------------------------------------------------------
# Mode video (replikasi sel 16 notebook)
# ---------------------------------------------------------------------------
def run_video(pipe: ALPRPipeline, video_path: str, run_dir: str) -> None:
    cfg = pipe.config
    dirs = make_run_dirs(run_dir)
    tag = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: {w}x{h} @ {fps:.1f} fps, {total} frame (~{total / fps:.1f} detik)')

    out_path = os.path.join(dirs['video'], f'{tag}_annotated.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    vlog_path = os.path.join(dirs['txt'], f'{tag}_log.jsonl')

    tracker = pipe.new_tracker() if cfg.temporal else None

    t0, idx, last_dets, n_rows, n_caps = time.time(), 0, [], 0, 0
    last_cap_t = -cfg.capture_every_s
    with open(vlog_path, 'w') as vlog:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % cfg.frame_skip == 0:
                last_dets = pipe.detect_and_read(frame, tracker=tracker, frame_idx=idx)
                for d in last_dets:
                    vlog.write(json.dumps({'frame': idx, 'timestamp_s': round(idx / fps, 2),
                                           'file': tag, **d}) + '\n')
                    n_rows += 1
            annotated = pipe.annotate(frame, last_dets)
            t_sec = idx / fps
            if t_sec - last_cap_t >= cfg.capture_every_s:
                cap_path = os.path.join(dirs['caps'], f'{tag}_t{int(t_sec):03d}s_frame{idx}.jpg')
                cv2.imwrite(cap_path, annotated)
                n_caps += 1
                last_cap_t = t_sec
            writer.write(annotated)
            idx += 1
            if idx % 100 == 0:
                print(f'  ...{idx}/{total} frame ({time.time() - t0:.0f} dtk)')

    cap.release()
    writer.release()
    mins = (time.time() - t0) / 60
    print(f'\n✅ Video beranotasi → {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)')
    print(f'✅ {n_rows} deteksi dicatat → {vlog_path}')
    print(f'✅ {n_caps} capture frame → {dirs["caps"]}/')
    print(f'   {idx} frame diproses dalam {mins:.1f} menit.')

    # Rekap plat final per track (sel 17 notebook)
    if tracker is not None:
        summary = tracker.summarize()
        print(f'\n📋 Rekap {len(summary)} track:')
        for s in summary:
            lock_mark = '🔒' if s['locked'] else '  '
            print(f"  {lock_mark} track #{s['track_id']}: {s['plat_final']} "
                  f"(votes {s['vote_count']}/{s['total_votes']}, "
                  f"stab {s['stability']:.2f}, conf {s['avg_ocr_conf']:.2f})")
        if tracker.override_events:
            print(f'\n⚠️ {len(tracker.override_events)} override (kunci berubah):')
            for ev in tracker.override_events:
                print(f"   track #{ev['track_id']} f{ev['frame']}: "
                      f"{ev['from']} → {ev['to']}")
        else:
            print('\n✅ Tidak ada override — semua kunci stabil sejak awal.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description='ALPR — deteksi plat + PaddleOCR (uji_7c)')
    ap.add_argument('--input', default=None,
                    help='Path gambar / folder gambar / video. '
                         'Default: test_video/sample_vidio_2.mp4')
    ap.add_argument('--output', default=OUTPUT_DIR,
                    help='Root folder output (default: output/)')
    ap.add_argument('--run-name', default=None,
                    help='Nama folder run di bawah output/. Default: '
                         'uji_paddleocr_<nama_file> (video) atau images_<nama> (gambar)')
    ap.add_argument('--model', default=None, help='Override path model .pt')
    args = ap.parse_args()

    cfg = ALPRConfig()
    if args.model:
        cfg.model_path = args.model
    if not os.path.exists(cfg.model_path):
        raise SystemExit(f'Model tidak ditemukan: {cfg.model_path}')

    input_path = args.input or VIDEO_PATH
    if input_path == VIDEO_PATH:
        check_video()  # pesan error jelas bila video dari WA belum ditaruh
    elif not os.path.exists(input_path):
        raise SystemExit(f'Input tidak ditemukan: {input_path}')

    # Kumpulkan target
    if os.path.isdir(input_path):
        targets = sorted(os.path.join(input_path, f) for f in os.listdir(input_path)
                         if f.lower().endswith(IMG_EXTS))
        if not targets:
            raise SystemExit(f'Tidak ada gambar di folder: {input_path}')
        mode = 'image'
    elif input_path.lower().endswith(VID_EXTS):
        targets = [input_path]
        mode = 'video'
    elif input_path.lower().endswith(IMG_EXTS):
        targets = [input_path]
        mode = 'image'
    else:
        raise SystemExit(f'Format tidak dikenal: {input_path}')

    tag = os.path.basename(input_path)
    if args.run_name:
        run_name = args.run_name
    elif mode == 'video':
        run_name = RUN_NAME_UJI7C if tag == os.path.basename(VIDEO_PATH) \
            else f'uji_paddleocr_{tag}'
    else:
        run_name = f'images_{os.path.splitext(tag)[0]}'

    run_dir = os.path.join(args.output, run_name)
    if mode == 'video' and run_name == RUN_NAME_UJI7C \
            and os.path.exists(os.path.join(run_dir, 'predictions_txt')):
        print('⚠️  Folder hasil uji_7c sudah ada dan akan DITIMPA (reproduksi).')
        print('    Gunakan --run-name lain bila ingin mempertahankannya.')

    print(f'🔧 Model: {cfg.model_path}')
    print(f'📁 Output → {run_dir}/')

    pipe = ALPRPipeline(cfg)
    if mode == 'image':
        run_images(pipe, targets, run_dir)
    else:
        run_video(pipe, targets[0], run_dir)


if __name__ == '__main__':
    main()
