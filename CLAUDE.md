# ClipForge — Project Instructions

## Context
Baca docs/PRD-ClipForge.md dan docs/PRD-ADDENDUM.md sebelum mengerjakan task apa pun.

## Environment
- Windows 11, Python 3.11, GPU NVIDIA GTX 1060 6GB (CUDA)
- Constraint VRAM: model Whisper max `small` INT8; proses GPU harus serial
- FFmpeg tersedia di PATH; encoding pakai h264_nvenc

## Conventions
- Python: type hints wajib, ruff format, struktur src/clipforge/
- Config via .env (python-dotenv), JANGAN hardcode API key
- Setiap modul harus bisa dites standalone via `python -m clipforge.<modul>`
- Error handling: setiap stage pipeline punya checkpoint, bisa di-retry

## Testing
- Video uji: data/samples/test_5min.mp4 (siapkan manual)
- Jalankan: pytest tests/ -v
