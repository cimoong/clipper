---
name: whisper-transcription
description: Gunakan skill ini setiap kali menulis kode transkripsi/ASR dengan Whisper (faster-whisper) di ClipForge — memuat model, mengatur VRAM, dan menghasilkan segmen bertimestamp.
---

# Whisper Transcription Rules
- Constraint VRAM GTX 1060 6GB: model MAX `small`, compute_type=`int8` (atau `int8_float16`). JANGAN pakai medium/large.
- Pakai `faster-whisper` (CTranslate2), bukan openai-whisper, untuk efisiensi memori.
- Proses GPU harus SERIAL: hanya satu job Whisper aktif pada satu waktu; lindungi dengan lock/queue, jangan paralel dengan stage GPU lain (NVENC).
- Selalu load model sekali dan reuse; jangan re-instantiate per file. Bebaskan model saat idle bila perlu VRAM.
- Aktifkan `word_timestamps=True` untuk kebutuhan burning caption yang frame-accurate.
- Set `vad_filter=True` untuk membuang silence dan menstabilkan segmentasi.
- Output kanonik: list segmen `{start, end, text, words[]}` — simpan sebagai checkpoint JSON agar stage bisa di-retry tanpa transkripsi ulang.
- Tangani audio kosong/terlalu pendek secara eksplisit; log durasi dan bahasa terdeteksi.
- Modul harus bisa dijalankan standalone: `python -m clipforge.transcription`.
