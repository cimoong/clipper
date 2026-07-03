---
name: fastapi-conventions
description: Gunakan skill ini setiap kali menulis endpoint, router, atau schema FastAPI di ClipForge — untuk struktur, validasi, dan menjalankan pipeline berat secara aman.
---

# FastAPI Conventions
- Type hints wajib di semua signature; request/response model pakai pydantic (bukan dict mentah).
- Struktur: router per domain di `src/clipforge/api/routers/`, dependency di `deps.py`, schema di `schemas.py`.
- Config (port, path, API key) lewat .env via pydantic-settings/dotenv; JANGAN hardcode.
- Pipeline berat (Whisper/NVENC) TIDAK boleh jalan sinkron di request handler — jadwalkan sebagai background job dan kembalikan `job_id`; sediakan endpoint polling status.
- Hormati constraint GPU serial: gunakan queue/worker tunggal untuk stage GPU, jangan spawn job GPU paralel dari beberapa request.
- Setiap stage adalah checkpoint yang bisa di-retry; endpoint retry harus idempoten terhadap `job_id`.
- Error handling: raise `HTTPException` dengan status yang tepat; jangan bocorkan stderr/stacktrace mentah ke client — log detail di server.
- Sertakan endpoint `/health`; validasi keberadaan FFmpeg & model saat startup.
- Modul harus bisa dijalankan standalone: `python -m clipforge.api` (uvicorn).
