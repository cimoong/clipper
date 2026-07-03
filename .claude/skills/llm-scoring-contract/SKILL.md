---
name: llm-scoring-contract
description: Gunakan skill ini setiap kali memanggil LLM untuk scoring/ranking segmen viral di ClipForge — mendefinisikan kontrak input/output JSON dan penanganan error yang deterministik.
---

# LLM Scoring Contract Rules
- Model default: pakai model Claude terbaru (mis. `claude-opus-4-8` untuk kualitas, `claude-haiku-4-5` untuk cepat/murah). Konfigurasikan lewat .env, JANGAN hardcode.
- API key HANYA dari .env (python-dotenv). Jangan pernah commit key.
- Kontrak OUTPUT wajib JSON valid dan tervalidasi (pydantic). Skema per segmen:
  `{ "id": str, "score": float 0..1, "hook": str, "reason": str, "clip_worthy": bool }`
- Minta output JSON via prompt eksplisit + parsing defensif; kalau parse gagal, retry dengan instruksi perbaikan (max N kali), lalu fallback skor 0.
- Kirim segmen dalam batch dengan konteks ringkas (start/end + text), bukan seluruh transkrip mentah, untuk hemat token.
- Deterministik: set `temperature` rendah (mis. 0.2) untuk scoring yang konsisten.
- Setiap panggilan adalah checkpoint: cache hasil scoring per segmen (keyed by hash konten) agar bisa di-retry tanpa biaya ulang.
- Log token usage dan latency; tangani rate limit dengan backoff.
- Modul harus bisa dijalankan standalone: `python -m clipforge.scoring`.
