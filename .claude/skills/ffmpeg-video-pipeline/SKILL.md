---
name: ffmpeg-video-pipeline
description: Gunakan skill ini setiap kali menulis kode FFmpeg untuk cutting, cropping, caption burning, atau encoding NVENC di proyek ClipForge.
---

# FFmpeg Pipeline Rules
- Selalu pakai -ss SEBELUM -i untuk seek cepat, lalu re-encode segmen (bukan stream copy) agar frame-accurate
- Encoding: -c:v h264_nvenc -preset p5 -rc vbr -cq 23 (GTX 1060)
- Output vertikal: 1080x1920, -pix_fmt yuv420p, audio AAC 128k
- Burn subtitle ASS: -vf "ass=subs.ass" (escape path Windows: ganti \ dengan /)
- Selalu cek exit code dan capture stderr FFmpeg untuk logging
