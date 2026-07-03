# ClipForge

Local-first AI video clipper: URL → transcribe → LLM scoring → reframe 9:16 → captions → MP4.

## Quickstart

1. **Prereqs:** Windows 11, Python 3.11, NVIDIA GPU (CUDA), FFmpeg on PATH, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. **Get the code:** clone this repo and open a terminal in the project folder.
3. **Configure:** copy `.env.example` to `.env` and set `GEMINI_API_KEY` (from https://aistudio.google.com).
4. **Run:** double-click `run.bat` (or run it from a terminal).
5. On first launch it runs `uv sync` to install dependencies (this can take a while).
6. It starts the server on **http://localhost:8420** and opens your default browser.
7. Keep the console window open — it streams live logs. Press **Ctrl+C** to stop.

> First transcription run downloads the Whisper model; give it a moment.
