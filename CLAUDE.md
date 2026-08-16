# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TranscriMate: audio/video transcription with speaker diarization, plus a chatbot over the resulting transcript. FastAPI backend (`backend/main.py`) + Vue 3 SPA (`frontend/`). Code, comments, logs and UI are in French — keep new code consistent with that.

## Commands

Backend (run from `backend/`, so `main.py` can `import temp_manager` / `session_manager`):

```bash
pip install -r requirements.txt          # torch==2.8.0+cu126 — needs the CUDA wheel index
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (from `frontend/`):

```bash
npm install
npm run serve     # dev server on 0.0.0.0:8080
npm run build     # -> frontend/dist
npm run lint      # eslint via vue-cli-service (the only linter in the repo)
```

Whole stack: `docker-compose up --build` (backend :8000, frontend :8080). The backend image is CUDA-based; pass `--gpus all` when running it standalone.

There is no test suite. The root `test_*.py` files are throwaway manual scripts, not pytest: `test_mode2.py` drives `/transcribe_streaming/` against a *running* server (its file path is Windows-style — fix to `backend/Multimedia/dialogue_test.mp3` on Linux), and `test_streaming_syntax.py` imports a `process_streaming_audio` function that no longer exists. Verify changes with real requests instead:

```bash
curl http://localhost:8000/health/                            # model load state + CUDA info
curl -F "file=@backend/Multimedia/dialogue_test.mp3" http://localhost:8000/transcribe_simple/
open http://localhost:8000/docs                               # Swagger, tagged by mode
```

## Environment variables

Backend reads exactly two, via `python-dotenv` from `backend/.env`:

- `HF_TOKEN` — **required**; startup raises without it (gates Whisper, pyannote, Chocolatine downloads).
- `OPENAI_API_KEY` — optional; only needed for the `gpt-4o-mini` chat model.

Frontend uses `VUE_APP_API_URL` and `VUE_APP_WEBSOCKET_URL` (see `.env.development` / `.env.production`), baked in at build time by webpack.

Note the README and `API_DOCUMENTATION.md` predate the current `main.py` and name different variables (`HuggingFace_API_KEY`, `OPENAI_API_KEY_MCF`) and endpoints (`/uploadfile/`, `/initialize/`, `/device_type/`, `/keep_alive/`, `/diarization/`, `/streaming_audio/`) that no longer exist. `nginx/nginx.conf` proxies that same stale route list. Trust `backend/main.py`.

## Architecture

### Models are process-global singletons

`load_core_models()` runs once in the FastAPI `lifespan` and populates four module-level globals: `Transcriber_Whisper` (whisper-large-v3-turbo, for file transcription), `Transcriber_Whisper_Light` (whisper-base, for the live WebSocket), `diarization_model` (pyannote speaker-diarization-3.1), and `Chocolatine_pipeline` (local French LLM; failure to load is tolerated and leaves it `None`). Every endpoint null-checks these. Startup is slow and VRAM-heavy — prefer `--reload` off when iterating on model code, and expect the first request after boot to block.

Model choice is **hardcoded** in `load_core_models()`. `POST /settings/` writes to the `current_settings` dict but only `task` (`transcribe` | `translate`) is ever consumed (passed as `generate_kwargs`); `model` and `lang` are accepted and ignored, even though the frontend offers a Whisper model dropdown.

### Three transcription modes (the API's organizing principle)

1. `POST /transcribe_simple/` — synchronous; diarize the whole file, transcribe each turn, return one JSON blob.
2. `POST /transcribe_streaming/?session_id=` — the mode the UI actually uses. Returns a `StreamingResponse` of SSE-shaped `data: {json}\n\n` lines (declared `media_type="text/plain"`). The client reads it with `fetch` + `body.getReader()` and switches on `data.status`: `started` → `audio_ready` → `diarization_start` → `diarization_done` → repeated `transcribing`/`segment_done` → `audio_urls_ready` → `completed`. Adding a status means editing both the generator in `main.py` and the `if/else` chain in `App.vue#uploadFile`.
3. `WS /live_transcription/` — client streams raw 16 kHz mono Int16 PCM; the server buffers 2 s, transcribes with the light model, and sends back `{status: "transcription", text, ...}`.

`WS /progress/` is separate: it registers a callback on the module-global `ProgressCapture` and pushes diarization stage progress (`segmentation` / `speaker_counting` / `embeddings`) to the browser. Because `progress_capture` is global, all connected clients receive every job's progress. The progress values are **simulated** (fixed step sequences / a background `asyncio` task), not real pyannote telemetry — `CustomProgressHook` exists but is not wired into the diarization calls.

### Two overlapping temp-file layers

- `temp_manager.TempFileManager` — per-request scratch files in the OS temp dir, extension-allowlisted, cleaned up by `async_temp_manager_context` when the request ends.
- `session_manager.SessionManager` (global `session_manager`, rooted at the **relative** path `backend/temp/`, so the CWD you launch uvicorn from matters) — per-session directories that outlive the request, so the browser can fetch segment audio afterwards. A daemon thread purges sessions idle for 24 h plus orphaned UUID directories.

`GET /temp_audio/{filename}` bridges them: it parses the session UUID out of the `streaming_{session_id}_{uuid}.wav` filename convention, then falls back to `backend/temp/` and the system temp dir. Renaming that convention breaks audio playback in the UI.

### Frontend

`frontend/src/App.vue` is the whole application — ~3900 lines, Options API, tabs selected by `activeTab` (`streaming` / chatbot / live). It owns transcription state, dark mode, speaker colors, per-segment playback, the ASCII VU-meter/spectrogram, and the WebSocket clients. `components/` holds only leaf pieces (`QuestionForm.vue` calls `/ask_question/` itself; `MyDictaphone.vue` just emits `record`/`stop`). Tailwind is configured but `content: []`, so its utilities are not generated — styling is the scoped CSS at the bottom of `App.vue`.

Known drift to be aware of when touching these paths:

- `uploadFile()` gates session creation on `this.selectedMode`, which is not in `data()` — so it is always `undefined`, no session is created client-side, and the backend creates one implicitly.
- The chat toggle sets `settings.chat_model` to `'gpt-4'`, but `/ask_question/` only accepts `'chocolatine'` and `'gpt-4o-mini'` and returns an "unsupported model" string otherwise.

### RAG

`backend/RAG.py` builds a FAISS index over `backend/Multimedia/Use_Cases/*.txt` and is a standalone CLI (`python RAG.py "question"`). It is **not** imported by `main.py` — `/ask_question/` simply stuffs the full transcript into the prompt. Note the loader globs `.txt` while the directory contains `.md` files, so the index is currently empty.

## Deployment

`k8s-config/` targets a Scaleway registry (GPU resources commented out); `uvicorn.service` / `npm_frontend.service` are systemd units for an Azure VM with paths under `/home/azureuser/SourceCode/Transcription` — none of these paths match this checkout, treat them as reference only.
