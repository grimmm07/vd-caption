# 🎬 Video Caption Prototype

A small, self-contained local prototype that:

1. Uploads **one** video.
2. Sends it to the **Google Gemini API** for transcription.
3. Produces a **timestamped transcript** with strict validation.
4. Lets you **edit** the captions.
5. Generates **SRT** and **WebVTT** subtitle files.
6. Uses **FFmpeg** to **burn the captions into a new video** (and, optionally, a
   version with a *selectable* subtitle track).
7. Lets you **preview** and **download** everything (captioned video, SRT, VTT,
   and the transcript as structured JSON).
8. Includes a **batch cost & quota estimator** for planning the full ~450-video
   project.

This is a proof of concept intended to process ~450 videos (average 1.5–3
minutes each). It processes one video at a time, but the transcription service
is structured so batch processing can be layered on later.

---

## 1. Architecture

Chosen stack: **Python + Streamlit + `google-genai` + FFmpeg** — the "preferred
option" from the brief. The working directory was empty (no pre-existing app),
so the prototype was created fresh in this folder. The environment already had
`google-genai`, `streamlit`, and `pydantic` installed, which made this the
lowest-friction reliable choice.

```text
video-caption-prototype/
├── app.py                     # Streamlit UI + full user flow + estimator
├── config.py                  # Env-driven settings (API key, model, limits)
├── conftest.py                # Makes the project importable under pytest
├── services/
│   ├── gemini_service.py      # Upload → transcribe → validate (google-genai)
│   ├── subtitle_service.py    # seconds→SRT/VTT + document generation
│   └── video_service.py       # FFmpeg/FFprobe detect, probe, burn, mux
├── models/
│   └── transcript.py          # Strict Pydantic models + repair logic
├── utils/
│   ├── files.py               # Filename sanitization, unique work dirs
│   └── logging_config.py      # Structured logging (never logs the key)
├── tests/                     # pytest suite (Gemini fully mocked)
├── temp/                      # Per-upload working dirs (git-ignored)
├── outputs/                   # Reserved output dir (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

**Data flow**

```
upload → temp/<uuid>/<safe_name>  ──ffprobe──> metadata (duration/res/codecs)
      → Gemini Files API upload → generate_content(JSON schema)
      → parse → repair_segments → Transcript (validated) → editable table
      → SRT / WebVTT / JSON  → FFmpeg burn-in → captioned.mp4 → preview/download
```

The **original uploaded file is never modified** — FFmpeg always writes a new
output file.

---

## 2. Prerequisites

- **Python 3.10+** (developed and tested on 3.12).
- **FFmpeg and FFprobe** on your `PATH` (required for probing and rendering).
- A **Google Gemini API key** (from <https://aistudio.google.com/apikey>).

---

## 3. FFmpeg installation

FFmpeg ships **both** `ffmpeg` and `ffprobe`; the app checks for both at
startup and disables caption rendering if either is missing.

**Windows**
- Winget: `winget install --id=Gyan.FFmpeg -e`
- or Chocolatey: `choco install ffmpeg`
- or download a build from <https://www.gyan.dev/ffmpeg/builds/>, unzip, and add
  the `bin` folder to your `PATH`.
- Verify: `ffmpeg -version` and `ffprobe -version`

**macOS**
- `brew install ffmpeg`

**Linux (Debian/Ubuntu)**
- `sudo apt update && sudo apt install ffmpeg`

---

## 4. Python setup (Windows PowerShell)

```powershell
cd video-caption-prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and paste your GEMINI_API_KEY
streamlit run app.py
```

macOS / Linux equivalent:

```bash
cd video-caption-prototype
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env
streamlit run app.py
```

Then open <http://localhost:8501>.

---

## 5. Environment variables

| Variable         | Required | Default            | Purpose                                        |
|------------------|----------|--------------------|------------------------------------------------|
| `GEMINI_API_KEY` | yes\*    | —                  | Your Gemini key. Never commit it.              |
| `GEMINI_MODEL`   | no       | `gemini-2.5-flash` | Which model to use (see note below).           |
| `MAX_UPLOAD_MB`  | no       | `500`              | Upload size cap.                               |
| `DEBUG`          | no       | `0`                | `1` enables verbose logs (may log transcript). |

\* The UI and the cost estimator work without a key; **transcription** needs one.

### Which model identifier?

The official `google-genai` SDK **does not hard-code a model** — you pass any
identifier string, and the authoritative list of what *your* key can use comes
from `client.models.list()`. This prototype therefore:

- defaults to **`gemini-2.5-flash`** (a stable, widely-available model that
  supports audio/video understanding and JSON output), and
- provides a **"List models my key can use"** button in the sidebar that calls
  `client.models.list()` and shows every model supporting `generateContent`.

To list from the command line:

```powershell
python -c "import os; from google import genai; [print(m.name) for m in genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.list()]"
```

Set `GEMINI_MODEL` in `.env` to any identifier that appears there (e.g. a newer
flash/pro model if your account has access).

---

## 6. Supported video formats

`mp4`, `mov`, `webm`, `mkv`, `avi`, `m4v`. Uploads are size-limited by
`MAX_UPLOAD_MB`. Filenames are sanitized and each upload gets an isolated
working directory to prevent path traversal or collisions.

---

## 7. How transcription works

1. The file is uploaded via the **Gemini Files API** and polled until `ACTIVE`.
2. The model is asked (with a `system_instruction` + JSON `response_schema`) to
   return timestamped segments. The prompt explicitly instructs it to:
   preserve punctuation, detect the language, **not** summarise/rewrite, avoid
   hallucinating during silence, and keep segments short (≤ ~2 lines).
3. The JSON is parsed and passed through `repair_segments` (sort, de-overlap,
   drop blanks, re-index) and then **strictly validated** by the `Transcript`
   Pydantic model, which guarantees: `end > start`, chronological order, and no
   overlaps.
4. Malformed JSON or an unusable structure produces a **clear error**, not a
   crash. HTTP 429 / quota errors are surfaced distinctly and are **not**
   retried endlessly (no duplicate paid calls).

The remote uploaded file is **deleted** from Gemini after transcription.

### Timestamp accuracy & drift

Gemini *estimates* timestamps as it transcribes; it does not measure them
against a clock. Over a longer clip these small errors accumulate, so captions
can start in sync but **drift** later. Two mitigations are built in:

- **Duration grounding (always on):** the real media duration (from FFprobe) is
  sent in the prompt so the model anchors its timeline to the true length.
- **High-accuracy sync mode (opt-in checkbox):** the audio is split into short
  windows (~45 s), each transcribed independently, and the timestamps are
  offset back to absolute time. Because each window is short, drift can't
  accumulate across the whole video. This costs **~1 API request per window**
  (e.g. a 3-minute video ≈ 4 requests) instead of 1, so mind your daily quota —
  the checkbox shows the estimated request count before you run it.

---

## 8. How captions are added

FFmpeg is invoked via a **subprocess argument array** (never a shell string),
so filenames can never be interpreted as shell syntax.

- **Burned-in (required):** re-encodes to **H.264 / AAC**, white text with a
  dark semi-transparent box, bottom-centre, safe margin, font size scaled to the
  video height. Audio is preserved (copied to AAC); if the source has no audio
  track, `-an` is used. Output is a new `captioned.mp4`.
- **Selectable subtitles (optional):** muxes the SRT as a `mov_text` track into
  an MP4 without re-encoding (`captioned_soft.mp4`).

Rendering progress is parsed from FFmpeg's `-progress` output and shown as a
progress bar.

---

## 9. Privacy considerations

- When you click **Generate transcript**, the **video is uploaded to Google**
  (the Gemini Files API) for processing. This is the only external destination;
  the app sends your video nowhere else.
- The API key is read from the environment and **never logged**. Complete
  environment variables are never printed.
- Transcript text is only logged when `DEBUG=1`.
- Uploaded and generated media live in git-ignored `temp/`/`outputs/` and are
  cleaned per-upload; the remote Gemini file is deleted after use.

---

## 10. Gemini quota considerations

Known starting quota (free tier, subject to change):

- **5 requests / minute**
- **250,000 tokens / minute**
- **20 requests / day**

The app makes **one request per video**. On a `429`/`RESOURCE_EXHAUSTED`
response it shows a clear "quota may be reached" message, uses only a small
capped retry for *transient* (non-quota) errors, and never auto-retries a paid
request after an uncertain outcome. The **batch estimator** computes the
minimum number of days: with 450 videos at 20 requests/day → **⌈450/20⌉ = 23
days** minimum.

---

## 11. Known limitations

- FFmpeg/FFprobe must be installed separately; caption rendering is disabled
  until they are present.
- Timestamp accuracy depends on the model and can **drift** over longer clips
  (see "Timestamp accuracy & drift"). Use **High-accuracy sync mode** for
  better timing, or review/edit timestamps in the table before rendering.
  Editing text is fully supported; editing timestamps is optional.
- One video at a time (by design for this POC).
- Very long videos may hit the Files API processing timeout (configurable in
  `gemini_service.py`).
- The default model (`gemini-2.5-flash`) is a sensible starting point, not a
  guaranteed "best" — verify against `models.list()` for your account.

---

## 12. Running the tests / lint

```powershell
python -m pytest -q
```

The Gemini API is **fully mocked** — tests make no real (paid) API calls.
Coverage: SRT/VTT timestamp conversion, SRT/VTT generation, segment ordering,
overlap/invalid-timestamp detection, JSON transcript validation, repair logic,
filename sanitization + traversal, FFmpeg command construction (audio
preservation, H.264, argv-not-shell), and the Gemini flow (happy path, malformed
JSON, quota error).

---

## 13. Deployment

**Vercel / Netlify / serverless will NOT work.** This is a Streamlit app: it
needs a *persistent* server process with WebSockets, plus the **FFmpeg system
binary** for rendering. Serverless Python runtimes expect a WSGI/ASGI
`app`/`handler` export (which Streamlit does not provide) and cannot run FFmpeg.
That is the cause of the `Found app.py but it does not export a top-level "app"`
error.

Use a host that runs a long-lived container/process:

### Streamlit Community Cloud (recommended, free)

1. Push this repo to GitHub (done: `grimmm07/vd-caption`).
2. Go to <https://share.streamlit.io> → **New app** → pick the repo/branch and
   set the main file to `app.py`.
3. FFmpeg is installed automatically from **`packages.txt`** (already in the
   repo).
4. Add your key under **Settings → Secrets** (TOML), e.g.:
   ```toml
   GEMINI_API_KEY = "your-key"
   GEMINI_MODEL = "gemini-2.5-flash"
   MAX_UPLOAD_MB = "200"
   ```
   The app bridges `st.secrets` into env vars automatically.

### Other options

- **Hugging Face Spaces** (Streamlit SDK) — also reads `packages.txt`; set the
  key as a Space secret.
- **Render / Railway / Fly.io / any Docker host** — run
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`; install
  FFmpeg in the image (`apt-get install -y ffmpeg`).

### ⚠️ Before you deploy publicly

- **Cost & key exposure:** anyone with the URL can upload videos and spend your
  paid Gemini quota under *your* key. Put it behind auth, or keep it internal.
- **Resources:** video re-encoding is CPU/memory heavy. Free tiers (~1 GB RAM,
  short timeouts) may be slow or OOM on longer/HD videos. The inline synced
  preview also embeds the video in the page (40 MB cap).
- For the actual **450-video batch job**, running **locally** (or as the batch
  CLI described below) is cheaper and more reliable than a public web app.

---

## 14. Future batch-processing approach (all ~450 videos)

The `GeminiTranscriber.transcribe()` method already takes a single path and
returns a validated `Transcript`, so a batch driver is a thin wrapper:

1. Enqueue all 450 files; process sequentially or with a **small** concurrency.
2. **Respect quota**: throttle to ≤ 5 req/min and ≤ 20 req/day (≈ 23 days
   minimum at that daily cap; request a quota increase to compress this).
3. **Idempotency / resume**: write `transcript.json` + `.srt`/`.vtt` per video
   into a per-video output folder; skip any video whose outputs already exist so
   an interrupted run can resume without duplicate paid calls.
4. **Backoff**: on `429`, pause and resume the next day rather than hammering.
5. Run FFmpeg burn-in as a separate, parallelizable local stage (no quota).
6. Emit a manifest/CSV (filename, language, duration, cost estimate, status) for
   tracking and cost reconciliation.

A CLI entry point (`batch.py`) reusing the existing services is the recommended
next step.
