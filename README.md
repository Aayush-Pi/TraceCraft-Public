# TraceCraft: From Pixels to Strokes

> Turn any raster image into clean, editable vector shapes, then watch it get painted back together stroke by stroke.

**Built for a hackathon** by **WeLarp**.

---

## Table of Contents

- [Demo video](#demo-video)
- [What is TraceCraft?](#what-is-tracecraft)
- [Why we built it](#why-we-built-it)
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [API reference](#api-reference)
- [Export formats](#export-formats)
- [Configuration and tuning](#configuration-and-tuning)
- [Metrics explained](#metrics-explained)
- [Troubleshooting](#troubleshooting)
- [Hackathon notes](#hackathon-notes)
- [Credits](#credits)

---

## Demo video


https://github.com/user-attachments/assets/b0012ee4-9963-4fe0-9c8d-add907f35bbc



**[Watch the demo (demo.mp4)](demo.mp4)** · about 1 min 40 s
The demo uses a sample image found online for demonstration purposes only. It is not owned by the project, and will be removed on request.

What the walkthrough shows, in order:

1. Launching the app and loading the built-in sample shapes
2. Uploading a portrait and tracing it at about 650 shapes
3. Raising the detail to about 2500 shapes and tracing again
4. Replaying the build on the timeline and changing playback speed
5. Inspecting the Error map, where the vector differs from the original
6. Switching to **Clean SVG** mode (95 non-overlapping shapes)
7. Browsing the auto-detected layers and the Run log of past traces



---

## What is TraceCraft?

TraceCraft is a web app that **vectorizes images**. You upload a photo or illustration, and the backend rebuilds it out of a limited number of vector shapes (polygons and Bézier curves). The result is resolution-independent, editable, and exportable to design tools such as Figma, Illustrator and Canva.

Unlike a one-click "image to SVG" converter, TraceCraft shows its work. You can replay the reconstruction step by step, inspect every shape, compare the vector against the original, and see exactly where the error is.

It runs **two engines side by side** on every upload:

| Mode | Goal | Shape count | Best for |
|---|---|---|---|
| **Painterly** | High visual fidelity, built from overlapping shapes like brush strokes | Hundreds to thousands | Art, portraits, photos |
| **Clean SVG** | Few, non-overlapping, gap-free shapes with smooth curves | Roughly 10 to 300 | Logos, icons, design handoff |

---

## Why we built it

Most image tracers give you a pile of thousands of unnamed paths with no way to understand or edit them. We wanted a tool that:

- Spends its shape budget **where the eye looks** (faces, eyes, edges) instead of spreading it evenly.
- Produces output a designer can actually **open and edit**.
- Makes the process **transparent and measurable**, with real quality metrics instead of "looks good to me".

---

## Features

**Tracing**
- Two vectorizers in one request: Painterly (high detail) and Clean (editable)
- Adjustable shape budget from 50 to 2500 with presets: Low, Medium, High, Ultra, Max
- Face and eye detection that pushes extra detail to the most important regions
- Automatic semantic layers: background, form, shadows, highlights, details, outlines

**Inspection**
- Split or single view with Original, Focus map and Error map tabs
- Comparison tools: curtain wipe, onion skin, error overlay, focus overlay
- Timeline player with Play, Replay and step forward or back
- Convergence chart (PSNR and MSE per stage)
- Shape inspector with live editing, reorder (forward or back) and delete
- Layer panel with hide and solo per layer
- Undo for edits
- "Compare low and high" button to see the effect of the budget side by side

**Workflow**
- Persistent history vault stored in the browser (IndexedDB)
- Run log table of past traces
- Keyboard shortcuts and accessible menus (arrow keys, Escape, Home and End)

**Export**
- Clean editable SVG, Painterly SVG, Layered SVG, Canva kit (zipped SVG per layer), PNG, WebM timelapse video, JSON quality report

---

## How it works

### Pipeline overview

```
Upload  ->  Resize  ->  Importance map  ->  Painterly engine  ->  Clean engine  ->  JSON response  ->  Browser UI
```

### 1. Preprocessing
The image is resized so its longest side is about 720 px. Small images are upscaled, but never more than 3x. Every tuning constant scales with resolution, so results stay consistent.

### 2. Importance (saliency) map
A weight map decides where detail matters. It fuses:
- Canny edges
- Sobel gradient magnitude
- Local contrast
- Haar-cascade face and eye priors (OpenCV)

If no face is found, the most salient 35% of pixels act as the region of interest instead.

### 3. Painterly engine (high fidelity)
A progressive, error-driven loop that only keeps a shape if it **reduces reconstruction error**.

| Stage | What it does | Shape type |
|---|---|---|
| 0 | Background fill from the median of the image corners | Rectangle |
| 1 | Large colour regions from a coarsely quantised, bilateral-filtered image | Polygons |
| 2 | Mid-size regions at two colour granularities | Polygons and quadratic Béziers |
| 3a | Fine detail inside the face and saliency region | Polygons and quadratic Béziers |
| 3b | Edge-following cubic ribbons plus greedy "brush dabs" placed where the weighted error is highest, oriented along the local edge tangent | Cubic Béziers, ellipses |

Stage 3b stops early when the error has converged, progress has stagnated, or the attempt limit is reached. The reason is reported back to the UI.

### 4. Clean engine (designer friendly)
A vtracer-style hierarchical pipeline:
1. Edge-preserving flattening (double bilateral filter)
2. Saliency-weighted k-means in Lab colour space (16 or 24 colours)
3. Gaussian label voting for smooth borders without speckle
4. Greedy **Ward merging** of adjacent regions down to the target count, with saliency weighting so eyes and lips survive
5. Each region traced once and **stacked** big to small, so there are no holes, gaps or duplicate overlaps
6. Outlines smoothed, simplified, and fitted with cubic Béziers while keeping sharp corners

### 5. Evaluation
Both outputs are scored against the working-resolution original with MSE, PSNR and SSIM. An error heatmap is generated so you can see where the vector falls short.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Image processing | OpenCV, NumPy, Pillow |
| Metrics | scikit-image (SSIM) |
| Frontend | Single-file HTML, CSS and vanilla JavaScript (no build step) |
| Browser storage | IndexedDB |

---

## Project structure

```
.
├── server.py      # FastAPI app: serves the UI and exposes the reconstruct API
├── engine.py      # TraceCraftEngine: saliency, painterly and clean vectorizers, metrics
├── index.html     # Complete frontend (UI, editor, exporters, history)
├── demo.mp4       # Demo walkthrough video
└── README.md
```

---

## Getting started

### Prerequisites
- Python 3.9 or newer
- pip

### 1. Clone the repository
```bash
git clone <https://github.com/Aayush-Pi/TraceCraft/>
cd <TraceCraft>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn python-multipart opencv-python numpy pillow scikit-image
```

Or save the following as `requirements.txt` and run `pip install -r requirements.txt`:

```text
fastapi
uvicorn
python-multipart
opencv-python
numpy
pillow
scikit-image
```

### 4. Run the server
```bash
python server.py
```

### 5. Open the app
Go to **http://127.0.0.1:8000** in your browser.

The first request can take a few seconds. Larger shape budgets take longer, since the painterly loop is CPU-bound.

---

## Using the app

1. Click **Get started** on the intro screen.
2. **Upload an image** from the left panel.
3. Pick a **Detail** level with the slider or a preset.
4. Click **Trace image**.
5. Switch between **Painterly** and **Clean SVG** to compare the two outputs.
6. Use **Split view** with the Original, Focus map and Error map tabs to judge quality.
7. Press **Play** on the timeline to replay how the image was built.
8. Open the **Inspector** to edit a shape, or the **Layers** tab to hide or solo layers.
9. Use **Export** to download your result.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `[` and `]` | Step the timeline back or forward by one shape |
| `Ctrl + Z` | Undo the last shape edit |
| `Delete` / `Backspace` | Delete the selected shape |
| `Esc` | Close the export menu or the intro |
| Arrow keys, `Home`, `End` | Navigate menus and tabs |

---

## API reference

The server accepts the same request on three routes: `/api/reconstruct`, `/reconstruct` and `/trace`.

### `POST /api/reconstruct`

Content type: `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` or `image` | file | required | The image to vectorize |
| `max_shapes` | int | `650` | Painterly shape budget (clamped to 20 to 3000) |
| `budget` | int | none | Alias for `max_shapes`; takes priority if set |
| `detail_mode` | string | `"balanced"` | Accepted for compatibility |
| `clean_shapes` | int | `95` | Target shape count for Clean mode (clamped to 10 to 300) |

**Example**
```bash
curl -X POST http://127.0.0.1:8000/api/reconstruct \
  -F "file=@photo.png" \
  -F "max_shapes=650" \
  -F "clean_shapes=95" \
  -o result.json
```

**Response (abridged)**
```json
{
  "ai_model": "Dual-Engine Neuro-Symbolic & Clean Graphic Vectorizer",
  "width": 720,
  "height": 540,
  "shapes": [ { "id": "s1_p_1", "stage": 1, "type": "polygon", "points": [[0,0]], "fill": "#a1b2c3", "opacity": 0.94, "layer": "form" } ],
  "clean_shapes": [ ],
  "metrics":       { "mse": 0, "psnr": 0, "ssim": 0, "runtime_sec": 0, "total_shapes": 0, "stop_reason": "budget_reached" },
  "clean_metrics": { "mse": 0, "psnr": 0, "ssim": 0, "total_shapes": 0 },
  "convergence_curve": [ { "step": 0, "mse": 0, "psnr": 0 } ],
  "layer_summary": { "background": 0, "form": 0, "shadows": 0, "highlights": 0, "details": 0, "outlines": 0 },
  "original_image_b64": "...",
  "reconstructed_image_b64": "...",
  "clean_reconstructed_b64": "...",
  "saliency_heatmap_b64": "...",
  "error_heatmap_b64": "..."
}
```

Images are returned as base64-encoded PNG strings.

### `GET /api/health` (also `/health`)
Returns `{"status": "ok", "ai_model": "..."}`.

### `GET /`
Serves `index.html`.

**Errors:** `400` for a missing or empty upload, `500` if reconstruction fails.

---

## Export formats

| Export | Output | Use it for |
|---|---|---|
| Clean editable SVG | `.svg` | Figma and Illustrator: fewer, non-overlapping shapes |
| Painterly SVG | `.svg` | High-fidelity artistic output |
| Layered SVG | `.svg` | Named groups for design suites |
| Canva kit | `.zip` | One SVG per layer |
| PNG | `.png` | Raster of the finished trace |
| Timelapse video | `.webm` | Sharing the build process |
| Quality report | `.json` | Metrics and convergence data |

---

## Configuration and tuning

Key constants live at the top of `TraceCraftEngine` in `engine.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MAX_DIM` | `720` | Longest side of the working and output resolution |
| `MAX_UPSCALE` | `3.0` | Maximum upscaling applied to small inputs |
| `BASE_DIM` | `360.0` | Resolution the tuning constants were designed for |

To change the host or port, edit the last line of `server.py`:

```python
uvicorn.run(app, host="127.0.0.1", port=8000)
```

> **Note:** CORS is currently open (`allow_origins=["*"]`) for easy hackathon development. Restrict it before any real deployment.

---

## Metrics explained

| Metric | Better when | Meaning |
|---|---|---|
| **MSE** | lower | Average squared pixel difference from the original |
| **PSNR** | higher | Signal quality in dB; roughly 30 dB or more looks close to the source |
| **SSIM** | closer to 1 | Structural similarity, which tracks human perception better than MSE |

Painterly mode is tuned to maximise these scores. Clean mode trades some of them for fewer shapes and easier editing.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| `Form data requires "python-multipart"` | Run `pip install python-multipart` |
| Blank page or 404 on `/` | Make sure `index.html` sits in the same folder as `server.py` |
| Tracing is slow | Lower the shape budget; Ultra and Max are CPU-heavy |
| No face-based detail boost | The Haar cascade did not detect a face; the saliency fallback is used instead |
| Port already in use | Change the port in `server.py` |

---

## Hackathon notes

**What we focused on**
- A working end-to-end product, from upload to exportable vector, with no build tooling
- Transparency: every shape, stage and error is visible and measurable
- Two modes so the same tool serves artists and designers

**Known limitations**
- CPU-only; high budgets can take a while
- Haar cascades only catch frontal faces
- Very noisy or photographic images need higher budgets to look good in Painterly mode
- No authentication or rate limiting

**Ideas for next steps**
- GPU or multi-process acceleration for the painterly loop
- A learned saliency model in place of Haar cascades
- Gradient fills and stroke-based output
- Batch processing and a public hosted demo

---

## Credits

Made by **WeLarp** for a hackathon.

Built with FastAPI, OpenCV, NumPy, Pillow and scikit-image.
