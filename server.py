"""
server.py - FastAPI Backend for TraceCraft: From Pixels to Strokes.
Supports both /api/reconstruct and /reconstruct endpoints seamlessly.
"""

from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from engine import TraceCraftEngine

app = FastAPI(title="TraceCraft Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TraceCraftEngine()
BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def serve_index():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
@app.get("/health")
async def health():
    return {"status": "ok", "ai_model": getattr(engine, "model_name", "TraceCraft Engine")}


@app.post("/api/reconstruct")
@app.post("/reconstruct")
@app.post("/trace")
async def reconstruct_endpoint(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    max_shapes: int = Form(650),
    budget: int = Form(None),
    detail_mode: str = Form("balanced"),
    clean_shapes: int = Form(95),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="No image file uploaded.")

    shapes_target = budget if budget is not None else max_shapes
    shapes_target = max(20, min(3000, int(shapes_target)))

    img_bytes = await upload.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = engine.reconstruct(
            image_input=img_bytes,
            max_shapes=shapes_target,
            detail_mode=detail_mode,
            clean_budget=max(10, min(300, int(clean_shapes))),
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
