import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.tasks import OUTPUTS_DIR, UPLOADS_DIR, cleanup_old_outputs, create_task, tasks

app = FastAPI()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), scale: int = Form(4)):
    if scale not in (1, 2, 4):
        raise HTTPException(400, "scale must be 1, 2 or 4")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File size exceeds 10MB limit")

    task_id = str(uuid.uuid4())
    input_path = str(UPLOADS_DIR / f"{task_id}{ext}")
    Path(input_path).write_bytes(content)

    create_task(task_id, input_path, file.filename or f"image{ext}", scale)
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    resp = {"task_id": task_id, "status": task["status"]}
    if task["status"] == "failed":
        resp["error"] = task.get("error", "Unknown error")
    if "progress" in task:
        resp["progress"] = task["progress"]
    return resp


@app.get("/api/download/{task_id}")
async def download(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed yet")

    output_path = task["output_path"]
    if not Path(output_path).exists():
        raise HTTPException(404, "Output file not found")

    ext = Path(output_path).suffix.lower()
    media_type = MIME_MAP.get(ext, "application/octet-stream")
    return FileResponse(output_path, media_type=media_type, filename=task["filename"])


@app.on_event("startup")
async def startup():
    UPLOADS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    cleanup_old_outputs()


# 靜態檔案掛載（前端）放在最後，避免覆蓋 API 路由
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
