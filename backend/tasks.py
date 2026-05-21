import os
import threading
import time
from pathlib import Path

from backend.models import upscale, upscale_gif

tasks: dict[str, dict] = {}

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_upscale(task_id: str, input_path: str, output_path: str, scale: int) -> None:
    tasks[task_id]["status"] = "processing"
    try:
        def on_progress(current, total):
            tasks[task_id]["progress"] = {"current": current, "total": total}

        if input_path.lower().endswith(".gif"):
            upscale_gif(input_path, output_path, scale, progress_callback=on_progress)
        else:
            upscale(input_path, output_path, scale, progress_callback=on_progress)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["output_path"] = output_path
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def create_task(task_id: str, input_path: str, filename: str, scale: int) -> None:
    ext = Path(filename).suffix
    output_path = str(OUTPUTS_DIR / f"{task_id}{ext}")
    tasks[task_id] = {
        "status": "pending",
        "filename": filename,
        "scale": scale,
        "output_path": output_path,
        "created_at": time.time(),
    }
    t = threading.Thread(target=run_upscale, args=(task_id, input_path, output_path, scale))
    t.start()


def cleanup_old_outputs(max_age_seconds: int = 3600) -> None:
    now = time.time()
    to_remove = []
    for tid, info in tasks.items():
        if now - info.get("created_at", now) > max_age_seconds:
            if info.get("output_path") and os.path.exists(info["output_path"]):
                os.remove(info["output_path"])
            to_remove.append(tid)
    for tid in to_remove:
        del tasks[tid]
