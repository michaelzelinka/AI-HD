# app.py
# FastAPI backend - AI Helpdesk klasifikace
#
# Endpoints:
#   SYNCHRONNÍ (pro menší dávky, <~120 s):
#     GET  /                  -> {"status":"ok","service":"ai-hd"}
#     HEAD /                  -> 200 (umlčení 405 při Render HEAD probe)
#     GET  /healthz           -> {"status":"ok"} (wake-up)
#     POST /generate-upload   -> přijme CSV/XLSX (multipart), vrátí XLSX
#     GET  /generate          -> stáhne vstup ze vzdálené URL (CSV/XLSX), vrátí XLSX
#
#   ASYNCHRONNÍ (pro dlouhé běhy přes proxy limity):
#     POST /jobs              -> vytvoří úlohu, vrátí 202 + job_id + status/result URL
#     GET  /jobs/{job_id}     -> vrátí stav (queued|running|done|failed) + diagnostiku
#     GET  /jobs/{job_id}/result -> stáhne hotové XLSX (409, pokud není hotovo)
#
# ENV proměnné (minimálně):
#   API_KEY nebo RENDER_SECRET  - očekáváno v hlavičce X-API-Key
#   OPENAI_API_KEY              - pro LLM volání ve skriptu
#   AI_PROVIDER (default "openai"), AI_MODEL (default "gpt-4o-mini"), RPM (default "40")
#   SUBJECT_COL (default "Problém"), DESC_COL (default "Popis problému")
#   CLASSIFIER_SCRIPT (default "hd_classify6.py")
#   JOB_TIMEOUT_SEC (default "900") - timeout jednoho jobu v sekundách
#
# Na Renderu spouštěj uvicorn shell formou:
#   CMD uvicorn app:app --host 0.0.0.0 --port $PORT

from __future__ import annotations

import os
import sys
import uuid
import shutil
import pathlib
import subprocess
import asyncio
import threading
from typing import Optional, Tuple, List
from enum import Enum
from datetime import datetime

from fastapi import (
    FastAPI, File, UploadFile, Form, Header, HTTPException, BackgroundTasks, Response, Query
)
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="AI Helpdesk API", version="1.1.0")

ALLOWED_EXT = {".csv", ".xlsx"}

# -------------------------
# Helpers (obecné)
# -------------------------

def temp_path(suffix: str) -> str:
    return f"/tmp/{uuid.uuid4().hex}{suffix}"

def normalize_bool(v: Optional[str | int | bool]) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def ensure_xlsx_or_csv(filename: str) -> None:
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        allowed = ", ".join(sorted(ALLOWED_EXT))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {allowed}")

def infer_ext_from_headers(url: str, content_type: Optional[str]) -> str:
    path = pathlib.PurePosixPath(url.split("?", 1)[0])
    ext = path.suffix.lower()
    if ext in ALLOWED_EXT:
        return ext
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in {"text/csv", "application/csv"}:
            return ".csv"
        if ct in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }:
            return ".xlsx"
    return ""

def get_api_key_from_request(x_api_key: Optional[str]) -> str:
    env_key = os.getenv("API_KEY") or os.getenv("RENDER_SECRET")
    if not env_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY/RENDER_SECRET is not set")
    if (x_api_key or "") != env_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return env_key

def get_default_cols() -> Tuple[str, str]:
    subj = os.getenv("SUBJECT_COL", "Problém")
    desc = os.getenv("DESC_COL", "Popis problému")
    return subj, desc

def build_classifier_cmd(
    in_path: str,
    out_path: str,
    subject_col: str,
    desc_col: str,
    story: bool
) -> List[str]:
    script = os.getenv("CLASSIFIER_SCRIPT", "hd_classify6.py")
    rpm = os.getenv("RPM", "40")
    ai_provider = os.getenv("AI_PROVIDER", "openai")
    ai_model = os.getenv("AI_MODEL", "gpt-4o-mini")

    cmd = [
        sys.executable, script,
        "--input", in_path,
        "--output", out_path,
        "--subject-col", subject_col,   # POZOR: spojovníky dle argparse
        "--desc-col", desc_col,
        "--rpm", rpm,
        "--provider", ai_provider,
        "--model", ai_model,
    ]
    if story:
        cmd.append("--story")
    return cmd

def require_openai_key_present() -> None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=500, detail="Server misconfiguration: OPENAI_API_KEY is not set")

# -------------------------
# Validace výsledného XLSX + diagnostika
# -------------------------

def is_valid_xlsx(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 100:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"\x50\x4B\x03\x04"  # "PK.."
    except Exception:
        return False

def read_head_tail(path: str, n: int = 256) -> dict:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(n)
            f.seek(max(0, size - n))
            tail = f.read(n)
        import binascii
        return {
            "size": size,
            "head_hex": binascii.hexlify(head).decode("ascii"),
            "tail_hex": binascii.hexlify(tail).decode("ascii"),
        }
    except Exception as e:
        return {"error": str(e)}

def preflight_diag(script_path: str) -> dict:
    import importlib.util, platform
    def module_present(name: str) -> bool:
        return importlib.util.find_spec(name) is not None
    return {
        "python": platform.python_version(),
        "script_exists": os.path.exists(script_path),
        "script": script_path,
        "modules": {
            "pandas": module_present("pandas"),
            "openpyxl": module_present("openpyxl"),
            "xlsxwriter": module_present("xlsxwriter"),
            "numpy": module_present("numpy"),
            "openai": module_present("openai"),
        }
    }

# -------------------------
# Spouštění skriptu (s timeoutem) + úklid
# -------------------------

def run_cmd_capture(cmd: List[str], extra_env: Optional[dict] = None, timeout: int | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        out, err = proc.communicate()
        return 124, out, (err or "") + f"\n[timeout] process killed after {timeout or 0}s"

def bg_cleanup(paths: List[str]) -> None:
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass

# -------------------------
# Základní routes (sync)
# -------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "ai-hd"}

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/generate-upload")
async def generate_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    story: Optional[int | str | bool] = Form(default=0),
    subject_col: Optional[str] = Form(default=None),
    desc_col: Optional[str] = Form(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    get_api_key_from_request(x_api_key)
    require_openai_key_present()

    ensure_xlsx_or_csv(file.filename or "")

    in_suffix = pathlib.Path(file.filename).suffix.lower()
    in_path = temp_path(in_suffix)
    out_path = temp_path(".xlsx")
    with open(in_path, "wb") as f:
        f.write(await file.read())

    default_subj, default_desc = get_default_cols()
    subj = (subject_col or default_subj).strip()
    desc = (desc_col or default_desc).strip()

    cmd = build_classifier_cmd(
        in_path=in_path,
        out_path=out_path,
        subject_col=subj,
        desc_col=desc,
        story=normalize_bool(story),
    )

    # Synchronous varianta – bez extra timeoutu (pozor na proxy limity pro velké dávky)
    rc, stdout, stderr = run_cmd_capture(cmd, extra_env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")})

    if rc != 0 or not is_valid_xlsx(out_path):
        diag = preflight_diag(os.path.join(os.getcwd(), os.getenv("CLASSIFIER_SCRIPT", "hd_classify6.py")))
        file_probe = read_head_tail(out_path) if os.path.exists(out_path) else {"exists": False}
        # úklid vstupu/výstupu
        bg_cleanup([in_path, out_path])
        return JSONResponse(
            status_code=500,
            content={
                "error": "classification_failed",
                "returncode": rc,
                "stdout": (stdout or "")[-8000:],
                "stderr": (stderr or "")[-8000:],
                "cmd": cmd,
                "diag": diag,
                "file_probe": file_probe,
            },
        )

    # úklid vstupu po odeslání; výstup smažeme také (po odeslání)
    background_tasks.add_task(bg_cleanup, [in_path, out_path])

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx",
        background=background_tasks,
    )

@app.get("/generate")
def generate_from_url(
    background_tasks: BackgroundTasks,
    url: str = Query(..., description="URL na CSV/XLSX soubor"),
    story: Optional[int | str | bool] = Query(default=0),
    subject_col: Optional[str] = Query(default=None),
    desc_col: Optional[str] = Query(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    import urllib.request

    get_api_key_from_request(x_api_key)
    require_openai_key_present()

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type")
            ext = infer_ext_from_headers(url, content_type)
            if ext not in ALLOWED_EXT:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot infer file type from URL/headers. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
                )
            in_path = temp_path(ext)
            out_path = temp_path(".xlsx")
            with open(in_path, "wb") as f:
                f.write(resp.read())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download input: {e}")

    default_subj, default_desc = get_default_cols()
    subj = (subject_col or default_subj).strip()
    desc = (desc_col or default_desc).strip()

    cmd = build_classifier_cmd(
        in_path=in_path,
        out_path=out_path,
        subject_col=subj,
        desc_col=desc,
        story=normalize_bool(story),
    )
    rc, stdout, stderr = run_cmd_capture(cmd, extra_env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")})

    if rc != 0 or not is_valid_xlsx(out_path):
        diag = preflight_diag(os.path.join(os.getcwd(), os.getenv("CLASSIFIER_SCRIPT", "hd_classify6.py")))
        file_probe = read_head_tail(out_path) if os.path.exists(out_path) else {"exists": False}
        bg_cleanup([in_path, out_path])
        return JSONResponse(
            status_code=500,
            content={
                "error": "classification_failed",
                "returncode": rc,
                "stdout": (stdout or "")[-8000:],
                "stderr": (stderr or "")[-8000:],
                "cmd": cmd,
                "diag": diag,
                "file_probe": file_probe,
            },
        )

    background_tasks.add_task(bg_cleanup, [in_path, out_path])

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx",
        background=background_tasks,
    )

# -------------------------
# ASYNCHRONNÍ JOBY (pro dlouhé běhy bez timeoutů)
# -------------------------

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", "900"))  # 15 min

def _set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(kwargs)

def _new_job_record(job_id: str, in_path: str, out_path: str, cmd: list[str]):
    now = datetime.utcnow().isoformat()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": JobStatus.queued,
            "created": now,
            "started": None,
            "finished": None,
            "in_path": in_path,
            "out_path": out_path,
            "cmd": cmd,
            "rc": None,
            "stdout": "",
            "stderr": "",
            "error": None
        }

async def _run_job_async(job_id: str):
    _set_job(job_id, status=JobStatus.running, started=datetime.utcnow().isoformat())
    rec = JOBS[job_id]
    in_path, out_path, cmd = rec["in_path"], rec["out_path"], rec["cmd"]

    rc, stdout, stderr = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: run_cmd_capture(
            cmd,
            extra_env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")},
            timeout=JOB_TIMEOUT_SEC
        )
    )
    _set_job(job_id, rc=rc, stdout=(stdout or "")[-8000:], stderr=(stderr or "")[-8000:])

    if rc == 0 and is_valid_xlsx(out_path):
        _set_job(job_id, status=JobStatus.done, finished=datetime.utcnow().isoformat())
    else:
        file_probe = read_head_tail(out_path) if os.path.exists(out_path) else {"exists": False}
        _set_job(job_id, status=JobStatus.failed, finished=datetime.utcnow().isoformat(),
                 error="classification_failed", file_probe=file_probe)

    # vstup smažeme hned; výstup ponecháme do stažení
    try:
        if os.path.exists(in_path):
            os.remove(in_path)
    except Exception:
        pass

@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    story: Optional[int | str | bool] = Form(default=0),
    subject_col: Optional[str] = Form(default=None),
    desc_col: Optional[str] = Form(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    get_api_key_from_request(x_api_key)
    require_openai_key_present()
    ensure_xlsx_or_csv(file.filename or "")

    in_suffix = pathlib.Path(file.filename).suffix.lower()
    in_path = temp_path(in_suffix)
    out_path = temp_path(".xlsx")
    with open(in_path, "wb") as f:
        f.write(await file.read())

    default_subj, default_desc = get_default_cols()
    subj = (subject_col or default_subj).strip()
    desc = (desc_col or default_desc).strip()

    cmd = build_classifier_cmd(
        in_path=in_path,
        out_path=out_path,
        subject_col=subj,
        desc_col=desc,
        story=normalize_bool(story)
    )

    job_id = uuid.uuid4().hex
    _new_job_record(job_id, in_path=in_path, out_path=out_path, cmd=cmd)

    # Spustíme na pozadí – HTTP se vrátí hned
    asyncio.create_task(_run_job_async(job_id))

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status_url": f"/jobs/{job_id}",
            "result_url": f"/jobs/{job_id}/result"
        }
    )

@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    get_api_key_from_request(x_api_key)
    rec = JOBS.get(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job_not_found")
    # Zkrátíme průběžné logy, ať nevracíme megabajty
    resp = dict(rec)
    if resp.get("status") in (JobStatus.queued, JobStatus.running):
        resp["stdout"] = (resp.get("stdout", "") or "")[-200:]
        resp["stderr"] = (resp.get("stderr", "") or "")[-200:]
    return resp

@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    get_api_key_from_request(x_api_key)
    rec = JOBS.get(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job_not_found")
    if rec["status"] != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"job_not_ready ({rec['status']})")
    out_path = rec["out_path"]
    if not is_valid_xlsx(out_path):
        raise HTTPException(status_code=500, detail="invalid_xlsx_on_disk")

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx",
    )

# -------------------------
# Lokální spuštění
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
