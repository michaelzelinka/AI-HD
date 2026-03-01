# app.py
import os
import uuid
import subprocess
import requests
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form,
    Header, Query, BackgroundTasks
)
from fastapi.responses import FileResponse

app = FastAPI(title="AI Report Generator (FastAPI)")

# --- ENV config ---
API_KEY = os.getenv("API_KEY", "CHANGE_ME")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # nebo "azure"
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
SUBJECT_COL = os.getenv("SUBJECT_COL", "Problém")
DESC_COL = os.getenv("DESC_COL", "Popis problému")
RPM = os.getenv("RPM", "40")
DEFAULT_SRC = os.getenv("DEFAULT_SRC")            # volitelné: fallback URL pro GET /generate
CLASSIFIER_SCRIPT = os.getenv("CLASSIFIER_SCRIPT")  # např. "hd_classify6.py" nebo "classify6.py"


# ---------- Helpers ----------
def pick_classifier_script() -> str:
    """Vybere název skriptu klasifikátoru."""
    if CLASSIFIER_SCRIPT and Path(CLASSIFIER_SCRIPT).exists():
        return CLASSIFIER_SCRIPT
    if Path("hd_classify6.py").exists():
        return "hd_classify6.py"
    if Path("classify6.py").exists():
        return "classify6.py"
    raise HTTPException(status_code=500, detail="Classifier script not found (hd_classify6.py / classify6.py).")


def build_cmd(in_file: str, out_file: str, story: bool, model: Optional[str], rpm: Optional[str]) -> list:
    """Sestaví příkaz pro spuštění klasifikátoru."""
    script = pick_classifier_script()
    cmd = [
        "python3", script,
        "--input", in_file,
        "--output", out_file,
        "--provider", AI_PROVIDER,
        "--model", model or AI_MODEL,
        "--subject-col", SUBJECT_COL,
        "--desc-col", DESC_COL,
        "--rpm", rpm or RPM
    ]
    if story:
        cmd.append("--story")
    return cmd


def ensure_xlsx(filename: str):
    """Povolit jen .xlsx (openpyxl neumí .xls)."""
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Očekáván Excel .xlsx.")


def download_to_tmp(src_url: str) -> str:
    """Stáhne soubor z URL do /tmp a vrátí cestu k dočasnému XLSX."""
    in_file = f"/tmp/{uuid.uuid4()}.xlsx"
    try:
        with requests.get(src_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(in_file, "wb") as f:
                for chunk in r.iter_content(1024 * 512):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download error: {e}")
    return in_file


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/generate")
def generate_get(
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=""),
    key: Optional[str] = Query(default=None),
    src: Optional[str] = Query(default=None, description="URL na vstupní .xlsx"),
    story: bool = Query(default=True),
    model: Optional[str] = Query(default=None),
    rpm: Optional[str] = Query(default=None)
):
    """
    GET /generate?src=...&story=1
    - stáhne XLSX z URL (src nebo DEFAULT_SRC),
    - spustí klasifikátor,
    - vrátí hotový XLSX.
    """
    # --- Auth ---
    provided_key = x_api_key or (key or "")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    real_src = src or DEFAULT_SRC
    if not real_src:
        raise HTTPException(status_code=400, detail="Chybí 'src' a není nastaven DEFAULT_SRC.")

    # --- stáhnout do /tmp ---
    in_file = download_to_tmp(real_src)
    out_file = f"/tmp/{uuid.uuid4()}.xlsx"

    # --- spustit klasifikátor ---
    cmd = build_cmd(in_file, out_file, story=story, model=model, rpm=rpm)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        # úklid
        background_tasks.add_task(Path(in_file).unlink, missing_ok=True)
        background_tasks.add_task(Path(out_file).unlink, missing_ok=True)
        raise HTTPException(status_code=500, detail={"stdout": proc.stdout, "stderr": proc.stderr})

    # --- po odeslání odpovědi ukliď ---
    background_tasks.add_task(Path(in_file).unlink, missing_ok=True)
    background_tasks.add_task(Path(out_file).unlink, missing_ok=True)

    return FileResponse(
        out_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx"
    )


@app.post("/generate-upload")
async def generate_upload(
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=""),
    key: Optional[str] = Form(default=None),
    story: bool = Form(True),
    file: UploadFile = File(...)
):
    """
    POST /generate-upload (multipart/form-data)
      fields:
        - file: .xlsx (binary)
        - story: 1/0 (bool)
        - key: (volitelné – fallback; preferujeme X-API-Key v hlavičce)
    """
    # --- Auth ---
    provided_key = x_api_key or (key or "")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    ensure_xlsx(file.filename or "")

    # --- uložení nahraného souboru ---
    in_file = f"/tmp/{uuid.uuid4()}.xlsx"
    out_file = f"/tmp/{uuid.uuid4()}.xlsx"
    try:
        data = await file.read()
        with open(in_file, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {e}")

    # --- spustit klasifikátor ---
    cmd = build_cmd(in_file, out_file, story=story, model=None, rpm=None)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        # úklid
        background_tasks.add_task(Path(in_file).unlink, missing_ok=True)
        background_tasks.add_task(Path(out_file).unlink, missing_ok=True)
        raise HTTPException(status_code=500, detail={"stdout": proc.stdout, "stderr": proc.stderr})

    # --- po odeslání odpovědi ukliď ---
    background_tasks.add_task(Path(in_file).unlink, missing_ok=True)
    background_tasks.add_task(Path(out_file).unlink, missing_ok=True)

    return FileResponse(
        out_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx"
    )
