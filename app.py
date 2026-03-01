# app.py
# FastAPI backend pro AI Helpdesk klasifikaci (Render-friendly)
# Endpoints:
#   GET  /           -> {"status":"ok","service":"ai-hd"}
#   HEAD /           -> 200 (umlčení 405 při Render HEAD probe)
#   GET  /healthz    -> {"status":"ok"} (wake-up)
#   POST /generate-upload -> přijme CSV/XLSX (multipart), vrátí XLSX
#   GET  /generate   -> stáhne vstup ze vzdálené URL (CSV/XLSX), vrátí XLSX
#
# ENV proměnné (minimálně):
#   API_KEY nebo RENDER_SECRET  - očekáváno v hlavičce X-API-Key
#   OPENAI_API_KEY              - pro LLM volání ve skriptu
#   AI_PROVIDER (default "openai"), AI_MODEL (default "gpt-4o-mini"), RPM (default "40")
#   SUBJECT_COL (default "Problém"), DESC_COL (default "Popis problému")
#   CLASSIFIER_SCRIPT (default "hd_classify6.py")
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
from typing import Optional, Tuple, List

from fastapi import (
    FastAPI, File, UploadFile, Form, Header, HTTPException, BackgroundTasks, Response, Query
)
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------
# Konfigurace aplikace
# ---------------------------------------------------------

app = FastAPI(title="AI Helpdesk API", version="1.0.0")

ALLOWED_EXT = {".csv", ".xlsx"}


# ---------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------

def temp_path(suffix: str) -> str:
    """Vytvoří jedinečnou cestu v /tmp."""
    return f"/tmp/{uuid.uuid4().hex}{suffix}"


def normalize_bool(v: Optional[str | int | bool]) -> bool:
    """Normalizace truthy hodnot z form-data/query (1, true, yes)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def ensure_xlsx_or_csv(filename: str) -> None:
    """Povolí jen .csv nebo .xlsx na základě přípony."""
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        allowed = ", ".join(sorted(ALLOWED_EXT))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {allowed}")


def infer_ext_from_headers(url: str, content_type: Optional[str]) -> str:
    """
    Zjistí příponu podle URL nebo Content-Type.
    Preferuje příponu z URL; když chybí, mapuje Content-Type na .csv/.xlsx.
    """
    path = pathlib.PurePosixPath(url.split("?", 1)[0])
    ext = path.suffix.lower()
    if ext in ALLOWED_EXT:
        return ext
    # Mapování podle MIME
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in {"text/csv", "application/csv"}:
            return ".csv"
        if ct in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }:
            return ".xlsx"
    # Fallback – raději zakážeme než tipovat špatně
    return ""


def get_api_key_from_request(x_api_key: Optional[str]) -> str:
    """
    Ověří X-API-Key proti API_KEY/RENDER_SECRET v ENV.
    Vrací platný klíč, jinak 401/500.
    """
    env_key = os.getenv("API_KEY") or os.getenv("RENDER_SECRET")
    if not env_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY/RENDER_SECRET is not set")
    if (x_api_key or "") != env_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return env_key


def get_default_cols() -> Tuple[str, str]:
    """Výchozí názvy sloupců ze smysluplných ENV fallbacků."""
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
    """
    Poskládá příkaz pro klasifikátor (samostatný proces kvůli izolaci a sběru stdout/stderr).
    """
    script = os.getenv("CLASSIFIER_SCRIPT", "hd_classify6.py")
    rpm = os.getenv("RPM", "40")
    ai_provider = os.getenv("AI_PROVIDER", "openai")
    ai_model = os.getenv("AI_MODEL", "gpt-4o-mini")

    cmd = [
        sys.executable, script,
        "--input", in_path,
        "--output", out_path,
        # POZOR: správné argumenty se SPOJOVNÍKEM, aby seděly s argparse skriptu
        "--subject-col", subject_col,
        "--desc-col", desc_col,
        "--rpm", rpm,
        "--provider", ai_provider,
        "--model", ai_model,
    ]
    if story:
        cmd.append("--story")
    return cmd


def run_cmd_capture(cmd: List[str], extra_env: Optional[dict] = None) -> tuple[int, str, str]:
    """Spustí příkaz a vrátí (returncode, stdout, stderr)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = proc.communicate()
    return proc.returncode, out, err


def bg_cleanup(paths: List[str]) -> None:
    """Smaže soubory/složky; chyby ignoruje."""
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass


def require_openai_key_present() -> None:
    """Včasná kontrola, aby uživatel dostal srozumitelnou chybu, když chybí LLM klíč."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: OPENAI_API_KEY is not set"
        )


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "ai-hd"}


@app.head("/")
def root_head():
    # Umlčí 405 "Method Not Allowed" na Renderu při HEAD probe.
    return Response(status_code=200)


@app.get("/healthz")
def healthz():
    # Jednoduchý wake-up endpoint (doporučeno volat před uploadem).
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
    # Auth + základní validace prostředí
    get_api_key_from_request(x_api_key)
    require_openai_key_present()

    # Validace přípony
    ensure_xlsx_or_csv(file.filename or "")

    # Uložení vstupu do /tmp
    in_suffix = pathlib.Path(file.filename).suffix.lower()
    in_path = temp_path(in_suffix)
    out_path = temp_path(".xlsx")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    # Sloupce (defaulty z ENV => přepíše form-data)
    default_subj, default_desc = get_default_cols()
    subj = (subject_col or default_subj).strip()
    desc = (desc_col or default_desc).strip()

    # Sestavení & spuštění skriptu
    cmd = build_classifier_cmd(
        in_path=in_path,
        out_path=out_path,
        subject_col=subj,
        desc_col=desc,
        story=normalize_bool(story),
    )

    rc, stdout, stderr = run_cmd_capture(cmd, extra_env={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    })

    if rc != 0 or not os.path.exists(out_path):
        # Vrátit detailní debug JSON (omezená délka logu kvůli velikosti odpovědi)
        return JSONResponse(
            status_code=500,
            content={
                "error": "classification_failed",
                "returncode": rc,
                "stdout": (stdout or "")[-8000:],
                "stderr": (stderr or "")[-8000:],
                "cmd": cmd,
            },
        )

    # Úklid dočasných dat po odeslání
    background_tasks.add_task(bg_cleanup, [in_path, out_path])

    # Stream XLSX zpět
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
    """
    Stáhne vzdálený CSV/XLSX a zpracuje ho. Nepotřebuje 'requests' – používá stdlib urllib.
    Vhodné pro jednoduché scénáře nebo rychlé testy.
    """
    import urllib.request

    # Auth + kontrola LLM klíče
    get_api_key_from_request(x_api_key)
    require_openai_key_present()

    # Stáhnout do /tmp, odhadnout příponu
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

    # Sloupce
    default_subj, default_desc = get_default_cols()
    subj = (subject_col or default_subj).strip()
    desc = (desc_col or default_desc).strip()

    # Spuštění
    cmd = build_classifier_cmd(
        in_path=in_path,
        out_path=out_path,
        subject_col=subj,
        desc_col=desc,
        story=normalize_bool(story),
    )
    rc, stdout, stderr = run_cmd_capture(cmd, extra_env={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    })

    if rc != 0 or not os.path.exists(out_path):
        # Úklid vstupu (výstup nejspíš nevznikl)
        bg_cleanup([in_path, out_path])
        return JSONResponse(
            status_code=500,
            content={
                "error": "classification_failed",
                "returncode": rc,
                "stdout": (stdout or "")[-8000:],
                "stderr": (stderr or "")[-8000:],
                "cmd": cmd,
            },
        )

    # Úklid po odeslání
    background_tasks.add_task(bg_cleanup, [in_path, out_path])

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx",
        background=background_tasks,
    )


# ---------------------------------------------------------
# Lokální spuštění (volitelné)
# ---------------------------------------------------------
if __name__ == "__main__":
    # pro lokální běh: python app.py
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
