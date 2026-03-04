from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import subprocess
from typing import Optional

app = FastAPI()

# --- Cesty a výstupní adresář ---
OUTPUT_DIR = "/tmp/hd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Konfigurace z ENV (s rozumnými defaulty) ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
AI_MODEL = os.getenv("AI_MODEL", "gpt-5-mini")
RPM = os.getenv("RPM", "100")
BATCH_SIZE = os.getenv("BATCH_SIZE", "50")
CLASSIFIER_SCRIPT = os.getenv("CLASSIFIER_SCRIPT", "hd_classify6.py")
DEFAULT_SUBJECT_COL = os.getenv("SUBJECT_COL", "Problém")
DEFAULT_DESC_COL = os.getenv("DESC_COL", "Popis problému")
REQUIRE_API_KEY = os.getenv("X-API_KEY")  # je-li nastaveno, bude vyžadováno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # pokud chybí, spustíme --offline-only

@app.get("/")
def root():
    return {"status": "OK", "message": "AI-HD API is running"}

# ---------- interní helper: běh klasifikace + log ----------
def _run_classification(cmd: list, log_path: str, rc_path: str):
    """
    Spustí klasifikační skript a přesměruje stdout/stderr do logu.
    Uloží návratový kód do rc souboru.
    """
    # otevřeme log v append režimu (text)
    with open(log_path, "a", buffering=1) as lf:
        lf.write("== PROCESS START ==\n")
        lf.write(f"CMD: {' '.join(cmd)}\n")
        try:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)
            rc = proc.returncode
        except Exception as e:
            lf.write(f"\n[ERROR] Subprocess failed to start/run: {e}\n")
            rc = 1
        lf.write(f"\n== PROCESS END (rc={rc}) ==\n")
    # zapiš návratový kód do .rc
    try:
        with open(rc_path, "w") as rcf:
            rcf.write(str(rc))
    except Exception:
        pass

# ---------- POST /generate ----------
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    story: int = Form(0),
    subject_col: str = Form(DEFAULT_SUBJECT_COL),
    desc_col: str = Form(DEFAULT_DESC_COL),
    x_api_key_header1: Optional[str] = Header(None, alias="X-API-KEY"),
    x_api_key_header2: Optional[str] = Header(None, alias="X-API_KEY"),
):
    # (volitelné) API klíč z env → vyžaduj shodu
    expected_key = REQUIRE_API_KEY
    provided_key = x_api_key_header1 or x_api_key_header2
    if expected_key:
        if not provided_key or provided_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-API-KEY")

    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    # ulož příchozí binár
    body = await file.read()
    print("DEBUG file size:", len(body))
    with open(input_path, "wb") as f:
        f.write(body)

    # slož příkaz pro klasifikátor
    cmd = [
        "python3", CLASSIFIER_SCRIPT,
        "--input", input_path,
        "--output", output_path,
        "--provider", AI_PROVIDER,
        "--model", AI_MODEL,
        "--batch-size", str(BATCH_SIZE),
        "--rpm", str(RPM),
        "--subject-col", subject_col,
        "--desc-col", desc_col,
    ]
    if int(story) == 1:
        cmd.append("--story")

    # Pokud chybí OPENAI_API_KEY → aspoň rules-only (offline)
    if not OPENAI_API_KEY:
        cmd.append("--offline-only")

    # zapiš head logu a plánuj background úkol
    with open(log_path, "w") as lf:
        lf.write(f"== JOB {job_id} ==\n")
        lf.write(f"INPUT: {input_path}\nOUTPUT: {output_path}\n")
        lf.write(f"ENV: AI_PROVIDER={AI_PROVIDER} AI_MODEL={AI_MODEL} RPM={RPM} BATCH_SIZE={BATCH_SIZE}\n")
        lf.write(f"SUBJECT_COL='{subject_col}' DESC_COL='{desc_col}' STORY={story}\n\n")

    # background úkol poběží v rámci FastAPI workeru (Render to nechá doběhnout)
    background_tasks.add_task(_run_classification, cmd, log_path, rc_path)

    # okamžitě vrať job_id
    return {"job_id": job_id}

# ---------- GET /status/{job_id} ----------
@app.get("/status/{job_id}")
def status(job_id: str, tail: int = 120):
    """
    Stav jobu: existuje XLSX? jaký je návratový kód? vrať tail logu.
    """
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    ready = os.path.exists(output_path)
    rc_val = None
    if os.path.exists(rc_path):
        try:
            with open(rc_path, "r") as rcf:
                rc_text = rcf.read().strip()
                rc_val = int(rc_text) if rc_text else None
        except Exception:
            rc_val = None

    log_tail = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as lf:
                lines = lf.readlines()
            log_tail = "".join(lines[-max(1, int(tail)):])
        except Exception as e:
            log_tail = f"<log read error: {e}>"

    return {
        "job_id": job_id,
        "ready": ready,
        "output_exists": ready,
        "return_code": rc_val,
        "log_tail": log_tail,
    }

# ---------- GET /download/{job_id} ----------
@app.get("/download/{job_id}")
def download(job_id: str):
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    if not os.path.exists(output_path):
        return JSONResponse(status_code=404, content={"error": f"Result for job_id {job_id} not found."})
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx"
    )
