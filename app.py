from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import subprocess
import sys
from typing import Optional, List

app = FastAPI()

# --- Paths ---
OUTPUT_DIR = "/tmp/hd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ENV ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
RPM = os.getenv("RPM", "40")
BATCH_SIZE = os.getenv("BATCH_SIZE", "50")
CLASSIFIER_SCRIPT = os.getenv("CLASSIFIER_SCRIPT", "").strip()
DEFAULT_SUBJECT = os.getenv("SUBJECT_COL", "Problem")
DEFAULT_DESC = os.getenv("DESC_COL", "Popis problemu")
REQUIRE_API_KEY = os.getenv("X_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FORCE_OFFLINE = os.getenv("OFFLINE_ONLY", "0") in ("1", "true", "yes")


# --- Utility: find classifier script ---
def resolve_classifier_script() -> str:
    candidates = [
        "/app/hd_classify6.py",
        "/app/hd_classify6 (1).py",
        "hd_classify6.py",
        "hd_classify6 (1).py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return ""


@app.get("/")
def root():
    return {"status": "OK", "message": "AI-HD API is running"}


# --- Background job runner ---
def run_job(cmd: list, log_path: str, rc_path: str):
    classifier_path = cmd[1]
    workdir = os.path.dirname(classifier_path)

    with open(log_path, "a", buffering=1) as lf:
        lf.write("== PROCESS START ==\n")
        lf.write(f"CMD: {' '.join(cmd)}\n")
        lf.write(f"CWD: {workdir}\n\n")

    try:
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=lf,
            text=True,
            cwd=workdir,     # DŮLEŽITÉ
        )
        rc = proc.returncode
    except Exception as e:
        lf.write(f"[ERROR] {e}\n")
        rc = 1

    with open(rc_path, "w") as f:
        f.write(str(rc))


# --- POST /generate ---
@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    story: int = Form(0),
    subject_col: str = Form(DEFAULT_SUBJECT),
    desc_col: str = Form(DEFAULT_DESC),
    x_api_key_header1: Optional[str] = Header(None, alias="X-API-KEY"),
    x_api_key_header2: Optional[str] = Header(None, alias="X-API_KEY"),
):
    expected_key = REQUIRE_API_KEY
    provided = x_api_key_header1 or x_api_key_header2

    if expected_key:
        if not provided or provided != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    # save input
    body = await file.read()
    with open(input_path, "wb") as f:
        f.write(body)

    classifier = resolve_classifier_script()

    with open(log_path, "w") as lf:
        lf.write(f"== JOB {job_id} ==\n")
        lf.write(f"INPUT: {input_path}\n")
        lf.write(f"OUTPUT: {output_path}\n")
        lf.write(f"CLASSIFIER: {classifier or '<NOT FOUND>'}\n\n")

    if not classifier:
        with open(rc_path, "w") as rcf:
            rcf.write("1")
        return {"job_id": job_id}

    cmd = [
        sys.executable,
        os.path.abspath(classifier),
        "--input", input_path,
        "--output", output_path,
        "--provider", AI_PROVIDER,
        "--model", AI_MODEL,
        "--batch-size", str(BATCH_SIZE),
        "--rpm", str(RPM),
        "--subject-col", subject_col,
        "--desc-col", desc_col
    ]

    if story == 1:
        cmd.append("--story")

    if FORCE_OFFLINE or not OPENAI_API_KEY:
        cmd.append("--offline-only")

    # run in background
    
background_tasks.add_task(
    run_job,
    cmd,
    log_path,
    rc_path,
)


    return {"job_id": job_id}


# --- GET /status ---
@app.get("/status/{job_id}")
def status(job_id: str, tail: int = 100):
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    ready = os.path.exists(output_path)

    rc_val = None
    if os.path.exists(rc_path):
        with open(rc_path, "r") as rcf:
            t = rcf.read().strip()
            if t:
                try:
                    rc_val = int(t)
                except:
                    rc_val = None

    log_tail = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as lf:
            lines = lf.readlines()
            log_tail = "".join(lines[-tail:])

    return {
        "job_id": job_id,
        "ready": ready,
        "return_code": rc_val,
        "log_tail": log_tail
    }


# --- GET /download ---
@app.get("/download/{job_id}")
def download(job_id: str):
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    if not os.path.exists(output_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Result for job_id {job_id} not found."}
        )
    return FileResponse(output_path, filename="classified.xlsx")
