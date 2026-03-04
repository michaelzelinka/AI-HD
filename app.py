from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
import uuid, os, subprocess, sys

app = FastAPI()

# --- Cesty ---
OUTPUT_DIR = "/tmp/hd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ENV konfigurace (s defaulty) ---
AI_PROVIDER      = os.getenv("AI_PROVIDER", "openai")
AI_MODEL         = os.getenv("AI_MODEL", "gpt-5-mini")
RPM              = os.getenv("RPM", "100")
BATCH_SIZE       = os.getenv("BATCH_SIZE", "50")
CLASSIFIER_SCRIPT= os.getenv("CLASSIFIER_SCRIPT", "").strip()  # může být prázdné
DEFAULT_SUBJECT  = os.getenv("SUBJECT_COL", "Problem")
DEFAULT_DESC     = os.getenv("DESC_COL", "Popis problemu")
REQUIRE_API_KEY  = os.getenv("X-API_KEY")  # pokud je nastaveno, vyžadujeme hlavičku
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
FORCE_OFFLINE    = os.getenv("OFFLINE_ONLY", "0") in ("1", "true", "yes", "y")

# --- Pomůcka: najdi existující klasifikační skript ---
def resolve_classifier_script() -> str:
    """
    Vrátí existující cestu ke skriptu klasifikace:
    - přednost má CLASSIFIER_SCRIPT z ENV, pokud existuje
    - jinak zkusí běžné názvy v repu
    """
    candidates: List[str] = []
    if CLASSIFIER_SCRIPT:
        candidates.append(CLASSIFIER_SCRIPT)
    # běžné názvy v repu
    candidates += [
        "hd_classify6.py",
        "hd_classify6 (1).py",
        "./hd_classify6.py",
        "./hd_classify6 (1).py",
        "/app/hd_classify6.py",
        "/app/hd_classify6 (1).py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""  # nic jsme nenašli

@app.get("/")
def root():
    return {"status": "OK", "message": "AI-HD API is running"}

# --- Vykonavací funkce: spustí klasifikátor a vše zaloguje ---
def run_job(cmd: list, log_path: str, rc_path: str):
    """
    Spustí klasifikační proces synchronně (v background tasku), napíše stdout/stderr do logu
    a uloží návratový kód do .rc souboru.
    """
    with open(log_path, "a", buffering=1) as lf:
        lf.write("== PROCESS START ==\n")
        lf.write(f"PYTHON: {sys.executable}\n")
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        try:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)
            rc = proc.returncode
        except Exception as e:
            lf.write(f"\n[ERROR] Subprocess failed: {e}\n")
            rc = 1
        lf.write(f"\n== PROCESS END (rc={rc}) ==\n")
    try:
        with open(rc_path, "w") as rcf:
            rcf.write(str(rc))
    except Exception:
        pass

# --- POST /generate ---
@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    story: int = Form(0),
    subject_col: str = Form("Problem"),
    desc_col: str = Form("Popis problemu"),
    x_api_key_header1: Optional[str] = Header(None, alias="X-API-KEY"),
    x_api_key_header2: Optional[str] = Header(None, alias="X-API_KEY"),
):
):
    # (volitelné) API klíč
    expected_key = REQUIRE_API_KEY
    provided_key = x_api_key_header1 or x_api_key_header2
    if expected_key:
        if not provided_key or provided_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-API-KEY")

    job_id = str(uuid.uuid4())
    input_path  = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path    = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path     = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    # ulož příchozí soubor
    body = await file.read()
    print("DEBUG file size:", len(body))
    with open(input_path, "wb") as f:
        f.write(body)

    # najdi skript
    classifier = resolve_classifier_script()
    with open(log_path, "w") as lf:
        lf.write(f"== JOB {job_id} ==\n")
        lf.write(f"INPUT:  {input_path}\n")
        lf.write(f"OUTPUT: {output_path}\n")
        lf.write(f"ENV: AI_PROVIDER={AI_PROVIDER} AI_MODEL={AI_MODEL} RPM={RPM} BATCH_SIZE={BATCH_SIZE}\n")
        lf.write(f"SUBJECT_COL='{subject_col}' DESC_COL='{desc_col}' STORY={story}\n")
        lf.write(f"OPENAI_KEY_SET={'yes' if OPENAI_API_KEY else 'no'} FORCE_OFFLINE={FORCE_OFFLINE}\n")
        lf.write(f"RESOLVED_CLASSIFIER='{classifier or '<not-found>'}'\n\n")

    # pokud skript neexistuje, zapiš chybu rovnou sem (a nech /status ji vypsat)
    if not classifier:
        with open(log_path, "a") as lf:
            lf.write("[FATAL] Classifier script not found. Set CLASSIFIER_SCRIPT env or ensure 'hd_classify6.py' is present.\n")
        with open(rc_path, "w") as rcf:
            rcf.write("1")
        # nečekáme – job se považuje za chybový; /download vrátí 404, /status ukáže log
        return {"job_id": job_id}

    # postav příkaz
    cmd = [
        sys.executable, classifier,             # stejné py runtime jako FastAPI
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

    # fail-safe: když není OPENAI_API_KEY nebo je vynucen offline režim, přidej --offline-only
    if FORCE_OFFLINE or not OPENAI_API_KEY:
        cmd.append("--offline-only")

    # spustíme to jako background task (uvnitř workeru, Render to nezabije)
    background_tasks.add_task(run_job, cmd, log_path, rc_path)

    return {"job_id": job_id}

# --- GET /status/{job_id} ---
@app.get("/status/{job_id}")
def status(job_id: str, tail: int = 120):
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path    = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    rc_path     = os.path.join(OUTPUT_DIR, f"{job_id}.rc")

    ready = os.path.exists(output_path)
    rc_val = None
    if os.path.exists(rc_path):
        try:
            with open(rc_path, "r") as rcf:
                txt = rcf.read().strip()
                rc_val = int(txt) if txt else None
        except Exception:
            rc_val = None

    log_tail = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as lf:
                lines = lf.readlines()
            tail_n = max(1, int(tail))
            log_tail = "".join(lines[-tail_n:])
        except Exception as e:
            log_tail = f"<log read error: {e}>"

    return {"job_id": job_id, "ready": ready, "return_code": rc_val, "log_tail": log_tail}

# --- GET /download/{job_id} ---
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
