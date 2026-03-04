from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid, os, subprocess

app = FastAPI()
OUTPUT_DIR = "/tmp/hd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "OK", "message": "AI-HD API is running"}

@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    story: int = Form(0),
    subject_col: str = Form("Problém"),
    desc_col: str = Form("Popis problému"),
):
    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")

    # Uložení binárního obsahu
    body = await file.read()
    print("DEBUG file size:", len(body))
    with open(input_path, "wb") as f:
        f.write(body)

    # Příkaz pro klasifikaci
    cmd = [
        "python3", "hd_classify6.py",
        "--input", input_path,
        "--output", output_path,
        "--provider", "openai",
        "--model", "gpt-5-mini",
        "--batch-size", "50",
        "--rpm", "100",
        "--subject-col", subject_col,
        "--desc-col", desc_col,
    ]
    if story == 1:
        cmd.append("--story")

    # Fail-safe: pokud není OPENAI_API_KEY, udělej aspoň rules-only výstup
    if not os.getenv("OPENAI_API_KEY"):
        cmd.append("--offline-only")

    # Spusť na pozadí a loguj stdout/stderr do souboru
    with open(log_path, "w") as lf:
        lf.write(f"== JOB {job_id} START ==\n")
        lf.write(f"INPUT: {input_path}\nOUTPUT: {output_path}\n")
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
    # otevřít log v append režimu pro Popen
    lf = open(log_path, "a")
    subprocess.Popen(cmd, stdout=lf, stderr=lf, close_fds=True)

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str, tail: int = 80):
    """
    Stav jobu: existuje XLSX? vrať stav + posledních `tail` řádků logu (default 80).
    """
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")
    log_path = os.path.join(OUTPUT_DIR, f"{job_id}.log")

    ready = os.path.exists(output_path)
    log_tail = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            log_tail = "".join(lines[-tail:])
        except Exception as e:
            log_tail = f"<log read error: {e}>"

    return {
        "job_id": job_id,
        "ready": ready,
        "output_exists": ready,
        "log_tail": log_tail,
    }

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
