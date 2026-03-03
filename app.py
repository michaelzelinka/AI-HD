from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import subprocess

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
    # ➕ NOVÉ: volitelná mapování hlaviček
    subject_col: str = Form("Problém"),
    desc_col: str = Form("Popis problému"),
):
    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.xlsx")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cmd = [
        "python3", "hd_classify6.py",
        "--input", input_path,
        "--output", output_path,
        "--provider", "openai",
        "--model", "gpt-4o-mini",
        "--batch-size", "300",
        "--rpm", "120",
        # ➕ NOVÉ: předáme mapování dál do skriptu
        "--subject-col", subject_col,
        "--desc-col", desc_col,
    ]
    if story == 1:
        cmd.append("--story")

    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={"error": "Classification failed", "stderr": run.stderr, "stdout": run.stdout}
        )

    return {"job_id": job_id}

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
