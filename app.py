Python 3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import os
... import uuid
... import requests
... import subprocess
... from fastapi import FastAPI, HTTPException, Query
... from fastapi.responses import FileResponse
... 
... app = FastAPI(title="AI Report Generator")
... 
... API_KEY = os.getenv("API_KEY", "CHANGE_ME")  # nastavíš v Renderu
... 
... @app.get("/generate")
... async def generate(
...     key: str = Query(...),
...     src: str = Query(...),
...     story: bool = Query(True)
... ):
...     # --- security ---
...     if key != API_KEY:
...         raise HTTPException(status_code=401, detail="Unauthorized")
... 
...     # --- download input ---
...     in_file = f"/tmp/{uuid.uuid4()}.xlsx"
...     out_file = f"/tmp/{uuid.uuid4()}.xlsx"
... 
...     try:
...         r = requests.get(src, timeout=120)
...         r.raise_for_status()
...     except Exception as e:
...         raise HTTPException(status_code=400, detail=f"Download error: {e}")
... 
...     with open(in_file, "wb") as f:
...         f.write(r.content)
... 
...     # --- build command ---
...     cmd = [
...         "python3", "hd_classify6.py",
        "--input", in_file,
        "--output", out_file,
        "--provider", os.getenv("AI_PROVIDER", "openai"),
        "--model", os.getenv("AI_MODEL", "gpt-4o-mini"),
        "--subject-col", os.getenv("SUBJECT_COL", "Problém"),
        "--desc-col", os.getenv("DESC_COL", "Popis problému"),
        "--rpm", os.getenv("RPM", "40")
    ]

    if story:
        cmd.append("--story")

    # --- execute ---
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"stdout": proc.stdout, "stderr": proc.stderr}
        )

    # --- return file ---
    return FileResponse(
        out_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="classified.xlsx"
