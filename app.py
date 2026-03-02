from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import subprocess
app = FastAPI()

@app.get("/")
def root():
   return {"status": "OK", "message": "AI-HD API running"}

@app.post("/generate-upload")
async def generate_upload(
   file: UploadFile = File(...),
   story: int = Form(0)
):
   """
   Přijme CSV/XLSX přes multipart/form-data,
   spustí hd_classify6.py a vrátí výstupní XLSX.
   """
   # 1) Uložení nahraného souboru
   input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
   output_path = input_path + "_classified.xlsx"
   with open(input_path, "wb") as f:
       f.write(await file.read())
   # 2) Příprava příkazu
   cmd = [
       "python3",
       "hd_classify6.py",
       "--input", input_path,
       "--output", output_path,
       "--provider", "openai",
       "--model", "gpt-4o-mini",
   ]
   if story == 1:
       cmd.append("--story")
   # 3) Spuštění klasifikace
   result = subprocess.run(cmd, capture_output=True, text=True)
   if result.returncode != 0:
       return JSONResponse(
           status_code=500,
           content={
               "error": "AI classification failed",
               "stderr": result.stderr,
               "stdout": result.stdout,
           },
       )
   # 4) Vrácení XLSX
   return FileResponse(
       output_path,
       media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
       filename="classified.xlsx"
   )
