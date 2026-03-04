# -*- coding: utf-8 -*-
"""
hd_classify6.py – Finální produkční verze (2026/03)

Funkce:
- Rule-first + Batch LLM klasifikace (minimalizace LLM volání)
- Storytelling (CZ)
- Robustní CSV loader (semicolon/diakritika/BOM/CRLF/multiline)
- Výstup pouze XLSX (Data, AI_Staging, volitelně Storytelling)
- Optimalizováno pro Render + Make (dvoufázové API v app.py)
- LLM: OpenAI chat/completions, model výchozí gpt-5-mini
- Žádný "temperature" v payloadu (gpt-5-mini nepodporuje), timeout 300 s

Příklad (lokálně):
python3 hd_classify6.py \
  --input /path/Incidenty.csv \
  --output /path/classified.xlsx \
  --provider openai \
  --model gpt-5-mini \
  --batch-size 120 \
  --rpm 120 \
  --subject-col "Problém" \
  --desc-col "Popis problému" \
  --story
"""

from __future__ import annotations
import argparse
import os
import re
import json
import time
import csv
import io
import unicodedata
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests


# ============================
# 0) Utility: text & safety
# ============================

def safe_text(x: str) -> str:
    """Očistí text: BOM, CRLF→LF, kontrolní znaky, zhuštění whitespace."""
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\ufeff", "")
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[\x00-\x1F\x7F]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_text(s: str) -> str:
    return safe_text(s)


def normalize_for_match(s: str) -> str:
    """Lower + odstranění diakritiky + kompaktní mezery (pro robustní porovnávání)."""
    s = normalize_text(s).lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


# ============================
# 1) Kategorie a rules-engine
# ============================

CATEGORIES = [
    "Instalace / aktualizace software",
    "Údržba / čištění disků a souborů",
    "Databáze / Oracle / MCC / Virtuan",
    "Interní aplikace (Unique, NemExpress, Argus…)",
    "Tiskárny a skenery",
    "Síť / Wi-Fi / VPN / GlobalProtect",
    "Office / Outlook / Teams",
    "Hardware (notebook, monitor, periferie)",
    "Přístupy / oprávnění / účty",
    "Cloud / OneDrive / SharePoint",
    "Mobil / iPhone / Android",
    "Ostatní",
]

KEYWORDS: Dict[str, List[str]] = {
    "Instalace / aktualizace software": ["instalace", "install", "setup", "update", "upgrade", "reinstall", "patch"],
    "Údržba / čištění disků a souborů": ["čištění", "cisteni", "úklid", "smazat", "mazání", "odstranit", "volne misto"],
    "Databáze / Oracle / MCC / Virtuan": ["oracle", "mcc", "virtuan", "sql", "database", "databáze", "db"],
    "Interní aplikace (Unique, NemExpress, Argus…)": ["unique", "nemexpress", "argus", "interni aplikace"],
    "Tiskárny a skenery": ["tisk", "tiskarna", "tiskárna", "print", "scan", "skener", "toner"],
    "Síť / Wi-Fi / VPN / GlobalProtect": ["wifi", "wi-fi", "vpn", "gp", "globalprotect", "síť", "network", "připojení"],
    "Office / Outlook / Teams": ["outlook", "office", "teams", "email", "mail", "kalendář", "calendar"],
    "Hardware (notebook, monitor, periferie)": ["notebook", "monitor", "klavesnice", "myš", "baterie", "charger"],
    "Přístupy / oprávnění / účty": ["přístup", "opravneni", "heslo", "password", "login", "permissions", "účty"],
    "Cloud / OneDrive / SharePoint": ["onedrive", "sharepoint", "cloud", "sdíl", "sdilena slozka"],
    "Mobil / iPhone / Android": ["iphone", "android", "mobil", "sim"],
    "Ostatní": []
}

ALIASES = [
    ("globalprotect", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("global connect", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("gp", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("nemexpress", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("unique", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("argus", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("oracle", "Databáze / Oracle / MCC / Virtuan"),
    ("mcc", "Databáze / Oracle / MCC / Virtuan"),
    ("nelze tisknout", "Tiskárny a skenery"),
]


def category_by_rules(text: str) -> Tuple[Optional[str], int, str]:
    """Jednoduchý rules engine (aliasy → keywords). Vrací (category, confidence, explanation)."""
    txt = normalize_for_match(text)

    # Alias match
    for phrase, cat in ALIASES:
        if normalize_for_match(phrase) in txt:
            return cat, 95, f"alias:{phrase}"

    # Keyword scoring
    best_cat, best_hits = None, 0
    for cat, words in KEYWORDS.items():
        hits = sum(1 for w in words if normalize_for_match(w) in txt)
        if hits > best_hits:
            best_hits, best_cat = hits, cat

    if best_hits >= 3:
        return best_cat, 90, "keywords-strong"
    if best_hits == 2:
        return best_cat, 75, "keywords-medium"
    if best_hits == 1:
        return best_cat, 55, "keywords-weak"
    return None, 0, "no-match"


def validate_category(cat: Optional[str]) -> str:
    if not cat:
        return "Ostatní"
    c_low = normalize_for_match(cat)
    for c in CATEGORIES:
        if normalize_for_match(c) == c_low:
            return c
    for c in CATEGORIES:
        if c_low in normalize_for_match(c):
            return c
    return "Ostatní"


# ============================
# 2) LLM klient (OpenAI)
# ============================

class LLMClient:
    def __init__(self, model: str, rpm: int):
        self.model = model
        self.rpm = max(1, int(rpm))
        self._last = 0.0
        self.min_interval = 60.0 / self.rpm

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY není nastaven.")
        self.url = "https://api.openai.com/v1/chat/completions"

    def _rate(self):
        now = time.time()
        wait = (self._last + self.min_interval) - now
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _post(self, payload: dict, timeout: int = 300) -> dict:
        self._rate()
        r = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=timeout
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        return r.json()

    # Řádková klasifikace (pro completeness; primárně používáme batch)
    def classify(self, text: str, strict: bool = False, lang: str = "cs") -> Dict:
        system = (
            f"Odpovídej {lang}. Jsi zkušený IT analytik. Urči jednu kategorii z předem daného seznamu."
            if not strict else
            f"Odpovídej {lang}. Jsi zkušený IT analytik. Vyber jednu nejbližší kategorii. 'Ostatní' jen v nouzi."
        )
        user_prompt = f"""
Seznam kategorií:
{json.dumps(CATEGORIES, ensure_ascii=False, indent=2)}

Text tiketu:
\"\"\"
{text}
\"\"\"

Vrať POUZE JSON:
{{"category":"...","confidence":0-100,"explanation":"..."}}
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = self._post(payload, timeout=300)
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return {"category": "Ostatní", "confidence": 0, "explanation": "no-json"}
        try:
            js = json.loads(m.group(0))
        except Exception as e:
            return {"category": "Ostatní", "confidence": 0, "explanation": f"json-error:{e}"}
        js["category"] = validate_category(js.get("category"))
        try:
            js["confidence"] = int(js.get("confidence", 0))
        except Exception:
            js["confidence"] = 0
        return js

    # Batch klasifikace (doporučeno)
    def classify_batch(self, items: List[dict], lang: str = "cs") -> List[dict]:
        """
        items: [{"id":"...", "subject":"...", "desc":"..."}, ...]
        return: [{"id":"...","category":"...","confidence":int,"explanation":"..."}, ...]
        """
        joined = "\n".join(
            f'- id:{it["id"]} subject:{normalize_text(it["subject"])[:150]} desc:{normalize_text(it["desc"])[:300]}'
            for it in items
        )
        schema = """
Vrať POUZE JSON pole objektů:
[
 {"id":"...","category":"...","confidence":0-100,"explanation":"..."},
 ...
]
"""
        system = f"Odpovídej {lang}. Jsi IT analytik. Klasifikuj položky do jedné z daných kategorií."
        user_prompt = f"""Seznam kategorií:
{json.dumps(CATEGORIES, ensure_ascii=False, indent=2)}

Položky:
\"\"\"
{joined}
\"\"\"

{schema}
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = self._post(payload, timeout=300)
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", content, flags=re.S)
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
        except Exception:
            return []
        out = []
        for r in arr:
            cid = str(r.get("id", ""))
            cat = validate_category(r.get("category"))
            try:
                conf = int(r.get("confidence", 0))
            except Exception:
                conf = 0
            exp = r.get("explanation", "")
            out.append({"id": cid, "category": cat, "confidence": conf, "explanation": exp})
        return out

    # Storytelling
    def story(self, rows: List[dict], subj: str, desc: str, lang="cs") -> str:
        samples = []
        for r in rows[:100]:
            samples.append(f"- {normalize_text(r.get(subj,''))[:150]} :: {normalize_text(r.get(desc,''))[:300]}")
        joined = "\n".join(samples[:60])
        system = f"Odpovídej {lang}. Jsi datový analytik IT helpdesku."
        user_prompt = f"""
Ukázky tiketů:
\"\"\"
{joined}
\"\"\"

Napiš stručné shrnutí (max 10 vět), ideálně po odstavcích:
1) hlavní témata a trendy
2) nejčastější kategorie/problémy
3) doporučení (co změnit, automatizovat, školit)
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = self._post(payload, timeout=300)
        return data["choices"][0]["message"]["content"].strip()


# ============================
# 3) I/O – robustní loader
# ============================

def load_dataframe(path: str) -> pd.DataFrame:
    """Načte XLSX nebo CSV robustně (autodetekce oddělovače; fallbacky; UTF-8(-sig))."""
    p = path.lower()

    # Excel (nejjednodušší varianta)
    if p.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, engine="openpyxl")
        return df.fillna("")

    # CSV – načti bezpečně, sjednoť encoding, čisti CRLF
    with open(path, "rb") as fh:
        raw = fh.read()

    try:
        txt = raw.decode("utf-8-sig", errors="replace")
    except Exception:
        txt = raw.decode("utf-8", errors="replace")

    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # 1) autodetekce (engine="python", sep=None)
    try:
        df = pd.read_csv(
            io.StringIO(txt),
            engine="python",
            sep=None,               # autodetect ',', ';', '\t', '|'
            on_bad_lines="skip",
            dtype=str
        )
        if df.shape[1] >= 2:
            return df.fillna("")
    except Exception:
        pass

    # 2) fallbacky
    for sep_try in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(txt),
                engine="python",
                sep=sep_try,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
                dtype=str
            )
            if df.shape[1] >= 2:
                return df.fillna("")
        except Exception:
            continue

    # 3) nouzový poslední pokus
    df = pd.read_csv(io.StringIO(txt), engine="python", on_bad_lines="skip", dtype=str)
    return df.fillna("")


def normalize_name(s: str) -> str:
    s = safe_text(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "").replace("'", "")
    return s


def find_column(df: pd.DataFrame, wanted: str, aliases: List[str]) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    candidates = [normalize_name(wanted)] + [normalize_name(a) for a in aliases]
    for key in candidates:
        if key in norm_map:
            return norm_map[key]
    return None


# ============================
# 4) Ukládání XLSX (s fallbackem)
# ============================

def save_xlsx(out_path: str,
              df: pd.DataFrame,
              ids: List[str],
              id_label: str,
              story: bool,
              offline_only: bool,
              llm,                 # LLMClient nebo None
              subj_real: str,
              desc_real: str,
              lang: str):
    """
    Uloží výstup do XLSX (Data, AI_Staging, volitelně Storytelling).
    Zkouší xlsxwriter, při chybě přepne na openpyxl.
    """
    staging = pd.DataFrame({
        id_label: ids,
        "AI_Category": df["AI_Category"],
        "AI_Confidence": df["AI_Confidence"],
        "AI_Explanation": df["AI_Explanation"],
        "AI_Method": df["AI_Method"],
    })

    def _write(writer):
        df.to_excel(writer, index=False, sheet_name="Data")
        staging.to_excel(writer, index=False, sheet_name="AI_Staging")
        if story and (not offline_only) and (df.shape[0] > 0) and llm is not None:
            rows = df[[subj_real, desc_real]].to_dict(orient="records")
            text = llm.story(rows, subj=subj_real, desc=desc_real, lang=lang)
            pd.DataFrame({"Story": [text]}).to_excel(writer, index=False, sheet_name="Storytelling")

    # preferuj xlsxwriter → fallback openpyxl
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            _write(writer)
    except ModuleNotFoundError:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            _write(writer)


# ============================
# 5) Argumenty
# ============================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Vstup CSV/XLSX")
    p.add_argument("--output", required=True, help="Výstup .xlsx (pouze XLSX)")
    p.add_argument("--subject-col", "--subject_col", dest="subject_col", default="Problém")
    p.add_argument("--desc-col", "--desc_col", dest="desc_col", default="Popis problému")
    p.add_argument("--id-col", default=None)
    p.add_argument("--provider", choices=["openai"], default="openai")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--rpm", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=120)
    p.add_argument("--rule-threshold", type=int, default=80)
    p.add_argument("--lang", default="cs")
    p.add_argument("--story", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--offline-only", action="store_true")
    return p.parse_args()


# ============================
# 6) Hlavní běh
# ============================

def main():
    args = parse_args()

    # Enforce .xlsx
    if not args.output.lower().endswith(".xlsx"):
        raise ValueError("Výstup musí být .xlsx (XLSX-only).")

    df = load_dataframe(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # Najdi sloupce (aliasy – CZ/EN, bez diakritiky)
    subj_real = find_column(
        df, args.subject_col,
        aliases=["subject", "nazev", "téma", "tema", "problem", "problém", "ticket", "predmet", "předmět"]
    )
    desc_real = find_column(
        df, args.desc_col,
        aliases=["description", "popis", "detail", "message", "text"]
    )
    if not subj_real or not desc_real:
        raise ValueError(f"Chybí sloupce {args.subject_col} / {args.desc_col} (zkontroluj hlavičky a kódování).")

    # ID mapping
    if args.id_col and args.id_col in df.columns:
        ids = df[args.id_col].astype(str).tolist()
        id_label = args.id_col
    else:
        ids = [str(i) for i in range(len(df))]
        id_label = "Index"

    # Klient LLM
    llm = None
    if not args.offline_only:
        if args.provider != "openai":
            raise ValueError("Aktuálně je podporován pouze provider 'openai'.")
        llm = LLMClient(args.model, rpm=args.rpm)

    # RULE PASS
    out_cat, out_conf, out_exp, out_method = [], [], [], []
    to_llm: List[int] = []
    rule_threshold = int(args.rule_threshold)

    for i, row in df.iterrows():
        subject = normalize_text(row.get(subj_real, ""))
        description = normalize_text(row.get(desc_real, ""))
        full = f"Předmět: {subject}\nPopis: {description}"

        c0, conf0, e0 = category_by_rules(full)
        if conf0 >= rule_threshold or args.offline_only or not llm:
            out_cat.append(validate_category(c0))
            out_conf.append(int(conf0))
            out_exp.append(e0)
            out_method.append("rules")
        else:
            out_cat.append("Ostatní")
            out_conf.append(int(conf0))
            out_exp.append(e0)
            out_method.append("rules")
            to_llm.append(i)

        if (i + 1) % 50 == 0:
            print(f"[INFO] Rule pass hotov pro {i+1} řádků…")

    # LLM PASS – batch
    if llm and not args.offline_only and len(to_llm) > 0:
        bs = max(1, int(args.batch_size))
        for start in range(0, len(to_llm), bs):
            chunk_idx = to_llm[start:start + bs]
            items = []
            for i in chunk_idx:
                items.append({
                    "id": ids[i],
                    "subject": normalize_text(df.iloc[i][subj_real]),
                    "desc": normalize_text(df.iloc[i][desc_real]),
                })
            results = llm.classify_batch(items, lang=args.lang)
            res_map = {r["id"]: r for r in results}
            for i in chunk_idx:
                r = res_map.get(ids[i])
                if r:
                    out_cat[i] = r["category"]
                    out_conf[i] = max(out_conf[i], r["confidence"])
                    out_exp[i] = r.get("explanation", "")
                    out_method[i] = "llm-batch"
        print(f"[INFO] LLM batch pass hotov: {len(to_llm)} řádků klasifikováno přes LLM.")

    # Výstupní sloupce
    df["AI_Category"] = out_cat
    df["AI_Confidence"] = out_conf
    df["AI_Explanation"] = out_exp
    df["AI_Method"] = out_method

    # Uložení – XLSX only (s fallbackem na openpyxl)
    save_xlsx(
        out_path=args.output,
        df=df,
        ids=ids,
        id_label=id_label,
        story=args.story,
        offline_only=args.offline_only,
        llm=llm,
        subj_real=subj_real,
        desc_real=desc_real,
        lang=args.lang
    )

    print("=== HOTOVO: Výstup uložen jako XLSX ===")
    print(args.output)


if __name__ == "__main__":
    main()
