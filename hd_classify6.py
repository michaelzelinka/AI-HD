# -*- coding: utf-8 -*-
"""
Hlavní myšlenka:
 1) RULE-FIRST: nejdřív lokální pravidla (aliasy/keywords). Pokud confidence >= threshold,
    řádek je hotový bez LLM.
 2) BATCH LLM: jen nejasné řádky jdou do LLM ve skupinách (např. 50–200 záznamů),
    výrazně levnější a rychlejší než jeden call na řádek.
 3) STORYTELLING: jeden prompt nad vzorkem/aggregátem, česky (lze přepnout jazykem).

Příklad (OpenAI, česky, storytelling zapnutý):
  python3 hd_classify6.py \
    --input "$HOME/Downloads/incidenty.csv" \
    --output "$HOME/Downloads/storytelling.xlsx" \
    --provider openai \
    --model gpt-4o-mini \
    --rpm 60 \
    --mode batch \
    --batch-size 120 \
    --rule-threshold 80 \
    --lang cs \
    --story

Pozn.: pro jiné hlavičky použij --subject-col / --desc-col (aliasy se spojovníkem i podtržítkem).
"""

from __future__ import annotations
import argparse
import os
import re
import json
import time
import unicodedata
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests



# 1) KATEGORIE

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
    "Instalace / aktualizace software": [
        "instalace", "doinstalace", "nainstalovat", "aktualizace",
        "update", "upgrade", "setup", "install", "reinstall", "patch"
    ],
    "Údržba / čištění disků a souborů": [
        "úklid", "cisteni", "čištění", "volné místo", "smazat", "mazání", "odstranit"
    ],
    "Databáze / Oracle / MCC / Virtuan": [
        "oracle", "mcc", "virtuan", "elfa", "db", "databáze", "database", "sql"
    ],
    "Interní aplikace (Unique, NemExpress, Argus…)": [
        "unique", "nemexpress", "argus", "interní aplikace", "internal app"
    ],
    "Tiskárny a skenery": [
        "tisk", "tisknout", "tiskárna", "tiskarna", "scan", "skener", "toner", "print"
    ],
    "Síť / Wi-Fi / VPN / GlobalProtect": [
        "wifi", "wi-fi", "síť", "sit", "vpn", "gp", "global protect", "připojení", "network"
    ],
    "Office / Outlook / Teams": [
        "outlook", "office", "teams", "email", "mail", "kalendář", "meeting", "pošta"
    ],
    "Hardware (notebook, monitor, periferie)": [
        "notebook", "laptop", "monitor", "obrazovka", "klávesnice", "myš", "baterie", "charger"
    ],
    "Přístupy / oprávnění / účty": [
        "přístup", "oprávnění", "heslo", "password", "login", "přihlášení", "permissions"
    ],
    "Cloud / OneDrive / SharePoint": [
        "onedrive", "sharepoint", "cloud", "sdílená složka", "shared folder", "upload"
    ],
    "Mobil / iPhone / Android": [
        "iphone", "android", "mobil", "sim", "roaming", "hovor"
    ],
    "Ostatní": []
}

ALIASES: List[Tuple[str, str]] = [
    ("global connect", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("globalprotect", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("gp", "Síť / Wi-Fi / VPN / GlobalProtect"),
    ("nelze tisknout", "Tiskárny a skenery"),
    ("nemexpress", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("unique", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("argus", "Interní aplikace (Unique, NemExpress, Argus…)"),
    ("oracle", "Databáze / Oracle / MCC / Virtuan"),
    ("mcc", "Databáze / Oracle / MCC / Virtuan"),
]


def normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\ufeff", "")  # odstraní BOM
    return s.strip()


def normalize_for_match(s: str) -> str:
    """Lower + odstranění diakritiky + kompaktní mezery"""
    s = normalize_text(s).lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def category_by_rules(text: str) -> Tuple[Optional[str], int, str]:
    """Jednoduchý rules engine (aliasy → keywords). Vrací (category, confidence, explanation)."""
    txt = normalize_for_match(text)

    for phrase, cat in ALIASES:
        if normalize_for_match(phrase) in txt:
            return cat, 95, f"alias:{phrase}"

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
        if (normalize_for_match(c) in c_low) or (c_low in normalize_for_match(c)):
            return c
    return "Ostatní"



# 2) LLM klient (OpenAI)

class LLMClient:
    def __init__(self, provider: str, model: str, rpm: int):
        self.provider = provider
        self.model = model
        self.rpm = max(1, int(rpm))
        self._last = 0.0
        self.min_interval = 60.0 / self.rpm

        if provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Chybí OPENAI_API_KEY v prostředí.")
            self.url = "https://api.openai.com/v1/chat/completions"
            self.header_name = "Authorization"
            self.header_value = f"Bearer {self.api_key}"
        else:
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            if not (self.api_key and endpoint and deployment):
                raise ValueError("Chybí Azure proměnné: AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT.")
            self.url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
            self.header_name = "api-key"
            self.header_value = self.api_key

    def _rate_limit(self):
        now = time.time()
        wait = (self._last + self.min_interval) - now
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _post(self, payload: dict, timeout: int = 60) -> dict:
        self._rate_limit()
        r = requests.post(
            self.url,
            headers={self.header_name: self.header_value},
            json=payload,
            timeout=timeout
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        return r.json()

    # --- ŘÁDKOVÁ klasifikace (pro --mode row) ---
    def classify(self, text: str, strict: bool = False, lang: str = "cs") -> Dict:
        system = (
            f"Odpovídej {lang}. Jsi zkušený IT analytik. Urči jednu kategorii z předem daného seznamu."
            if not strict else
            f"Odpovídej {lang}. Jsi zkušený IT analytik. Vyber jednu nejbližší kategorii. 'Ostatní' použij jen v úplné nouzi."
        )
        user_prompt = f"""
Seznam kategorií:
{json.dumps(CATEGORIES, ensure_ascii=False, indent=2)}

Text tiketu:
\"\"\"
{text}
\"\"\"

Vrať POUZE JSON:
{{
  "category": "...",
  "confidence": 0-100,
  "explanation": "..."
}}
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = self._post(payload)
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

    # --- BATCH klasifikace ---
    def classify_batch(self, items: List[dict], lang: str = "cs") -> List[dict]:
        """
        items: [{ "id": "...", "subject": "...", "desc": "..." }, ...]
        return: [{ "id": "...", "category": "...", "confidence": int, "explanation": "..." }, ...]
        """
        schema = """
Vrať POUZE JSON pole objektů tvaru:
[
  {"id":"...", "category":"...", "confidence":0-100, "explanation":"..."},
  ...
]
"""
        # zkrácená reprezentace položek (minifikované, aby se vešly do kontextu)
        joined = "\n".join(
            [f'- id:{it["id"]} | subject:{normalize_text(it["subject"])[:150]} | desc:{normalize_text(it["desc"])[:300]}'
             for it in items]
        )
        system = f"Odpovídej {lang}. Jsi IT analytik. Klasifikuj položky do jedné z daných kategorií."
        user_prompt = f"""Seznam kategorií:
{json.dumps(CATEGORIES, ensure_ascii=False, indent=2)}

Položky (vzorek):
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
        data = self._post(payload, timeout=120)
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

    # --- Storytelling (plain text) ---
    def story(self, rows: List[dict], subject_key: str, desc_key: str, lang: str = "cs") -> str:
        """Stručné shrnutí datasetu (max ~10 vět) – jazyk lze přepnout --lang."""
        samples = []
        for r in rows[:100]:  # limit pro prompt
            s = normalize_text(str(r.get(subject_key, "")))[:150]
            d = normalize_text(str(r.get(desc_key, "")))[:300]
            samples.append(f"- {s} :: {d}")
        joined = "\n".join(samples[:60])

        system = f"Odpovídej {lang}. Jsi datový analytik IT helpdesku. Vytvoř stručné shrnutí trendů a doporučení."
        user_prompt = f"""
Ukázky tiketů (vzorek):
\"\"\"
{joined}
\"\"\"

Napiš přehledné shrnutí (max 10 vět), ideálně po odstavcích:
1) hlavní témata a trendy
2) nejčastější kategorie/problémy
3) doporučení (co změnit, automatizovat, školit)
"""
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }
        data = self._post(payload, timeout=90)
        return data["choices"][0]["message"]["content"].strip()


# 3) I/O, mapování sloupců

# --- Robustní CSV/XLSX loader s autodetekcí oddělovače ---
import csv
import io

def load_dataframe(path: str) -> pd.DataFrame:
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

    # 1) Nejlepší varianta – autodetekce oddělovače
    try:
        df = pd.read_csv(
            io.StringIO(txt),
            engine="python",
            sep=None,              # autodetekce: , ; \t | 
            on_bad_lines="skip",
            dtype=str
        )
        if df.shape[1] >= 2:
            return df.fillna("")
    except Exception:
        pass

    # 2) Ruční fallbacky pro CSV s jediným sloupcem
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

    # 3) Nouzový poslední pokus
    df = pd.read_csv(
        io.StringIO(txt),
        engine="python",
        on_bad_lines="skip",
        dtype=str
    )
    return df.fillna("")


def normalize_name(s: str) -> str:
    s = normalize_text(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "").replace("'", "")
    return s


def find_column(df: pd.DataFrame, wanted: str, aliases: Optional[List[str]] = None) -> Optional[str]:
    aliases = aliases or []
    norm_map = {normalize_name(c): c for c in df.columns}
    candidates = [normalize_name(wanted), *[normalize_name(a) for a in aliases]]
    for key in candidates:
        if key in norm_map:
            return norm_map[key]
    return None



# 4) Argumenty a hlavní běh

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Vstup CSV/XLSX")
    p.add_argument("--output", required=True, help="Výstup XLSX/CSV (doporučeno .xlsx)")
    # alias se spojovníkem i podtržítkem:
    p.add_argument("--subject-col", "--subject_col", dest="subject_col",
                   default=os.getenv("SUBJECT_COL", "Problém"))
    p.add_argument("--desc-col", "--desc_col", dest="desc_col",
                   default=os.getenv("DESC_COL", "Popis problému"))
    p.add_argument("--provider", choices=["openai", "azure"], required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--rpm", type=int, default=60)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--offline-only", action="store_true")
    p.add_argument("--story", action="store_true", help="Vytvořit list Storytelling")
    p.add_argument("--mode", choices=["row", "batch"], default="batch",
                   help="Způsob LLM klasifikace: row (na řádek) nebo batch (ve skupinách)")
    p.add_argument("--batch-size", type=int, default=100, help="Velikost dávky pro batch klasifikaci")
    p.add_argument("--rule-threshold", type=int, default=80,
                   help="Práh confidence z rules; pod ním jde řádek do LLM")
    p.add_argument("--id-col", default=None,
                   help="Název sloupce s ID (když není, použije se index jako ID).")
    p.add_argument("--lang", default="cs", help="Jazyk výstupu pro LLM (např. cs,en).")
    p.add_argument("--story-plain", action="store_true",
                   help="Vynutit storytelling jako čistý text (bez JSON sekcí).")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_dataframe(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # Najdi sloupce (case-insensitive + aliasy)
    subj_real = find_column(
        df, args.subject_col,
        aliases=["subject", "nazev", "téma", "tema"]
    )
    desc_real = find_column(
        df, args.desc_col,
        aliases=["description", "popis", "detail", "text", "message"]
    )
    if not subj_real or not desc_real:
        raise ValueError(f"Chybí sloupce {args.subject_col} / {args.desc_col} (zkontroluj hlavičky a kódování).")

    # ID pro mapování (sloupec nebo index)
    if args.id_col and args.id_col in df.columns:
        ids = list(map(str, df[args.id_col].tolist()))
    else:
        ids = [str(i) for i in range(len(df))]

    # Klient
    llm = None
    if not args.offline_only:
        llm = LLMClient(args.provider, args.model, rpm=args.rpm)

    # --- RULE PASS ---
    out_cat, out_conf, out_exp, out_method = [], [], [], []
    to_llm: List[int] = []
    rule_threshold = int(args.rule_threshold)

    for i, row in df.iterrows():
        subject = normalize_text(row.get(subj_real, ""))
        desc = normalize_text(row.get(desc_real, ""))
        text = f"Předmět: {subject}\nPopis: {desc}"

        c0, k0, e0 = category_by_rules(text)
        if k0 >= rule_threshold or args.offline_only or not llm:
            out_cat.append(validate_category(c0))
            out_conf.append(int(k0))
            out_exp.append(e0)
            out_method.append("rules")
        else:
            out_cat.append("Ostatní")
            out_conf.append(int(k0))
            out_exp.append(e0)
            out_method.append("rules")
            to_llm.append(i)

        if (i + 1) % 50 == 0:
            print(f"[INFO] Rule pass hotov pro {i+1} řádků…")

    # --- LLM PASS (ROW nebo BATCH) ---
    if llm and not args.offline_only and len(to_llm) > 0:
        if args.mode == "row":
            # méně efektivní – pro úplnost
            for i in to_llm:
                subject = normalize_text(df.iloc[i][subj_real])
                desc = normalize_text(df.iloc[i][desc_real])
                text = f"Předmět: {subject}\nPopis: {desc}"

                r1 = llm.classify(text, strict=False, lang=args.lang)
                best_cat = r1.get("category", "Ostatní")
                best_conf = max(out_conf[i], int(r1.get("confidence", 0)))
                best_exp = r1.get("explanation", "")
                method = "llm-pass1"

                if best_cat == "Ostatní":
                    r2 = llm.classify(text, strict=True, lang=args.lang)
                    c2, k2 = r2.get("category", "Ostatní"), int(r2.get("confidence", 0))
                    if c2 != "Ostatní" and k2 >= best_conf:
                        best_cat, best_conf, best_exp = c2, k2, r2.get("explanation", "")
                        method = "llm-pass2"

                out_cat[i] = validate_category(best_cat)
                out_conf[i] = best_conf
                out_exp[i] = best_exp
                out_method[i] = method

        else:
            # BATCH 
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
                res_map = {str(r["id"]): r for r in results}
                for i in chunk_idx:
                    r = res_map.get(ids[i])
                    if r:
                        out_cat[i] = r["category"]
                        out_conf[i] = max(out_conf[i], int(r["confidence"]))
                        out_exp[i] = r.get("explanation", "")
                        out_method[i] = "llm-batch"
            print(f"[INFO] LLM batch pass hotov: {len(to_llm)} řádků klasifikováno přes LLM.")

    # Výstupní sloupce do Data
    df["AI_Category"]   = out_cat
    df["AI_Confidence"] = out_conf
    df["AI_Explanation"] = out_exp
    df["AI_Method"]     = out_method

    # --- Uložení ---
    out_lower = args.output.lower()
    if out_lower.endswith(".xlsx"):
  
        # Preferuj xlsxwriter (lepší formátování), fallback openpyxl
        try:
            with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")

                # AI_Staging – "mezi tabulka" s kompletním výstupem pro jednoduchý import
                staging = pd.DataFrame({
                    (args.id_col or "Index"): ids,
                    "AI_Category": out_cat,
                    "AI_Confidence": out_conf,
                    "AI_Explanation": out_exp,
                    "AI_Method": out_method,
                })
                staging.to_excel(writer, index=False, sheet_name="AI_Staging")

                # Storytelling (volitelně)
                if args.story and (not args.offline_only) and (df.shape[0] > 0):
                    rows = df[[subj_real, desc_real]].to_dict(orient="records")
                    story_text = LLMClient(args.provider, args.model, rpm=args.rpm).story(
                        rows, subject_key=subj_real, desc_key=desc_real, lang=args.lang
                    )
                    story_df = pd.DataFrame({"Story": [story_text]})
                    story_df.to_excel(writer, index=False, sheet_name="Storytelling")

        except ModuleNotFoundError:
            with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")

                staging = pd.DataFrame({
                    (args.id_col or "Index"): ids,
                    "AI_Category": out_cat,
                    "AI_Confidence": out_conf,
                    "AI_Explanation": out_exp,
                    "AI_Method": out_method,
                })
                staging.to_excel(writer, index=False, sheet_name="AI_Staging")

                if args.story and (not args.offline_only) and (df.shape[0] > 0):
                    rows = df[[subj_real, desc_real]].to_dict(orient="records")
                    story_text = LLMClient(args.provider, args.model, rpm=args.rpm).story(
                        rows, subject_key=subj_real, desc_key=desc_real, lang=args.lang
                    )
                    story_df = pd.DataFrame({"Story": [story_text]})
                    story_df.to_excel(writer, index=False, sheet_name="Storytelling")
    else:
        # CSV výstup
        df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n=== HOTOVO (v6 – batch) ===")
    print(f"Výstup: {args.output}")


if __name__ == "__main__":
    main()
