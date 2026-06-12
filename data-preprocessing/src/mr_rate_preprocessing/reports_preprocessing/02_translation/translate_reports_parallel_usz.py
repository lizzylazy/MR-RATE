#!/usr/bin/env python3
# =======================================================================
# Parallel German-to-English Translation for Anonymized Medical Reports
# Using OpenAI-compatible vLLM API
# =======================================================================

import os
import requests
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------

API_URL = "http://vl90864.usz.ch:8001/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

# -----------------------------------------------------------------------
# Original System Prompt
# -----------------------------------------------------------------------

system_prompt = """You are a professional medical translator specializing in Deutsch to English translation.

Your task is to translate Deutsch medical reports to English with the following strict requirements:

1) TRANSLATION REQUIREMENTS:
   - Translate the entire medical report accurately from Deutsch to English.
   - Preserve all medical terminology with correct English equivalents.
   - Maintain the clinical accuracy and meaning of all diagnoses, findings, and measurements.
   - Keep all numerical values, dates, and units exactly as they appear.
   - Preserve the structure and formatting of the original report.

2) PRESERVED ELEMENTS:
   - All measurements and units (cm, mm, ml, etc.)
   - All dates and times
   - All numerical values and ranges
   - Medical abbreviations (translate or keep standard international ones)
   - Anatomical terms (use standard English medical terminology)

3) ANONYMIZATION TOKENS:
   - The text contains anonymization tokens like [patient_1], [radiologist_1], [date_1], [hospital_1], etc.
   - Keep ALL anonymization tokens EXACTLY as they are. Do NOT translate or modify them.

4) OUTPUT FORMAT REQUIREMENTS:
   - OUTPUT ONLY the translated English text.
   - Do NOT add any introductory phrases like "Here is the translation:" or "Translation:".
   - Do NOT add any explanatory notes or comments.
   - Do NOT include the original Deutsch text.
   - Maintain paragraph structure and formatting from the original.
   - Send the translation only ONCE, do not repeat.
"""

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

session = requests.Session()


def clean_output(text: str) -> str:
    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    prefixes = [
        "Here is the translation:",
        "Translation:",
        "Here is the translated text:",
        "Translated text:",
        "Output:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text


def translate_single(report: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": str(report),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    try:
        response = session.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
            proxies={"http": None, "https": None},

        )

        response.raise_for_status()

        data = response.json()

        message = data["choices"][0]["message"]

        text = (
            message.get("content")
            or message.get("reasoning")
            or ""
        )

        return clean_output(text)

    except Exception as e:
        print(f"Translation failed: {e}")
        return ""


def translate_batch(reports, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(translate_single, reports))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():

    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Translate anonymized medical reports using vLLM API."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default='/data/zhliracc/data/mr_rate_data/patient_report.csv',
        help="Input CSV file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="translated_shards",
        help="Output directory",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API requests",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(
        args.output_dir,
        "translated_rank.csv",
    )

    print(f"Reading CSV: {args.input_file}")

    df = pd.read_csv(
        args.input_file,
        encoding="utf-8-sig",
        sep=";",
    )

    if "Anonymized_Rapor" not in df.columns:
        print(
            "CSV does not contain 'Anonymized_Rapor'. "
            "Using 'Rumpf'."
        )

        df["Anonymized_Rapor"] = df["Rumpf"]
        df["AccessionNo"] = (
            df["AccNr"]
            .astype(str)
            .str.split("-", expand=True)[0]
        )

    total_reports = len(df)

    print(f"Total reports: {total_reports}")

    # -------------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------------

    if os.path.exists(output_file):

        done_df = pd.read_csv(
            output_file,
            usecols=["AccessionNo"],
        )

        done_accessions = set(
            done_df["AccessionNo"].astype(str)
        )

        mask = ~df["AccessionNo"].astype(str).isin(
            done_accessions
        )

        df = df[mask].reset_index(drop=True)

        print(
            f"Resuming: {len(done_accessions)} already done, "
            f"{len(df)} remaining."
        )

    if len(df) == 0:
        print("All reports already translated.")
        return

    # -------------------------------------------------------------------
    # Translation
    # -------------------------------------------------------------------

    CHUNK_SIZE = 8

    save_cols = [
        c for c in df.columns
        if c != "Translated_Rapor"
    ] + ["Translated_Rapor"]

    total_processed = 0

    print(
        f"Running translation for {len(df)} reports "
        f"in chunks of {CHUNK_SIZE}..."
    )

    for chunk_start in range(0, len(df), CHUNK_SIZE):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            len(df),
        )

        chunk_df = df.iloc[
            chunk_start:chunk_end
        ].copy()

        reports = (
            chunk_df["Anonymized_Rapor"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        print(
            f"Processing chunk "
            f"{chunk_start}-{chunk_end} "
            f"/ {len(df)}"
        )

        translations = translate_batch(
            reports,
            workers=args.workers,
        )

        chunk_df["Translated_Rapor"] = translations

        result_cols = [
            c for c in save_cols
            if c in chunk_df.columns
        ]

        result_chunk = chunk_df[result_cols]

        write_header = not os.path.exists(
            output_file
        )

        result_chunk.to_csv(
            output_file,
            mode="a",
            index=False,
            header=write_header,
            encoding="utf-8-sig",
        )

        total_processed += len(chunk_df)

        print(
            f"Saved chunk "
            f"({total_processed}/{len(df)})"
        )

    print(
        f"Done. Translated {total_processed} reports."
    )

    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()