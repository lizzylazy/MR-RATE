# =======================================================================
# Parallel German/English Detection in Medical Reports
# Uses central vLLM API Server to check if reports are English or German
# =======================================================================

import os
import json
import argparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# =======================================================================
# 1. Global Configurations
# =======================================================================
API_URL = "http://vl90864.usz.ch:8001/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

SYSTEM_PROMPT = """You are a language detection expert. You will be given a medical report that is supposed to be in English.

Your task is to determine the PRIMARY language of the report. The report may contain:
- Medical terms in Latin (e.g., "corpus callosum", "sella turcica") - these are NOT German
- Imaging terms used internationally (e.g., "sagittal", "coronal", "axial", "FLAIR", "T1", "T2") - these are NOT German
- Anatomical terms (e.g., "cerebellum", "ventricle", "hippocampus") - these are NOT German

Focus on the SENTENCE STRUCTURE, CAPITALIZATION, and GRAMMATICAL WORDS to determine the language:
- German sentences use capitalized nouns (e.g., "Befund", "Patient", "Veränderung", "Untersuchung")
- German uses words/verbs like: und (and), ein/eine (a/one), mit (with), für (for), ohne (without), ist/sind (is/are), keine (no), normal (normal), regelrecht, unauffällig, zeigt, sichtbar
- English sentences use lowercase nouns (unless proper nouns) and words like: is, are, was, were, the, and, with, in, of, no, normal, observed, noted, shows, visible

Respond with EXACTLY one of these JSON responses:
{"language": "english"} - if the report is primarily in English
{"language": "german"} - if the report is primarily in German
{"language": "mixed"} - if the report has significant portions in both languages

Return ONLY the JSON, no other text."""


# =======================================================================
# 2. Translate/Detect One by One (Single-Input Function)
# =======================================================================
def predict_language_single(report_text, session):
    """
    Sends a single report text to the API and parses the response.
    """
    cleaned_text = report_text if isinstance(report_text, str) and report_text.strip() else "[EMPTY]"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Determine the language of this report:\n\n{cleaned_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }

    raw = "unknown"
    language = "unknown"

    try:
        response = session.post(API_URL, json=payload, timeout=30, proxies={"http": None, "https": None},)
        if response.status_code == 200:
            res_json = response.json()
            raw = res_json['choices'][0]['message']['content'].strip()

            # Post-processing to bypass thinking models if active
            text = raw
            if "</think>" in text:
                text = text[text.rfind("</think>") + len("</think>"):].strip()

            # 1st Extraction Layer: Direct JSON Parse
            try:
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json.loads(text[json_start:json_end])
                    language = parsed.get("language", "unknown").lower()
            except json.JSONDecodeError:
                pass

            # 2nd Extraction Layer: Text Fallback Check
            if language == "unknown":
                search_text = (raw + " " + text).lower()
                if '"german"' in search_text or "'german'" in search_text or "language: german" in search_text:
                    language = "german"
                elif '"english"' in search_text or "'english'" in search_text or "language: english" in search_text:
                    language = "english"
                elif '"mixed"' in search_text or "'mixed'" in search_text:
                    language = "mixed"
        else:
            raw = f"API Error Code {response.status_code}: {response.text}"

    except Exception as e:
        raw = f"Network Error: {str(e)}"

    return language, raw


# =======================================================================
# 3. Multiprocess / Multithread Wrapper Function
# =======================================================================
def process_chunk_parallel(chunk_df, session, max_workers):
    """
    Wraps the concurrent thread pool worker to execute single-translations in parallel.
    """
    rows = []

    # Internal worker to handle row tuple formatting seamlessly
    def worker(row_tuple):
        _, row = row_tuple
        lang, raw_out = predict_language_single(row["Translated_Rapor"], session)
        return {
            "AccessionNo": row["AccessionNo"],
            "detected_language": lang,
            "raw_output": raw_out[:500],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(worker, chunk_df.iterrows())
        for result in results:
            rows.append(result)

    return pd.DataFrame(rows)

# =======================================================================
# 3. Main Execution Block
# =======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='/data/zhliracc/data/mr_rate_data/translated_shards/translated_rank.csv')
    parser.add_argument("--output_dir", type=str, default="detect_deutsch_shards")
    parser.add_argument("--workers", type=int, default=16, help="Parallel API connections per task")
    args = parser.parse_args()

    RANK = int(os.environ.get("SLURM_PROCID", "0"))
    WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

    print(f"[Rank /{WORLD_SIZE}] Starting via API on node {os.environ.get('SLURMD_NODENAME', 'local')}")

    # Initialize connection pooling session
    session = requests.Session()

    # Setup directories and output paths
    input_file = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"detect_rank_.csv")

    print(f"[Rank ] Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    total_reports = len(df)

    # Shard processing across SLURM cluster
    df = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)
    print(f"[Rank ] Total reports: {total_reports}, this shard: {len(df)}")

    # Resume support
    if os.path.exists(output_file):
        try:
            done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
            done_accessions = set(done_df["AccessionNo"].astype(str))
            mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
            df = df[mask].reset_index(drop=True)
            print(f"[Rank ] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")
        except Exception as e:
            print(f"[Rank ] Restarting output file parsing error: {e}")

    if len(df) == 0:
        print(f"[Rank ] All reports already checked.")
        return

    # Batched Multi-Threaded Processing Loop
    CHUNK_SIZE = 100
    total_processed = 0

    print(f"[Rank {RANK}] Starting processing pipeline with {args.workers} network threads...")

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end]

        print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")

        # Calling the newly built multiprocessing/multithreading layer
        result_chunk = process_chunk_parallel(chunk_df, session, max_workers=args.workers)

        # Commit changes progressively to disk
        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)

        # Print real-time chunk statistics
        langs = result_chunk["detected_language"].value_counts()
        print(f"[Rank {RANK}] Chunk stats: {dict(langs)}")
        print(f"[Rank {RANK}] Progress saved ({total_processed}/{len(df)} total).")

    print(f"[Rank {RANK}] Done. Verified output -> {output_file}")


if __name__ == "__main__":
    main()