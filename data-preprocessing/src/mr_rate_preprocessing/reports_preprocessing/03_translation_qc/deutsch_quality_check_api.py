# =======================================================================
# Parallel Translation Quality Check
# Compares deutsch and English reports sentence by sentence
# Saves thinking tokens separately for review
# =======================================================================

import os
import json
import requests
import concurrent.futures
import argparse
import pandas as pd

from mr_rate_preprocessing.reports_preprocessing.llm import fetch_llm_response

if __name__ == "__main__":
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    # =======================================================================
    # 0. CLI arguments and rank/world size
    # =======================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="quality_check_shards")
    args = parser.parse_args()
    # -----------------------------------------------------------------------
    # API Configuration
    # -----------------------------------------------------------------------

    API_URL = "http://localhost:8000/v1/chat/completions"
    MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    session = requests.Session()

    # =======================================================================
    # 2. Load Reports and shard
    # =======================================================================
    input_file = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"tran_qc_rank_.csv")

    print(f"[Rank ] Reading CSV file: {input_file}")
    df_all = pd.read_csv(input_file, encoding="utf-8-sig")
    df = df_all.iloc[:5]
    total_reports = len(df)

    # Shard
    print(f"[Rank ] Total reports: {total_reports}, this shard: {len(df)}")

    # Resume support
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
        done_accessions = set(done_df["AccessionNo"].astype(str))
        mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
        df = df[mask].reset_index(drop=True)
        print(f"[Rank ] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

    if len(df) == 0:
        print(f"[Rank ] All reports already checked.")
        exit(0)

    # =======================================================================
    # 3. System Prompt
    # =======================================================================
    system_prompt = """You are a bilingual (German-English) medical radiology translation quality reviewer.

    You will receive a German radiology report and its English translation. Your job is to find REAL translation errors that change clinical meaning.

    IMPORTANT: You must be VERY conservative. Only flag errors you are 100% certain about. When in doubt, PASS.

    CRITICAL RULE: Before reporting ANY issue, you MUST re-read the English translation to confirm the error actually exists. Do NOT report an error from memory — go back and verify the exact English text contains the problem you think it does. Many errors you think you see will not actually be there when you look again.

    ## What counts as a REAL error (FAIL):
    - WRONG LATERALITY: "rechts" (right) translated as "left", or vice versa, or laterality added/removed
    - WRONG ANATOMY: one anatomical structure translated as a completely different one (e.g., "Gallenblase" → "urinary bladder", "Kleinhirnbrückenwinkel" → "cerebellopontine angle" mistranslated as "cerebral cortex")
    - WRONG NUMBERS: measurements or counts changed (e.g., "6 mm" → "8 mm", "drei Läsionen" → "two lesions")
    - MISSING FINDINGS: an entire clinical finding or diagnosis is absent from the English (not just a single word)
    - OPPOSITE MEANING: "ausgeschlossen" translated as "confirmed", "Belüftung" as "opacification", "oder" as "and" when it changes differential diagnosis
    - WRONG CLINICAL TERM: "Bandscheibenvorfall" as "disc bulge" (if it's a herniation), "Vasogenes Ödem" as "vascular edema", "Liquorraum" as "vascular space"

    ## What is NOT an error (PASS these):
    - Spelling variants: "disk"/"disc", "millimeter"/"milimeter", "perimesencephalic"/"permesencephalic"
    - Synonym choices: "sichtbar"/"observed"/"noted", "Milz"/"spleen"/"lien", "Herz"/"heart"/"cor"
    - Word order differences that preserve meaning
    - Missing greeting/closing phrases ("Sehr geehrte Kolleginnen und Kollegen", "Mit freundlichen Grüßen")
    - Section header format differences (e.g., "Befund:" vs "Findings:")
    - Minor grammatical differences (singular/plural when context is clear, adjective/noun forms)
    - Technique section details (contrast agent names, doses, sequence parameters like "T2w TSE")
    - Placeholder tokens like [patient_1], [date_1]
    - Order differences in lists of structures (e.g., anatomical structures listed in a different sequence)
    - "3D-Sequenz" vs "3D sequence", "KM-Anreicherung" vs "contrast enhancement"

    ## Response format:
    For each issue you report, you MUST quote the exact German text AND the exact English text that contains the error. If you cannot find the error in the actual English text, do NOT report it.

    {"verdict": "PASS", "issues": []}
    or
    {"verdict": "FAIL", "issues": ["EXACT German: '...' → EXACT English: '...' — [description of error]"]}

    Return ONLY the JSON."""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 5
    MAX_NEW = 30000
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW,
    )

    total_processed = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        # Wir bauen jetzt Listen von Nachrichten-Strukturen (Chat-Format) statt roher Strings
        chunk_messages = []
        for _, row in chunk_df.iterrows():
            deutsch = (
                row["Anonymized_Rapor"]
                if isinstance(row["Anonymized_Rapor"], str)
                else "[EMPTY]"
            )
            english = (
                row["Translated_Rapor"]
                if isinstance(row["Translated_Rapor"], str)
                else "[EMPTY]"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"deutsch ORIGINAL:\n{deutsch}\n\n---\n\nENGLISH TRANSLATION:\n{english}",
                },
            ]
            chunk_messages.append(messages)

        print(f"[Rank ] Processing chunk {chunk_start}-{chunk_end} / {len(df)} via API...")


        rows = []
        outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(
                lambda msg: fetch_llm_response(messages, sampling_params),
                chunk_messages
            ))
        for i, raw_text in enumerate(outputs):
            raw = raw_text.strip()

            # Split thinking and response (Bleibt exakt gleich wie in deinem Code)
            thinking = ""
            response = raw
            if "</think>" in raw:
                think_end = raw.rfind("</think>")
                thinking = raw[:think_end].strip()
                if thinking.startswith("<think>"):
                    thinking = thinking[len("<think>"):].strip()
                response = raw[think_end + len("</think>"):].strip()

            # Parse verdict (Bleibt exakt gleich wie in deinem Code)
            verdict = "unknown"
            issues = ""
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json.loads(response[json_start:json_end])
                    verdict = parsed.get("verdict", "unknown").upper()
                    issues_list = parsed.get("issues", [])
                    if isinstance(issues_list, list):
                        issues = " | ".join(str(x) for x in issues_list)
                    else:
                        issues = str(issues_list)
            except json.JSONDecodeError:
                pass

            # Fallback parsing (Bleibt exakt gleich wie in deinem Code)
            if verdict == "unknown":
                resp_lower = (raw + " " + response).lower()
                if '"pass"' in resp_lower or "'pass'" in resp_lower:
                    verdict = "PASS"
                elif '"fail"' in resp_lower or "'fail'" in resp_lower:
                    verdict = "FAIL"

            rows.append(
                {
                    "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
                    "verdict": verdict,
                    "issues": issues,
                    "thinking": thinking,
                    "response": response,
                }
            )

        result_chunk = pd.DataFrame(rows)

        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(
            output_file,
            mode="a",
            index=False,
            header=write_header,
            encoding="utf-8-sig",
        )

        total_processed += len(chunk_df)

        # Quick stats
        verdicts = pd.DataFrame(rows)["verdict"].value_counts()
        print(f"[Rank ] Chunk stats: {dict(verdicts)}")
        print(f"[Rank ] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank ] Done. Checked {total_processed} reports -> {output_file}")
