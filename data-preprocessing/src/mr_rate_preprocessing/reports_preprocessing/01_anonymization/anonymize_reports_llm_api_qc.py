# =======================================================================
# Parallel Qwen Anonymization Script for Turkish Reports
# Each SLURM task processes its shard of the data on its own GPU.
# =======================================================================
import concurrent.futures
import os
import json

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mr_rate_preprocessing.reports_preprocessing.llm import fetch_llm_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='/home/USZ/zhliracc/data/mr_rate/6445959_metadata/6445959_anonymized_shards/anonymized_rank_.csv')
    parser.add_argument("--output_dir", type=str, default='/home/USZ/zhliracc/data/mr_rate/6445959_metadata/check_named_entities')
    args = parser.parse_args()
    

    # =======================================================================
    # 2. Load Turkish Reports and shard
    # =======================================================================
    input_file = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"anonymized_rank_qc.csv")

    print(f"[Rank ] Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, encoding="utf-8-sig", sep=',')
    df = df.iloc[:5]
    if not {"AccessionNo", "Anonymized_Rapor"}.issubset(df.columns):

        raise ValueError("CSV must contain 'AccessionNo' and 'Anonymized_Rapor' columns.")

    total_reports = len(df)
    print(f"[Rank ] Total reports: {total_reports}, this shard: {len(df)}")

    # Resume support: skip already-processed rows
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
        done_accessions = set(done_df["AccessionNo"].astype(str))
        mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
        df = df[mask].reset_index(drop=True)
        print(f"[Rank ] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

    if len(df) == 0:
        print(f"[Rank ] All reports already processed.")
        exit(0)

    # =======================================================================
    # 3. Prepare Prompts for Anonymization
    # =======================================================================
    system_prompt = """You are a deterministic data engineering pipeline component specialized in Privacy Compliance and PII (Personally Identifiable Information) detection for Swiss and German medical records.

Your sole task is to analyze the provided German medical report and extract all personal, institutional, and temporal (date) identifiers.

Output Constraints:

Your output must be RAW JSON ONLY (a single flat JSON array).

Do not include markdown formatting wraps (such as json or ).

Do not include any conversational text, explanations, or leading/trailing commentary.

If no PII or target dates are found, return an empty list: []

Extraction Target Categories:

PATIENT_NAME: Patient names (e.g., "Herr Müller", "Max Mustermann").

DOCTOR_NAME: Doctor names and clinical academic titles (e.g., "Dr. med. Meier", "Frau Dr. Schmidt").

HOSPITAL_NAME: Hospital, clinic, or facility names (e.g., "Universitätsspital Zürich", "USZ", "Klinik im Park").

DEPARTMENT: Medical departments, divisions, or institutes (e.g., "Institut für Radiologie", "Frauenklinik").

DATE: Specific dates linked to the patient or the report (e.g., birth dates like "geb. 1970" or exam/report dates like "vom 02.09.2020").

Mandatory Output Schema:

You must return a flat list of JSON objects matching this exact structure:

[
{
"category": "PATIENT_NAME" | "DOCTOR_NAME" | "HOSPITAL_NAME" | "DEPARTMENT" | "DATE",
"value": "The exact extracted text from the document",
"confidence_score": 0.0 to 1.0
}
]

Example Target Output:

[
{
"category": "PATIENT_NAME",
"value": "Hans Meier",
"confidence_score": 1.0
},
{
"category": "DATE",
"value": "05.12.2018",
"confidence_score": 1.0
},
{
"category": "DOCTOR_NAME",
"value": "Dr. med. Anna Keller",
"confidence_score": 1.0
},
{
"category": "HOSPITAL_NAME",
"value": "Universitätsspital Zürich",
"confidence_score": 0.95
}
]"""

    print(f"[Rank ] Preparing prompts...")

    reports_list = df["Anonymized_Rapor"].tolist()
    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 5
    MAX_TOK = 30000

    print(f"[Rank ] Running anonymization ({len(df)} reports)...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOK
    )



    # =======================================================================
    # 5. Process in chunks with checkpointing
    # =======================================================================

    total_processed = 0
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=MAX_TOK,
    )

#
    messages = [[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{report}"
                }
            ]
        for report in reports_list
    ]
    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        chunk_prompts = messages[chunk_start:chunk_end]

        print(f"[Rank ] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
        outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(
                executor.map(
                    lambda msg: fetch_llm_response(msg, sampling_params),
                    messages,
                )
            )
        anonym_qc_rows = []
        qc_rank = []
        check_value = []

        for i, output in enumerate(outputs):
            result = json.loads(output)
            accession = chunk_df.iloc[i]["AccessionNo"]
            if len(result) == 0:
                anonym_qc_rows.append({
                    "AccessionNo": accession,
                    "qc_rank": 'gut',
                    "check_value": None
                })

            else:
                anonym_qc_rows.append({
                    "AccessionNo": accession,
                    "qc_rank": 'check',
                    "check_value": output
                })

        anonym_qc_df = pd.DataFrame(anonym_qc_rows)

        anonym_qc_df.to_csv(output_file, index=False)



        print(f"[Rank ] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank ] Done. Processed {total_processed} reports -> {output_file}")





