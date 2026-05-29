# =======================================================================
# Parallel Turkish-to-English Translation for Anonymized Reports
# Each SLURM task translates its shard on its own GPU via vLLM.
# =======================================================================

import os
import json

# --- HF / Cache Setup ---
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"] + "/transformers"
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] + "/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =======================================================================
# 0. CLI arguments and rank/world size
# =======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="translated_shards")
args = parser.parse_args()

# =======================================================================
# 1. Load local Model with vLLM
# =======================================================================
from vllm import LLM
from transformers import AutoTokenizer

model_dir = "/data/zhliracc/llm_models/Qwen/Qwen3.5-35B-A3B-FP8"

llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    dtype="bfloat16",
    skip_tokenizer_init=False,
    limit_mm_per_prompt={"image": 0, "video": 0},  # deaktiviert multimodal

)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# =======================================================================
# 2. Load Anonymized Reports and shard
# =======================================================================
input_file = args.input_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"translated_rank.csv")

print(f"[Rank   Reading CSV file: {input_file}")
df = pd.read_csv(input_file, encoding="utf-8-sig", sep=";")
if "Anonymized_Rapor" not in df.columns:
    print("CSV must contain 'Anonymized_Rapor' column. using original data without anonymization.")
    df["Anonymized_Rapor"] = df["Rumpf"]
    df["AccessionNo"] = df["AccNr"].str.split("-", expand=True)[0]

total_reports = len(df)

# Shard: each rank takes every WORLD_SIZE-th row
print(f"[Rank    ] Total reports: {total_reports}, this shard: {len(df)}")

# Resume support: skip already-processed rows
if os.path.exists(output_file):
    done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
    done_accessions = set(done_df["AccessionNo"].astype(str))
    mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
    df = df[mask].reset_index(drop=True)
    print(f"[Rank    ] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

if len(df) == 0:
    print(f"[Rank    ] All reports already translated.")
    exit(0)

# =======================================================================
# 3. Prepare Prompts for Translation
# =======================================================================
system_prompt = """You are a professional medical translator specializing in Deutsch to English translation.

Your task is to translate Turkish medical reports to English with the following strict requirements:

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
   - Do NOT include the original Turkish text.
   - Maintain paragraph structure and formatting from the original.
   - Send the translation only ONCE, do not repeat."""

print(f"[Rank    ] Preparing prompts...")

reports_list = df["Anonymized_Rapor"].tolist()

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(report)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for report in reports_list
]

# =======================================================================
# 4. Batched Generation with vLLM
# =======================================================================
MAX_NEW = 30000
CHUNK_SIZE = 5000

print(f"[Rank    ] Running translation ({len(df)} reports)...")
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW,
)


def clean_output(text):
    """Strip thinking tags and common LLM prefixes."""
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


# =======================================================================
# 5. Process in chunks with checkpointing
# =======================================================================
save_cols = [c for c in df.columns if c != "Translated_Rapor"] + ["Translated_Rapor"]

total_processed = 0

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[chunk_start:chunk_end].copy()
    chunk_prompts = prompts[chunk_start:chunk_end]

    print(f"[Rank    ] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
    outputs = llm.generate(chunk_prompts, sampling_params)

    translations = []
    for output in outputs:
        translated = clean_output(output.outputs[0].text)
        translations.append(translated)

    chunk_df["Translated_Rapor"] = translations

    result_cols = [c for c in save_cols if c in chunk_df.columns]
    result_chunk = chunk_df[result_cols]

    write_header = not os.path.exists(output_file)
    result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    total_processed += len(chunk_df)
    print(f"[Rank    ] Saved chunk ({total_processed}/{len(df)} total).")

print(f"[Rank    ] Done. Translated {total_processed} reports -> {output_file}")
