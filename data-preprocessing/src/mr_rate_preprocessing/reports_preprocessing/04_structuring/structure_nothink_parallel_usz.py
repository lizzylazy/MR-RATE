# =======================================================================
# Parallel Report Structuring — No-Think mode for stubborn reports
# =======================================================================
import concurrent.futures
import os
import json


import argparse
import pandas as pd
from vllm import SamplingParams

from mr_rate_preprocessing.reports_preprocessing.llm import fetch_llm_response

if __name__ == "__main__":
    # =======================================================================
    # 0. CLI arguments 
    # =======================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='/data/zhliracc/data/mr_rate_data/translated_shards/translated_rank.csv')
    parser.add_argument("--output_dir", type=str, default="/data/zhliracc/data/mr_rate_data/structure_shards")
    args = parser.parse_args()

    # =======================================================================
    # 2. Load Reports and shard
    # =======================================================================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"structure_rank_.csv")

    print(f"[Rank ] Reading CSV file: {args.input_file}")
    df = pd.read_csv(args.input_file, encoding="utf-8-sig")
    df = df.iloc[:5]
    total_reports = len(df)

    print(f"[Rank ] Total reports: {total_reports}, this shard: {len(df)}")

    # Resume support
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
        done_accessions = set(done_df["AccessionNo"].astype(str))
        mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
        df = df[mask].reset_index(drop=True)
        print(f"[Rank ] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

    if len(df) == 0:
        print(f"[Rank ] All reports already structured.")
        exit(0)

    # =======================================================================
    # 3. System Prompt — with /no_think to disable thinking
    # =======================================================================
    system_prompt = """You are a medical radiology report structuring assistant. You will receive an English radiology report and must parse it into 4 structured sections. Do NOT omit or add any content — just parse and format.

/no_think

CRITICAL: NEVER use "..." or ellipsis as a placeholder in any field. You MUST include the FULL actual content for every section. If a section is empty in the original report, use an empty string "". But if content exists, you MUST include ALL of it.

## Your task:
1. Parse the report into exactly 4 sections: clinical_information, technique, findings, impression.
2. Keep ALL content. Do NOT remove any information.
3. Normalize the formatting as specified below.

## Section extraction:
- **clinical_information**: The clinical indication/history/reason for the exam. If none exists, use "".
- **technique**: The imaging technique, sequences, contrast info. Include ALL technique info (brain, cervical, thoracic, etc.). If none exists, use "".
- **findings**: ALL observations from the report.
- **impression**: ALL conclusions from the report.

## Multi-section reports:
Many reports cover multiple exams (e.g., "BRAIN MRI, CERVICAL AND THORACIC VERTEBRA MRI"). For these:

### Findings:
- Add a section title line before each section's findings, derived from the report title.
- Example for "BRAIN AND CERVICAL VERTEBRA MRI":
  Cranial:
  [cranial findings as paragraph sentences...]

  Cervical:
  [cervical findings as paragraph sentences...]

- Example for "BRAIN MRI, CERVICAL AND THORACIC VERTEBRA MRI":
  Cranial:
  [cranial findings...]

  Cervical:
  [cervical findings...]

  Thoracic:
  [thoracic findings...]

- If the report only covers one exam (e.g., "CRANIAL MRI"), do NOT add any section title — just write the findings directly.

### Impression:
- Same logic: add section title lines for multi-section reports, no titles for single-section reports.
- Example:
  Cranial:
  — [cranial impression bullet 1]
  — [cranial impression bullet 2]

  Cervical:
  — [cervical impression bullet 1]

### How to determine section titles:
- Use the report's own title line. If it says "BRAIN AND CERVICAL VERTEBRA MRI", use "Cranial:" and "Cervical:".
- If the title says "CRANIAL, CERVICAL AND THORACIC VERTEBRA MRI", use "Cranial:", "Cervical:", "Thoracic:".
- If MR angiography is a separate section in the report, use "MR Angiography:" or similar based on what the report says.
- Use short, clean labels: "Cranial:", "Cervical:", "Thoracic:", "Lumbar:", "MR Angiography:".

## Formatting rules:

### Findings format — PARAGRAPH SENTENCES:
- Write findings as flowing paragraph text with sentences.
- Remove all bullet points (•, *, ·, -, numbered lists) from findings and convert to plain sentences.
- Each distinct observation should be a sentence. Separate paragraphs with a single newline.
- Remove excessive blank lines.
- Remove tab characters.

### Impression format — BULLET POINTS:
- Each impression item must be a bullet point using "— " (em dash followed by a space, Unicode U+2014).
- One finding per bullet point.
- If the original impression is a single sentence, still format it as "— sentence".

### General:
- Keep all content within sections exactly as-is (measurements, dates, placeholder tokens like [patient_1], [date_1], [radiologist_1]).
- Remove standalone greeting lines, title lines, and trailing signatures that are outside the 4 sections.

## Output format:
Return ONLY valid JSON, no preamble, no code fences.

If the report contains valid radiology content:
{"clinical_information": "...", "technique": "...", "findings": "...", "impression": "..."}

If the report is garbage (empty, placeholder only, not a radiology report):
{"invalid": true, "reason": "brief description"}

CRITICAL: Output ONLY the JSON. Do NOT omit any content from the report."""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 5
    MAX_TOK = 2048
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=MAX_TOK,
    )

    def parse_response(raw_text):
        """Strip thinking/fences and parse JSON from response."""
        response = raw_text.strip()
        if "</think>" in response:
            response = response[response.rfind("</think>") + len("</think>"):].strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1]
            if response.endswith("```"):
                response = response[:-3].strip()
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end]), response
        except json.JSONDecodeError:
            pass
        return None, response

    def fix_impression_bullets(text):
        """Ensure impression uses em dash (—) bullets, no leading single quotes."""
        if not text:
            return text
        lines = text.split("\n")
        fixed = []
        for line in lines:
            # Strip leading single quote
            if line.startswith("'"):
                line = line[1:]
            # Replace regular dash bullets with em dash
            if line.startswith("- "):
                line = "\u2014 " + line[2:]
            fixed.append(line)
        return "\n".join(fixed)

    def make_row(i, chunk_df, parsed, response):
        """Build output row from parsed JSON."""
        raw_report = str(chunk_df.iloc[i]["Translated_Rapor"]) if isinstance(chunk_df.iloc[i]["Translated_Rapor"], str) else ""
        base = {
            "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
           # "UID": chunk_df.iloc[i]["UID"],
            "raw_report": raw_report,
        }
        if parsed is None:
            return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                    "parse_status": "parse_failed", "invalid_reason": response[:200]}
        if parsed.get("invalid"):
            return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                    "parse_status": "invalid", "invalid_reason": parsed.get("reason", "")}
        if "findings" in parsed or "impression" in parsed:
            return {**base,
                    "clinical_information": parsed.get("clinical_information", ""),
                    "technique": parsed.get("technique", ""),
                    "findings": parsed.get("findings", ""),
                    "impression": fix_impression_bullets(parsed.get("impression", "")),
                    "parse_status": "ok", "invalid_reason": ""}
        return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                "parse_status": "parse_failed", "invalid_reason": response[:200]}

    total_processed = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        # Build clean message structure for the initial run
        chunk_messages = []
        for _, row in chunk_df.iterrows():
            english = (
                row["Translated_Rapor"]
                if isinstance(row["Translated_Rapor"], str)
                else "[EMPTY]"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Structure this radiology report:\n\n{english}",
                },
            ]
            chunk_messages.append(messages)

        print(f"[Rank ] Processing chunk {chunk_start}-{chunk_end} / {len(df)} via API...")

        # Concurrently process the initial chunk messages
        outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(
                executor.map(
                    lambda msg: fetch_llm_response(msg, sampling_params),
                    chunk_messages,
                )
            )

        rows = []
        retry_indices = []
        for i, output in enumerate(outputs):
            # output is the raw text string returned by fetch_llm_response
            parsed, response = parse_response(output)
            row = make_row(i, chunk_df, parsed, response)
            if row["parse_status"] == "parse_failed":
                retry_indices.append(i)
            else:
                rows.append(row)

        # Retry failures with pre-filled assistant prefix to force JSON
        if retry_indices:
            print(f"[Rank ] Retrying {len(retry_indices)} parse failures with JSON prefix...")

            retry_messages = []
            for i in retry_indices:
                english = (
                    chunk_df.iloc[i]["Translated_Rapor"]
                    if isinstance(chunk_df.iloc[i]["Translated_Rapor"], str)
                    else "[EMPTY]"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Structure this radiology report:\n\n{english}",
                    },
                    {
                        "role": "assistant",
                        "content": "{",
                    },
                ]
                retry_messages.append(messages)

            # Keep it as a dictionary so **retry_params unpacking works inside your script
            retry_params = {"temperature": 0.0, "max_tokens": 2048}

            retry_outputs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                retry_outputs = list(
                    executor.map(
                        lambda msg: fetch_llm_response(msg, retry_params),
                        retry_messages,
                    )
                )

            # Parse the retried outputs and merge them into rows
            for idx, retry_output in zip(retry_indices, retry_outputs):
                # Prepend "{" back if the backend stripped it out
                full_retry_output = retry_output if retry_output.startswith("{") else "{" + retry_output
                parsed, response = parse_response(full_retry_output)
                row = make_row(idx, chunk_df, parsed, response)
                rows.append(row)

            for j, output in enumerate(retry_outputs):
                i = retry_indices[j]
                # Prepend the "{" we used as prefix
                raw_text = "{" + output.outputs[0].text
                parsed, response = parse_response(raw_text)
                rows.append(make_row(i, chunk_df, parsed, response))

            retry_ok = sum(1 for j in range(len(retry_indices))
                          if rows[-(len(retry_indices)-j)]["parse_status"] != "parse_failed")
            print(f"[Rank ] Retry recovered {retry_ok}/{len(retry_indices)}")

        result_chunk = pd.DataFrame(rows)

        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)

        # Quick stats
        stats = pd.DataFrame(rows)
        status_counts = stats["parse_status"].value_counts()
        print(f"[Rank ] Status: {dict(status_counts)}")
        print(f"[Rank ] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank ] Done. Structured {total_processed} reports -> {output_file}")
