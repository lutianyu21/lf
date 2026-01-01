#!/usr/bin/env python
"""
Build text-only parquet from UniProtKB JSON data.
Extracts key fields and combines them into descriptive text.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def extract_protein_name(protein_desc: dict) -> tuple[str, str, list[str]]:
    """Extract protein name, short name, and EC numbers."""
    full_name = ""
    short_names = []
    ec_numbers = []

    if not protein_desc:
        return full_name, short_names, ec_numbers

    # Recommended name
    rec_name = protein_desc.get("recommendedName", {})
    if rec_name:
        full_name = rec_name.get("fullName", {}).get("value", "")
        short_names = [s.get("value", "") for s in rec_name.get("shortNames", [])]
        ec_numbers = [e.get("value", "") for e in rec_name.get("ecNumbers", [])]

    # Fallback to submitted name
    if not full_name:
        sub_names = protein_desc.get("submissionNames", [])
        if sub_names:
            full_name = sub_names[0].get("fullName", {}).get("value", "")

    return full_name, short_names, ec_numbers


def extract_comments(comments: list) -> dict:
    """Extract key information from comments."""
    result = {
        "function": [],
        "subcellular_location": [],
        "catalytic_activity": [],
        "subunit": [],
        "domain": [],
        "cofactor": [],
        "pathway": [],
        "similarity": [],
        "ptm": [],
    }

    for comment in comments:
        ctype = comment.get("commentType", "")

        if ctype == "FUNCTION":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["function"].append(t["value"])

        elif ctype == "SUBCELLULAR LOCATION":
            locs = comment.get("subcellularLocations", [])
            for loc in locs:
                loc_val = loc.get("location", {}).get("value", "")
                if loc_val:
                    result["subcellular_location"].append(loc_val)

        elif ctype == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            if reaction.get("name"):
                result["catalytic_activity"].append(reaction["name"])

        elif ctype == "SUBUNIT":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["subunit"].append(t["value"])

        elif ctype == "DOMAIN":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["domain"].append(t["value"])

        elif ctype == "COFACTOR":
            cofactors = comment.get("cofactors", [])
            for cof in cofactors:
                if cof.get("name"):
                    result["cofactor"].append(cof["name"])

        elif ctype == "PATHWAY":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["pathway"].append(t["value"])

        elif ctype == "SIMILARITY":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["similarity"].append(t["value"])

        elif ctype == "PTM":
            texts = comment.get("texts", [])
            for t in texts:
                if t.get("value"):
                    result["ptm"].append(t["value"])

    return result


def extract_features(features: list) -> dict:
    """Extract key features with positions."""
    result = {
        "domains": [],  # (name, start, end)
        "active_sites": [],
        "binding_sites": [],
    }

    for feat in features:
        ftype = feat.get("type", "")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        desc = feat.get("description", "")

        if ftype == "Domain" and desc:
            result["domains"].append((desc, start, end))
        elif ftype == "Active site":
            result["active_sites"].append(start)
        elif ftype == "Binding site" and desc:
            result["binding_sites"].append((desc, start))

    return result


def extract_keywords(keywords: list) -> list[str]:
    """Extract keyword names."""
    return [kw.get("name", "") for kw in keywords if kw.get("name")]


def build_description(entry: dict) -> str:
    """Build a natural language description from entry fields."""
    parts = []

    # 1. Basic info: name, organism
    accession = entry.get("primaryAccession", "")
    protein_desc = entry.get("proteinDescription", {})
    full_name, short_names, ec_numbers = extract_protein_name(protein_desc)

    organism = entry.get("organism", {})
    org_name = organism.get("scientificName", "")

    # Opening sentence
    if full_name:
        opening = f"This protein ({accession}) is {full_name}"
        if org_name:
            opening += f" from {org_name}"
        opening += "."
        parts.append(opening)
    elif accession:
        opening = f"This protein ({accession})"
        if org_name:
            opening += f" is from {org_name}"
        opening += "."
        parts.append(opening)

    # Short names and EC numbers
    if short_names:
        parts.append(f"It is also known as {', '.join(short_names)}.")
    if ec_numbers:
        parts.append(f"EC number: {', '.join(ec_numbers)}.")

    # 2. Comments
    comments = extract_comments(entry.get("comments", []))

    # Function
    if comments["function"]:
        func_text = " ".join(comments["function"])
        parts.append(f"Function: {func_text}")

    # Catalytic activity
    if comments["catalytic_activity"]:
        parts.append(f"Catalytic activity: {'; '.join(comments['catalytic_activity'])}.")

    # Subcellular location
    if comments["subcellular_location"]:
        parts.append(f"Subcellular location: {', '.join(comments['subcellular_location'])}.")

    # Subunit
    if comments["subunit"]:
        parts.append(f"Subunit: {' '.join(comments['subunit'])}")

    # Domain info from comments
    if comments["domain"]:
        parts.append(f"Domain: {' '.join(comments['domain'])}")

    # Cofactor
    if comments["cofactor"]:
        parts.append(f"Cofactor: {', '.join(comments['cofactor'])}.")

    # Pathway
    if comments["pathway"]:
        parts.append(f"Pathway: {'; '.join(comments['pathway'])}.")

    # Similarity
    if comments["similarity"]:
        parts.append(f"{' '.join(comments['similarity'])}")

    # 3. Features
    features = extract_features(entry.get("features", []))

    if features["domains"]:
        domain_strs = []
        for name, start, end in features["domains"]:
            if start and end:
                domain_strs.append(f"{name} (residues {start}-{end})")
            else:
                domain_strs.append(name)
        parts.append(f"Contains domains: {', '.join(domain_strs)}.")

    # 4. Keywords
    keywords = extract_keywords(entry.get("keywords", []))
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}.")

    # 5. Sequence
    seq_info = entry.get("sequence", {})
    sequence = seq_info.get("value", "")
    seq_len = seq_info.get("length", len(sequence))

    if sequence:
        parts.append(f"The protein has {seq_len} amino acids.")
        # 每个氨基酸用空格隔开
        spaced_sequence = " ".join(list(sequence))
        parts.append(f"<seq>{spaced_sequence}</seq>")

    return " ".join(parts)


def parse_json_streaming(file_path: str):
    """Parse large JSON file in streaming fashion."""
    with open(file_path, 'r') as f:
        # Skip '{"results":['
        f.read(12)

        while True:
            depth = 0
            buffer = ''
            found_start = False

            while True:
                char = f.read(1)
                if not char:
                    return  # EOF

                if char == '{':
                    found_start = True
                    depth += 1
                    buffer += char
                elif found_start:
                    buffer += char
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            break

            if buffer:
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue


def count_entries(file_path: str) -> int:
    """Count total entries in JSON file (approximate by counting '{"entryType"')."""
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            count += line.count('"primaryAccession"')
    return count


def main():
    parser = argparse.ArgumentParser(description='Build text parquet from UniProtKB JSON')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output parquet file path')
    parser.add_argument('--split', type=str, default='text/swissprot', help='Split name for the dataset')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of entries (0 for all)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Split: {args.split}")

    # Count entries first (approximate)
    print("Counting entries...")
    total = count_entries(str(input_path))
    print(f"Approximate total entries: {total}")

    # Process entries
    records = []

    print("Processing entries...")
    for i, entry in enumerate(tqdm(parse_json_streaming(str(input_path)), total=total)):
        if args.limit > 0 and i >= args.limit:
            break

        # Build description text
        text = build_description(entry)

        # Get sequence info
        seq_info = entry.get("sequence", {})
        sequence = seq_info.get("value", "")
        seq_length = seq_info.get("length", len(sequence))

        record = {
            "split": args.split,
            "pdb_name": entry.get("primaryAccession", ""),
            "plddt": 0.0,
            "text": text,
            "seq_length": seq_length,
            "struct_length": 0,  # No structure for text-only data
        }
        records.append(record)

    print(f"Processed {len(records)} entries")

    # Create DataFrame and save
    df = pd.DataFrame(records)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Show sample
    print("\n" + "="*60)
    print("Sample records:")
    print("="*60)
    for i in range(min(2, len(df))):
        print(f"\n--- Record {i} ---")
        print(f"pdb_name: {df.iloc[i]['pdb_name']}")
        print(f"seq_length: {df.iloc[i]['seq_length']}")
        text = df.iloc[i]['text']
        if len(text) > 500:
            print(f"text: {text[:500]}...")
        else:
            print(f"text: {text}")


if __name__ == "__main__":
    main()
