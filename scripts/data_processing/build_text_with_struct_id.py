#!/usr/bin/env python
"""
Build text parquet with struct_id from UniProtKB JSON data.
Output format matches original parquet + struct_id for structure linking.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def extract_protein_name(protein_desc: dict) -> tuple[str, list[str], list[str]]:
    """Extract protein name, short names, and EC numbers."""
    full_name = ""
    short_names = []
    ec_numbers = []

    if not protein_desc:
        return full_name, short_names, ec_numbers

    rec_name = protein_desc.get("recommendedName", {})
    if rec_name:
        full_name = rec_name.get("fullName", {}).get("value", "")
        short_names = [s.get("value", "") for s in rec_name.get("shortNames", [])]
        ec_numbers = [e.get("value", "") for e in rec_name.get("ecNumbers", [])]

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

    return result


def extract_features(features: list) -> dict:
    """Extract key features with positions."""
    result = {"domains": []}

    for feat in features:
        ftype = feat.get("type", "")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        desc = feat.get("description", "")

        if ftype == "Domain" and desc:
            result["domains"].append((desc, start, end))

    return result


def extract_keywords(keywords: list) -> list[str]:
    """Extract keyword names."""
    return [kw.get("name", "") for kw in keywords if kw.get("name")]


def extract_pdb_ids(cross_refs: list) -> list[str]:
    """Extract PDB IDs from cross references."""
    pdb_ids = []
    for ref in cross_refs:
        if ref.get("database") == "PDB":
            pdb_id = ref.get("id", "").lower()
            if pdb_id:
                pdb_ids.append(pdb_id)
    return pdb_ids


def build_description(entry: dict) -> str:
    """Build a natural language description from entry fields."""
    parts = []

    # 1. Basic info
    accession = entry.get("primaryAccession", "")
    protein_desc = entry.get("proteinDescription", {})
    full_name, short_names, ec_numbers = extract_protein_name(protein_desc)

    organism = entry.get("organism", {})
    org_name = organism.get("scientificName", "")

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

    if short_names:
        parts.append(f"It is also known as {', '.join(short_names)}.")
    if ec_numbers:
        parts.append(f"EC number: {', '.join(ec_numbers)}.")

    # 2. Comments
    comments = extract_comments(entry.get("comments", []))

    if comments["function"]:
        func_text = " ".join(comments["function"])
        parts.append(f"Function: {func_text}")

    if comments["catalytic_activity"]:
        parts.append(f"Catalytic activity: {'; '.join(comments['catalytic_activity'])}.")

    if comments["subcellular_location"]:
        parts.append(f"Subcellular location: {', '.join(comments['subcellular_location'])}.")

    if comments["subunit"]:
        parts.append(f"Subunit: {' '.join(comments['subunit'])}")

    if comments["domain"]:
        parts.append(f"Domain: {' '.join(comments['domain'])}")

    if comments["cofactor"]:
        parts.append(f"Cofactor: {', '.join(comments['cofactor'])}.")

    if comments["pathway"]:
        parts.append(f"Pathway: {'; '.join(comments['pathway'])}.")

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

    # 5. Sequence info
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
        f.read(12)  # Skip '{"results":['

        while True:
            depth = 0
            buffer = ''
            found_start = False

            while True:
                char = f.read(1)
                if not char:
                    return

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
                except json.JSONDecodeError:
                    continue


def count_entries(file_path: str) -> int:
    """Count total entries in JSON file."""
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            count += line.count('"primaryAccession"')
    return count


def main():
    parser = argparse.ArgumentParser(description='Build text parquet with struct_id')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output parquet file path')
    parser.add_argument('--split', type=str, default='text/swissprot', help='Split name')
    parser.add_argument('--limit', type=int, default=0, help='Limit entries (0 for all)')
    parser.add_argument('--struct_mapping', type=str, default='',
                        help='Optional: parquet file with available struct_ids to filter')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Split: {args.split}")

    # Load available struct_ids if provided
    available_struct_ids = None
    if args.struct_mapping:
        print(f"Loading struct mapping from {args.struct_mapping}...")
        df_struct = pd.read_parquet(args.struct_mapping)
        # Assume pdb_name column contains struct_ids
        available_struct_ids = set(df_struct['pdb_name'].str.lower().tolist())
        print(f"Loaded {len(available_struct_ids)} available struct_ids")

    # Count entries
    print("Counting entries...")
    total = count_entries(str(input_path))
    print(f"Total entries: {total}")

    # Process entries
    records = []
    matched_count = 0

    print("Processing entries...")
    for i, entry in enumerate(tqdm(parse_json_streaming(str(input_path)), total=total)):
        if args.limit > 0 and i >= args.limit:
            break

        accession = entry.get("primaryAccession", "")

        # Determine struct_id
        # Priority 1: AlphaFold ID (AF-{accession}-F1-model_v4)
        # Priority 2: PDB cross-reference
        struct_id = ""
        struct_source = ""

        # Try AlphaFold
        af_id = f"AF-{accession}-F1-model_v4"
        if available_struct_ids is None or af_id.lower() in available_struct_ids:
            struct_id = af_id
            struct_source = "afdb"

        # Try PDB if no AlphaFold match
        if not struct_id:
            cross_refs = entry.get("uniProtKBCrossReferences", [])
            pdb_ids = extract_pdb_ids(cross_refs)
            for pdb_id in pdb_ids:
                # Format: pdb_id (we'll need chain info later)
                if available_struct_ids is None or pdb_id in available_struct_ids:
                    struct_id = pdb_id
                    struct_source = "pdb"
                    break

        if struct_id:
            matched_count += 1

        # Build description
        text = build_description(entry)

        # Get sequence info
        seq_info = entry.get("sequence", {})
        sequence = seq_info.get("value", "")
        seq_length = seq_info.get("length", len(sequence))

        record = {
            "split": args.split,
            "pdb_name": accession,  # UniProt accession as identifier
            "plddt": 0.0,
            "text": text,
            "seq_length": seq_length,
            "struct_length": 0,  # To be filled after struct mapping
            "struct_id": struct_id,  # Reference to PDB/AlphaFold
            "struct_source": struct_source,  # "pdb" or "afdb"
        }
        records.append(record)

    print(f"Processed {len(records)} entries")
    print(f"Matched with struct_id: {matched_count} ({matched_count/len(records)*100:.1f}%)")

    # Create DataFrame and save
    df = pd.DataFrame(records)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Stats
    print(f"\nStruct source distribution:")
    print(df['struct_source'].value_counts())

    # Save
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show samples
    print("\n" + "="*60)
    print("Sample records:")
    print("="*60)

    # Show one with afdb and one with pdb
    for source in ['afdb', 'pdb', '']:
        sample = df[df['struct_source'] == source].head(1)
        if len(sample) > 0:
            print(f"\n--- Source: {source if source else 'no match'} ---")
            row = sample.iloc[0]
            print(f"pdb_name: {row['pdb_name']}")
            print(f"struct_id: {row['struct_id']}")
            print(f"struct_source: {row['struct_source']}")
            print(f"seq_length: {row['seq_length']}")
            text = row['text']
            print(f"text: {text[:300]}..." if len(text) > 300 else f"text: {text}")


if __name__ == "__main__":
    main()
