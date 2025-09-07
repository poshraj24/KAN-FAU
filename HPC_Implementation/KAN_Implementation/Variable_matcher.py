#!/usr/bin/env python3
"""
Script to:
1. Replace 'NKX6-3' with 'NKX6_3' in all JSON files
2. Replace 'zoo' terms with '0' in formulas (zoo is an error term)
across multiple gene folders
"""

import os
import json
import re
from pathlib import Path


def clean_formula(formula_str):
    """
    Clean formula by replacing zoo terms with 0 and fixing formatting
    """
    if not isinstance(formula_str, str):
        return formula_str

    # Replace zoo terms with 0 (common patterns)
    # zoo can appear as: zoo, zoo*, *zoo, zoo(something), etc.
    zoo_patterns = [
        (r"\bzoo\*[^+\-]*", "0"),  # zoo* followed by anything until +/- -> 0
        (r"[^+\-]*\*zoo\b", "0"),  # anything*zoo -> 0
        (r"\bzoo\([^)]*\)", "0"),  # zoo(anything) -> 0
        (r"\bzoo\b", "0"),  # standalone "zoo" -> 0
    ]

    cleaned = formula_str
    for pattern, replacement in zoo_patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Clean up extra spaces and formatting
    cleaned = re.sub(r"\s+", " ", cleaned)  # Multiple spaces to single
    cleaned = re.sub(r"\+\s*\+", "+", cleaned)  # ++ to +
    cleaned = re.sub(r"-\s*-", "+", cleaned)  # -- to +
    cleaned = re.sub(r"\+\s*-", "-", cleaned)  # +- to -
    cleaned = re.sub(r"-\s*\+", "-", cleaned)  # -+ to -
    cleaned = re.sub(r"^\s*[+\-]\s*", "", cleaned)  # Remove leading +/-
    cleaned = re.sub(r"\s*[+\-]\s*$", "", cleaned)  # Remove trailing +/-
    cleaned = cleaned.strip()

    return cleaned


def process_json_content(content):
    """
    Process JSON content to replace NKX6-3 and clean formulas
    """
    replacements_made = {"nkx6": 0, "zoo": 0}

    # Replace NKX6-3 with NKX6_3
    if "NKX6-3" in content:
        count = content.count("NKX6-3")
        content = content.replace("NKX6-3", "NKX6_3")
        replacements_made["nkx6"] = count

    # Parse JSON to clean formulas
    try:
        data = json.loads(content)

        def clean_json_recursively(obj):
            zoo_count = 0
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and ("zoo" in value.lower()):
                        original = value
                        cleaned = clean_formula(value)
                        if cleaned != original:
                            obj[key] = cleaned
                            zoo_count += value.lower().count("zoo")
                    elif isinstance(value, (dict, list)):
                        zoo_count += clean_json_recursively(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str) and ("zoo" in item.lower()):
                        original = item
                        cleaned = clean_formula(item)
                        if cleaned != original:
                            obj[i] = cleaned
                            zoo_count += item.lower().count("zoo")
                    elif isinstance(item, (dict, list)):
                        zoo_count += clean_json_recursively(item)
            return zoo_count

        zoo_count = clean_json_recursively(data)
        replacements_made["zoo"] = zoo_count

        # Convert back to JSON string
        content = json.dumps(data, indent=2, ensure_ascii=False)

    except json.JSONDecodeError:
        # If it's not valid JSON, just do string replacement for zoo
        if "zoo" in content.lower():
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "zoo" in line.lower():
                    original_line = line
                    cleaned_line = clean_formula(line)
                    if cleaned_line != original_line:
                        lines[i] = cleaned_line
                        replacements_made["zoo"] += original_line.lower().count("zoo")
            content = "\n".join(lines)

    return content, replacements_made


def replace_and_clean_file(file_path):
    """
    Replace NKX6-3 and remove zoo terms in a single file
    Special handling: symbolic_formula.txt only gets zoo replacement, not NKX6-3 replacement
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if this is a symbolic_formula.txt file
        is_symbolic_txt = file_path.name == "symbolic_formula.txt"

        if is_symbolic_txt:
            # For symbolic_formula.txt, only replace zoo terms
            if "zoo" in content.lower():
                updated_content = clean_formula(content)
                zoo_count = content.lower().count("zoo")
                replacements = {"nkx6": 0, "zoo": zoo_count}
            else:
                replacements = {"nkx6": 0, "zoo": 0}
                updated_content = content
        else:
            # For all other files, process normally (both NKX6-3 and zoo)
            updated_content, replacements = process_json_content(content)

        # Only write if changes were made
        if replacements["nkx6"] > 0 or replacements["zoo"] > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            changes = []
            if replacements["nkx6"] > 0:
                changes.append(f"NKX6-3→NKX6_3: {replacements['nkx6']}")
            if replacements["zoo"] > 0:
                changes.append(f"zoo→0: {replacements['zoo']}")

            print(f" Fixed {', '.join(changes)} in: {file_path.name}")
            return replacements
        else:
            print(f"No changes needed in: {file_path.name}")
            return {"nkx6": 0, "zoo": 0}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"nkx6": 0, "zoo": 0}


def find_and_process_files(root_directory="."):
    """
    Find all files in gene folders and process them
    """
    # Look for the correct path structure
    kan_models_path = Path(root_directory) / "KAN_Implementation" / "kan_models"

    if not kan_models_path.exists():
        print(f"Error: Directory '{kan_models_path}' not found!")

        return

    print(f"Searching for files in: {kan_models_path.absolute()}")
    print("-" * 80)

    files_found = 0
    files_modified = 0
    total_nkx6_replacements = 0
    total_zoo_replacements = 0

    # Walk through all gene folders in kan_models
    for gene_folder in kan_models_path.iterdir():
        if gene_folder.is_dir():
            print(f"\nProcessing folder: {gene_folder.name}")

            # Look for common files in each gene folder (JSON and TXT)
            target_files = [
                "test_genes.json",
                "related_genes.json",
                "results.json",
                "symbolic_formula.json",
                "symbolic_formula.txt",
            ]

            folder_had_files = False
            for filename in target_files:
                target_file = gene_folder / filename

                if target_file.exists():
                    files_found += 1
                    folder_had_files = True

                    replacements = replace_and_clean_file(target_file)
                    if replacements["nkx6"] > 0 or replacements["zoo"] > 0:
                        files_modified += 1
                        total_nkx6_replacements += replacements["nkx6"]
                        total_zoo_replacements += replacements["zoo"]

            if not folder_had_files:
                print(f"No target files found in {gene_folder.name}")

    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Files found: {files_found}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total NKX6-3 → NKX6_3 replacements: {total_nkx6_replacements}")
    print(f"  Total 'zoo' → '0' replacements: {total_zoo_replacements}")

    if files_found == 0:
        print(f"\nNo target files found in {kan_models_path}")


def main():

    root_directory = "."  # Current directory

    response = input(
        f"\nThis will update JSON and TXT files in 'KAN_Implementation/kan_models/' subdirectories.\nContinue? (y/N): "
    )

    if response.lower() not in ["y", "yes"]:
        print("Operation cancelled.")
        return

    find_and_process_files(root_directory)
    print("\nOperation completed!")


if __name__ == "__main__":
    main()
