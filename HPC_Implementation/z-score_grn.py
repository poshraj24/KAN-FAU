#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Iterable


def _load_grn_table(input_file: str) -> pd.DataFrame:
    """
    Load a GRN table with three columns: regulator, target, importance.


    Returns
    -------
    DataFrame with columns: ['regulator', 'target', 'importance'].
    """
    p = Path(input_file)

    preferred_sep = "\t" if p.suffix.lower() == ".tsv" else None

    try:
        df = pd.read_csv(input_file, sep=preferred_sep, header=0, engine="python")
    except Exception:
        # Fallback 1: try tab
        try:
            df = pd.read_csv(input_file, sep="\t", header=0)
        except Exception:
            # Fallback 2: try comma
            df = pd.read_csv(input_file, sep=",", header=0)

    if df.shape[1] == 1:
        col0 = df.columns[0]
        # Replace literal backslash-t with actual tab for both header and cells
        header_parts = col0.replace("\\t", "\t").split("\t")
        split_cols = (
            df.iloc[:, 0]
            .astype(str)
            .str.replace("\\t", "\t", regex=False)
            .str.split("\t", expand=True)
        )

        if split_cols.shape[1] >= 3:
            split_cols = split_cols.iloc[:, :3]
            # If header had 3 parts, use them; else use default names
            if len(header_parts) >= 3 and all(h.strip() for h in header_parts[:3]):
                split_cols.columns = [h.strip() for h in header_parts[:3]]
            else:
                split_cols.columns = ["regulator", "target", "importance"]
            df = split_cols

    # Validate we have at least 3 columns
    if df.shape[1] < 3:
        raise ValueError(
            f"Could not parse 3 columns from {input_file}. "
            f"Columns found: {df.columns.tolist()}"
        )

    # Standardize to expected names (first 3 columns are used)
    df = df.rename(
        columns={
            df.columns[0]: "regulator",
            df.columns[1]: "target",
            df.columns[2]: "importance",
        }
    )
    return df[["regulator", "target", "importance"]]


def filter_grn_by_zscore(
    input_file: str,
    output_prefix: Optional[str] = None,
    cutoffs: Iterable[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    two_sided: bool = False,
) -> List[Dict]:
    """
    Filter GRN edges by per-regulator (local) z-scores.

    Steps:
      Load regulator, target, importance
     Compute per-regulator mean and population std (ddof=0)
      Drop zero-variance regulators (no ranking possible)
      Compute z = (x - mean)/std



    Returns
    -------
    List of dict summaries (one per cutoff).
    """
    input_path = Path(input_file)
    print(f"Reading: {input_file}")
    df = _load_grn_table(input_file)

    # Importance to numeric
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    n_bad = df["importance"].isna().sum()
    if n_bad:
        print(f"Dropping {n_bad} rows with non-numeric importance.")
        df = df.dropna(subset=["importance"])

    g = df.groupby("regulator")["importance"]
    reg_mean = g.mean().rename("reg_mean")
    reg_std = g.apply(lambda x: x.std(ddof=0)).rename("reg_std")
    stats = pd.concat([reg_mean, reg_std], axis=1).reset_index()

    # Treat zero std as NaN and drop those rows (no variation)
    stats["reg_std"].replace(0, np.nan, inplace=True)

    df_s = df.merge(stats, on="regulator", how="left")
    n_zero_var = df_s["reg_std"].isna().sum()
    if n_zero_var:
        print(
            f"Dropping {n_zero_var} rows from regulators with zero variance in importance."
        )
    df_s = df_s.dropna(subset=["reg_std"]).copy()

    # z-score
    df_s["z_score"] = (df_s["importance"] - df_s["reg_mean"]) / df_s["reg_std"]

    if output_prefix is None:
        out_dir = input_path.parent
        base = input_path.stem + "_filtered"
    else:
        p = Path(output_prefix)
        out_dir = p.parent if str(p.parent) not in ("", ".") else input_path.parent
        base = p.name if p.name else (input_path.stem + "_filtered")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFiltering and saving to: {out_dir}")
    print(f"{'Cutoff':<8} {'Edges':<8} {'Regs':<6} {'Targets':<8} Filename")

    results: List[Dict] = []
    for c in cutoffs:
        if two_sided:
            mask = df_s["z_score"].abs() >= c
            tag = f"abs_{c}"
        else:
            mask = df_s["z_score"] >= c
            tag = f"{c}"

        keep = (
            df_s.loc[mask, ["regulator", "target", "importance", "z_score"]]
            .sort_values(["regulator", "importance"], ascending=[True, False])
            .reset_index(drop=True)
        )

        out_file = out_dir / f"{base}_zscore_{tag}.csv"
        keep.to_csv(out_file, index=False)

        summary = {
            "cutoff": c,
            "interactions": len(keep),
            "regulators": keep["regulator"].nunique(),
            "targets": keep["target"].nunique(),
            "filename": str(out_file),
        }
        results.append(summary)
        print(
            f"{c:<8} {summary['interactions']:<8} {summary['regulators']:<6} "
            f"{summary['targets']:<8} {out_file.name}"
        )

    # Summary file
    summ_path = out_dir / f"{base}_summary.txt"
    with open(summ_path, "w") as f:
        f.write("GRN Z-score Filtering Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Original interactions: {len(df)}\n")
        f.write(f"Original regulators: {df['regulator'].nunique()}\n")
        f.write(f"Original targets: {df['target'].nunique()}\n\n")
        f.write(f"Two-sided: {two_sided}\n\n")
        f.write(f"{'Cutoff':<8}{'Edges':<10}{'Regs':<8}{'Targets':<10}\n")
        f.write("- *".replace("*", "") * 40 + "\n")  # simple separator
        for r in results:
            f.write(
                f"{r['cutoff']:<8}{r['interactions']:<10}{r['regulators']:<8}"
                f"{r['targets']:<10}\n"
            )

    print(f"\nSummary saved to: {summ_path}")
    return results


def main():

    input_file = "KAN_Implementation/Data/grnboost2_1139.tsv"
    cutoffs = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    two_sided = False
    # ====================================

    if not Path(input_file).exists():
        print(f"Input not found: {input_file}")
        print(f"cwd: {os.getcwd()}")
        return

    print(f"Found GRN file: {Path(input_file).name}")
    print(f"File size: {os.path.getsize(input_file)} bytes")
    print("=" * 50)

    output_prefix = Path(input_file).stem + "_filtered"
    try:
        filter_grn_by_zscore(
            input_file,
            output_prefix=output_prefix,
            cutoffs=cutoffs,
            two_sided=two_sided,
        )
        print("Successfully processed file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
