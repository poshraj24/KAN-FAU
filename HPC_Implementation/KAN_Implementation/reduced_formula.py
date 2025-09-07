from pathlib import Path
import re
import pandas as pd
import sympy as sp

# ---------------------- CONSTANTS----------------------
BASE_MODELS_DIR = Path(
    "KAN_Implementation/kan_models"
)  # folder containing <GeneX>/symbolic_formula.txt
GRN_FILES = [
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_0.5.csv"),
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_1.0.csv"),
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_1.5.csv"),
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_2.0.csv"),
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_2.5.csv"),
    Path("KAN_Implementation/Data/kan_1139_original_filtered_zscore_3.0.csv"),
]
OUTPUT_ROOT = Path("KAN_Implementation/kan_models_reduced")
# -----------------------------------------------------------------------

GENE_PATTERN = re.compile(r"Gene_\d+")


def folder_to_target_symbol(folder_name: str) -> str:
    """
    Convert a gene folder name to the target symbol used in formulas/GRNs.

    """
    m = re.search(r"(\d+)$", folder_name)
    if not m:

        if GENE_PATTERN.fullmatch(folder_name):
            return folder_name
        raise ValueError(f"Cannot infer target id from folder name: {folder_name}")
    return f"Gene_{int(m.group(1))}"


def list_gene_folders(base_dir: Path) -> list[Path]:
    return [p for p in base_dir.iterdir() if p.is_dir()]


def load_symbolic_formula(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_gene_symbols(expr_text: str) -> set[str]:
    return set(GENE_PATTERN.findall(expr_text))


def parse_expr(expr_text: str, symbol_names: set[str]) -> sp.Expr:
    # Create sympy Symbols for each Gene_* token
    local_syms = {name: sp.Symbol(name) for name in symbol_names}
    local_syms["exp"] = sp.exp
    return sp.sympify(expr_text, locals=local_syms)


def grn_targets_to_regulators(csv_path: Path) -> dict[str, set[str]]:
    """
    Return mapping: target_symbol -> set(regulator_symbols) for a GRN file.
    Column names are matched case-insensitively.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "regulator" not in cols or "target" not in cols:
        raise ValueError(f"{csv_path} must contain 'regulator' and 'target' columns.")

    reg_col = cols["regulator"]
    tgt_col = cols["target"]

    # Normalize to 'Gene_<id>' if necessary
    def norm_gene_name(x: str) -> str:
        if isinstance(x, str) and GENE_PATTERN.fullmatch(x):
            return x
        # attempt parse trailing digits
        m = re.search(r"(\d+)$", str(x))
        if m:
            return f"Gene_{int(m.group(1))}"
        return str(x)

    df["_reg"] = df[reg_col].map(norm_gene_name)
    df["_tgt"] = df[tgt_col].map(norm_gene_name)

    mapping: dict[str, set[str]] = {}
    for tgt, sub in df.groupby("_tgt"):
        mapping[tgt] = set(sub["_reg"].astype(str).tolist())
    return mapping


def reduce_formula(expr_text: str, allowed_regs: set[str]) -> str:
    """
    Substitute 0 for all Gene_* that are not in allowed_regs.
    Simplify the resulting expression and return as string with 'exp' style.
    """

    present = extract_gene_symbols(expr_text)

    expr = parse_expr(expr_text, present)

    subs = {sp.Symbol(g): 0 for g in present if g not in allowed_regs}
    reduced = sp.simplify(expr.subs(subs))

    return sp.sstr(reduced)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    gene_folders = list_gene_folders(BASE_MODELS_DIR)

    grn_maps = []
    for grn_file in GRN_FILES:
        grn_maps.append((grn_file, grn_targets_to_regulators(grn_file)))

    for gene_dir in gene_folders:
        target_symbol = folder_to_target_symbol(gene_dir.name)
        formula_path = gene_dir / "symbolic_formula.txt"
        if not formula_path.exists():

            continue

        expr_text = load_symbolic_formula(formula_path)

        for grn_file, tgt2regs in grn_maps:
            cutoff_name = grn_file.stem.split("_")[-1]  # e.g., '0.5'
            out_dir = OUTPUT_ROOT / f"cutoff_{cutoff_name}" / gene_dir.name
            ensure_dir(out_dir)

            allowed = tgt2regs.get(target_symbol, set())
            reduced_str = reduce_formula(expr_text, allowed)

            out_path = out_dir / "symbolic_formula_reduced.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(reduced_str + "\n")

    print("Done. Reduced formulas saved under:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
