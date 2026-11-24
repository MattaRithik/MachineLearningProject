"""Load functions and constants from the notebook without running it (no data, training or GPU)."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "Cross_Market_Project.ipynb"


def load_names(names, namespace):
    """Define the requested top-level names in `namespace`, in notebook order, from the notebook's own source."""
    wanted, found = set(names), set()
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined = {node.name}
            elif isinstance(node, ast.Assign):
                defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            else:
                continue
            if defined & wanted:
                exec(compile(ast.Module([node], []), "notebook", "exec"), namespace)
                found |= defined
    if wanted - found:
        raise NameError(f"Not found in the notebook: {sorted(wanted - found)}")
    return namespace


REPORT_NAMES = [
    "SEED", "DECILE", "BLOCK_MONTHS", "BOOTSTRAP_SAMPLES", "COST_BPS", "AUDIT_BOOTSTRAP", "AUDIT_CANDIDATES", "BASE_SPEC",
    "V2_TRANSFORMER_STRATEGIES", "VARIANTS", "BASE_BASELINE_SPEC", "BASELINE_VARIANTS", "ALL_VARIANTS", "LOCAL_COUNTRIES",
    "TRAIN_WINDOWS", "save_safely", "save_audit", "sharpe_ratio", "block_indices", "bootstrap_difference", "holm_adjust",
    "compounded_drawdown", "load_validation_rows", "validation_zero_mse", "validation_cells", "selection_table",
    "seed_aggregates", "aggregation_sensitivity", "audit_block_indices", "load_monthly", "existing_test_summary",
    "paired_ic_tests", "transfer_pairs", "portfolio_sensitivity", "run_reports", "file_sha256", "tree_manifest",
    "feature_hash", "verify_fingerprints",
]
