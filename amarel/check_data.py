"""Check the three Parquet files (rows and columns) without loading them. Usage: python amarel/check_data.py [data folder]"""
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_ROWS = {"USA": 1_604_094, "GBR": 534_295, "JPN": 1_175_157}
EXPECTED_COLUMNS = 165
REQUIRED = {"id", "eom", "excntry", "size_grp", "me", "ret_exc_lead1m", "ret_1_0", "common", "exch_main",
            "primary_sec", "obs_main"}


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CMP_DATA_DIR", "data"))
    reference, failed = None, False
    for country, rows in EXPECTED_ROWS.items():
        path = folder / f"{country}.parquet"
        if not path.exists():
            print(f"{country}: MISSING ({path})")
            failed = True
            continue
        file = pq.ParquetFile(path)
        schema = file.schema_arrow
        reference = reference or schema
        problems = []
        if file.metadata.num_rows != rows:
            problems.append(f"{file.metadata.num_rows:,} rows, expected {rows:,}")
        if len(schema.names) != EXPECTED_COLUMNS:
            problems.append(f"{len(schema.names)} columns, expected {EXPECTED_COLUMNS}")
        if not REQUIRED <= set(schema.names):
            problems.append(f"missing columns {sorted(REQUIRED - set(schema.names))}")
        if not schema.equals(reference):
            problems.append("column names or types differ from the first file")
        print(f"{country}: {'; '.join(problems) or 'OK'}")
        failed = failed or bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
