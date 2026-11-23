# Sourced by the job scripts. Override CMP_ENV, CMP_DATA_DIR or CMP_V1_RESULTS if your folders differ.
CMP_ENV="${CMP_ENV:-$HOME/envs/cross-market}"
if [ ! -x "$CMP_ENV/bin/python" ]; then
    echo "ERROR: no Python environment at $CMP_ENV (see README.md, or set CMP_ENV)." >&2
    exit 1
fi
export PATH="$CMP_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

# Data: this folder's data/, or the data folder of the Version 1 clone next to it
if [ -z "${CMP_DATA_DIR:-}" ]; then
    CMP_DATA_DIR="$PWD/data"
    [ -f "$CMP_DATA_DIR/USA.parquet" ] || CMP_DATA_DIR="$PWD/../MachineLearningProject/data"
fi
export CMP_DATA_DIR

# Version 1 results: read only, for the final comparison
if [ -z "${CMP_V1_RESULTS:-}" ]; then
    CMP_V1_RESULTS="$PWD/../MachineLearningProject/results"
    [ -d "$CMP_V1_RESULTS/evaluation" ] || CMP_V1_RESULTS="$PWD/results"
fi
export CMP_V1_RESULTS
