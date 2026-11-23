# Sourced by the job scripts. Override CMP_ENV if the Python environment is elsewhere.
CMP_ENV="${CMP_ENV:-$HOME/envs/cross-market}"
if [ ! -x "$CMP_ENV/bin/python" ]; then
    echo "ERROR: no Python environment at $CMP_ENV (see README.md, or set CMP_ENV)." >&2
    exit 1
fi
export PATH="$CMP_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
