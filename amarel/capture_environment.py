"""Write a lock file of the installed packages. Usage: python amarel/capture_environment.py <output file>

It records the environment as it is now, which may differ from the one used for a past run; compare it with the
versions stored in results_v2/<variant>/config.json.
"""
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path


def main():
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}, key=str.lower)
    lines = [f"# Python {platform.python_version()} on {platform.platform()} (libc {' '.join(platform.libc_ver())})"]
    try:
        import torch
        lines.append(f"# PyTorch {torch.__version__}, CUDA build {torch.version.cuda}, available {torch.cuda.is_available()}")
    except ImportError:
        lines.append("# PyTorch is not installed")
    Path(sys.argv[1]).write_text("\n".join(lines + packages) + "\n")
    print(f"{len(packages)} packages written to {sys.argv[1]}")


if __name__ == "__main__":
    main()
