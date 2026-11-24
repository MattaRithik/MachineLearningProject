"""Run the project notebook without a browser and stream its output to the job log.

The notebook kernel inherits the CMP_* environment variables, which its settings cell reads before any other work.
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path("Cross_Market_Project.ipynb")


class StreamingClient(NotebookClient):
    """Print text output as soon as the kernel produces it."""

    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        elif msg["msg_type"] in ("display_data", "execute_result"):
            print(msg["content"]["data"].get("text/plain", ""), flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    experiment = sys.argv[1]
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    epochs = int(os.environ.get("CMP_EPOCHS", 100))
    if os.environ.get("CMP_AUDIT") == "1":  # never inside a Version 2 results folder
        output_dir = Path(os.environ.get("CMP_AUDIT_DIR", "audit_v2")) / "private" / "executed"
    else:
        output_dir = Path("results_v2" if epochs == 100 else f"results_v2_{epochs}epochs") / "executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{experiment}_{os.environ.get('SLURM_JOB_ID', 'local')}.ipynb"

    client = StreamingClient(notebook, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}})
    try:
        client.execute()  # raises an error, and so fails the job, if any cell fails
    finally:
        nbformat.write(notebook, output)  # keeps a copy with outputs; the source notebook is unchanged
        print(f"\nExecuted notebook: {output}", flush=True)


if __name__ == "__main__":
    main()
