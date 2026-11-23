"""Check the Python environment; run inside a SLURM job, not on a login node."""
import platform

import IPython
import ipykernel
import joblib
import matplotlib
import nbclient
import nbformat
import numpy
import openpyxl
import pandas
import pyarrow
import scipy
import sklearn
import torch
import xgboost

print(f"python {platform.python_version()} on {platform.platform()}")
for package in (numpy, pandas, pyarrow, scipy, sklearn, xgboost, torch, matplotlib, joblib, openpyxl, IPython,
                ipykernel, nbclient, nbformat):
    print(f"{package.__name__} {package.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} | PyTorch CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}, {properties.total_memory / 1e9:.1f} GB")
