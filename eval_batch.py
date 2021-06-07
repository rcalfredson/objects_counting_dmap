import argparse
from glob import glob
import os
from pathlib import Path
import platform
import shutil
import subprocess

parser = argparse.ArgumentParser(
    description="Wrapper script for" " running the eval script for multiple experiments"
)
parser.add_argument(
    "eval_params",
    help="Options to pass the eval script (" "note: enclose them in quotation marks)",
)
parser.add_argument(
    "exp_list",
    help="Path to newline-separated file of"
    " experiment directories (can be relative to the current path). Note:"
    " nets to be measured are expected to be in a sub-folder named"
    " complete_nets.",
)
parser.add_argument(
    "result_dir",
    help="Name of sub-folder inside the"
    " experiment folder where the results will be copied at the end of"
    " running the evaluation script (folder will be created if necessary)",
)

opts = parser.parse_args()
with open(opts.exp_list, "r") as f:
    experiments = f.read().splitlines()

for exp in experiments:
    command = f"python eval.py -m \"{os.path.join(exp, 'complete_nets', '*.pth')}\"" +\
        f" {opts.eval_params}"
    subprocess.call(command, shell=True)
    results_source_dir = f"error_results_{platform.node()}"
    # results_files = os.listdir(results_source_dir)
    results_files = glob(os.path.join(results_source_dir, '*.json'))
    dest_dir = os.path.join(exp, opts.result_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for file_name in results_files:
        shutil.move(file_name, dest_dir)
    
