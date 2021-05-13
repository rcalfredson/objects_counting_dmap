import argparse
from glob import glob
import json
import os
import sys

import numpy as np
from lib.error_manager import ErrorManager
from lib.statistics import ttest_rel, ttest_ind

parser = argparse.ArgumentParser(
    description="Run t-test comparison of results from" " two FCRN learning experiments"
)
parser.add_argument("dir1", help="Folder containing first set of error results")
parser.add_argument("dir2", help="Folder containing second set of error results")
parser.add_argument(
    "--paired_name_map",
    help="JSON file mapping names of original nets to their paired counterparts as"
    " part of a dependent t-test for paired samples. The keys of the dictionary"
    " should correspond to nets from dir1, and the values to nets from dir2.",
)
opts = parser.parse_args()


if opts.paired_name_map:
    with open(opts.paired_name_map) as f:
        paired_name_map = json.load(f)


dirs_to_check = [opts.dir1, opts.dir2]
err_manager = ErrorManager(dirs_to_check, opts.paired_name_map)
for i, dir_name in enumerate(dirs_to_check):
    err_manager.parse_error_for_dir(dir_name, i)

# print('errors for first group:')
# for k in errors[0]:
#     print(k)
#     print(f'{len(errors[0][k])} entries')
#     for entry in sorted(errors[0][k]):
#         print(entry)
# input()
print('before continuing, errors:', err_manager.errors)
for err_cat in err_manager.error_categories:
    print("Error category:", err_cat)
    res = ttest_ind(
        np.asarray(err_manager.errors[0][err_cat]),
        np.asarray(err_manager.errors[1][err_cat]),
        msg=True,
    )
    print()