from glob import glob
import os
from pathlib import Path

batch_folders = glob('P:\\Robert\\objects_counting_dmap\\batch*')
for f in batch_folders:
    error_subdir = os.path.join(f, 'egg_base_dir_errors')
    if os.path.isdir(error_subdir):
        print('Found folder', error_subdir, 'and renamed it')
        os.rename(error_subdir, os.path.join(f, 'task1-uli-errors'))
        