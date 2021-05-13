from glob import glob
import os
from pathlib import Path
import shutil

valid_files = [os.path.basename(p) for p in glob(r'P:\Robert\objects_counting_dmap\batch29-fcrn-small-dataset-gaussian-no-rot\error_results*')]
for f in glob(r'P:\Robert\objects_counting_dmap\batch29-fcrn-small-dataset-gaussian-no-rot\egg_base_dir_errors\error_results*'):
    if os.path.basename(f) not in valid_files:
        shutil.move(f, os.path.join(Path(f).parent, 'unused', os.path.basename(f)))