from glob import glob
import os
from pathlib import Path
import shutil

valid_files = [os.path.basename(p) for p in glob(r'P:\Robert\objects_counting_dmap\batch29-fcrn-small-dataset-gaussian-no-rot\error_results*')]
for f in glob(r'P:\Robert\objects_counting_dmap\batch29-fcrn-small-dataset-gaussian-no-rot\task1-uli-results\errors_*'):
    net_name = f.split('errors_')[1].split('.json')[0]
    if f"error_results_{net_name}.txt" not in valid_files:
        shutil.move(f, os.path.join(Path(f).parent, 'unused', os.path.basename(f)))