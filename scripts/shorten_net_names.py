import json
import os
from pathlib import Path
import shutil

dirpath = (
    r"P:\Robert\objects_counting_dmap\batch30-fcrn-small-dataset-mae-loss"
    r"-function\complete nets"
)

paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
paths = [os.path.basename(p) for p in paths if 'pth' in str(p)]
ordered_paths = {i: p for i, p in enumerate([os.path.basename(p) for p in paths])}
print("ordered paths:", ordered_paths)
with open(os.path.join(dirpath, "name_map.json"), "w") as f:
    json.dump(ordered_paths, f, ensure_ascii=False, indent=4)
for p in ordered_paths:
    shutil.copy(
        os.path.join(dirpath, ordered_paths[p]),
        os.path.join(dirpath, f"{p}.pth"),
    )
