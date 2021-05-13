from glob import glob
import json
import os

map_path = (
    r"P:\Robert\objects_counting_dmap"
    r"\batch30-fcrn-small-dataset-mae-loss-function\complete_nets\name_map.json"
)

with open(map_path) as f:
    map = json.load(f)

new_map = {}
new_nets = glob(
    r"P:\Robert\objects_counting_dmap\batch35-fcrn-small-dataset-mae-loss-300epochs-from-batch30-smooth-lr\complete_nets\*.pth"
)
for k in map:
    for i, net_name in enumerate(new_nets):
        if f'retrain_{k}__' in net_name:
            print('found', k, 'in net name', net_name)
            break
    print('i:', i)
    new_map[os.path.basename(new_nets[i])] = map[k]
print("new map:", new_map)

with open(r'P:\Robert\objects_counting_dmap\batch35-fcrn-small-dataset-mae-loss-300epochs-from-batch30-smooth-lr\complete_nets\name_map.json', 'w') as f:
    json.dump(new_map, f, ensure_ascii=False, indent=4)