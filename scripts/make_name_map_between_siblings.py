import json

map_paths = [
    r"P:\Robert\objects_counting_dmap\batch35-fcrn-small-dataset-mae-loss-300epochs-from-batch30-smooth-lr\complete_nets\name_map.json",
    r"P:\Robert\objects_counting_dmap\batch30-fcrn-small-dataset-mae-loss-function\complete_nets\name_map2.json",
]
maps = {}
for i, p in enumerate(map_paths):
    with open(p, "r") as f:
        maps[i] = json.load(f)

inv_map = {v: k for k, v in maps[1].items()}

new_map = {}
for k in maps[0]:
    new_map[k] = inv_map[maps[0][k]]

print("mapping:", new_map)
with open(
    r"P:\Robert\objects_counting_dmap\batch35-fcrn-small-dataset-mae-loss-300epochs-from-batch30-smooth-lr\sib_to_sib_map.json",
    "w",
) as f:
    json.dump(new_map, f, ensure_ascii=False, indent=4)