from glob import glob
import json

lr_history_search_path = (
    r"P:\Robert\objects_counting_dmap\batch30-fcrn-small-dataset-mae-loss-function"
)

with open(
    r"P:\Robert\objects_counting_dmap\batch30-fcrn-small-dataset-mae-loss-function\complete_nets\name_map.json",
    "r",
) as f:
    name_map = json.load(f)

for k in name_map:
    print(
        "my search path:",
        lr_history_search_path
        + "\\*lr_history*%s" % (name_map[k].split("_")[-1].split(".pth")[0]),
    )
    lr_history_file = glob(
        lr_history_search_path
        + "\\*lr_history*%s" % (name_map[k].split("_")[-1].split(".pth")[0])
    )[0]
    print("for net", k)
    print("history file name:", lr_history_file)
    input()
