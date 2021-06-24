import argparse
from collections import Counter
from ntpath import join
import dabest
from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

sys.path.append(os.path.abspath("./"))
from plotutil.sinaplot import sinaplot


from lib.error_manager import ErrorManager
from lib.statistics import meanConfInt, ttest_ind


parser = argparse.ArgumentParser(
    description="Calculate accuracy of subsets of nets using bootstrap method."
)

parser.add_argument(
    "exp_list",
    help="Path to newline-separated file of"
    " experiment directories (can be relative to the current path). Note:"
    " nets to be measured are expected to be in a sub-folder named"
    " complete_nets.",
)

parser.add_argument(
    "error_subdirs",
    nargs="*",
    help="Space-separated list of sub-folders from which to load precalculated errors"
    " which get combined are are then used in ranking the nets.",
)

parser.add_argument(
    "results_subdir",
    help="Sub-folder inside experiment directories where the script should"
    " seek out JSON results files",
)

parser.add_argument("n", help="Number of sample sets to generate", type=int)

opts = parser.parse_args()

print(f"Opts values: {opts.exp_list}, {opts.error_subdirs}, {opts.results_subdir}, {opts.n}")


def copy_figure(fig):
    return pickle.loads(pickle.dumps(fig, pickle.HIGHEST_PROTOCOL))


results_files = []
net_names = {}
with open(opts.exp_list, "r") as f:
    experiments = f.read().splitlines()
for exp in experiments:
    outlier_path = os.path.join(exp, "outliers.txt")
    if os.path.isfile(outlier_path):
        with open(outlier_path) as f:
            outliers = f.read().splitlines()
    else:
        outliers = None
    possible_nets = glob(os.path.join(exp, opts.results_subdir, "*.json"))
    for net in possible_nets:
        net_names[net] = (
            os.path.normpath(net)
            .split(os.sep)[-1]
            .split("errors_")[1]
            .split(".json")[0]
        )
        # comparing bootstrapped distributions
        if outliers and f"error_results_{net_names[net]}" in outliers:
            print("skipping outlier", net_names[net])
            continue
        results_files.append(net)

n_nets = len(results_files)
errors = {
    "top_3": {
        "abs": {"mean": [], "max": []},
        "rel": {"mean": [], "mean0to10": [], "mean11to40": [], "mean41plus": []},
    },
    "top_50_pct": {
        "abs": {"mean": [], "max": []},
        "rel": {"mean": [], "mean0to10": [], "mean11to40": [], "mean41plus": []},
    },
}
ground_truth_counts = {}
error_keys = (
    ("abs", "mean"),
    ("abs", "max"),
    ("rel", "mean"),
    ("rel", "mean0to10"),
    ("rel", "mean11to40"),
    ("rel", "mean41plus"),
)
error_titles = (
    "mean absolute error",
    "max absolute error",
    "mean relative error",
    "mean relative error, 0-10 eggs",
    "mean relative error, 11-40 eggs",
    "mean relative error, 41+ eggs",
)
for i in range(opts.n * 2):
    sample = random.choices(results_files, k=n_nets)
    net_name_to_file = {}
    counter = Counter(sample)
    # print(counter.most_common(6))
    print("how many samples:", len(sample))
    # input()
    weighted_errors_by_net = {}
    for net in sample:
        split_path = os.path.abspath(net).split(os.sep)
        net_name = net_names[net]
        # print("net name:", net_name)
        if net_name in weighted_errors_by_net:
            continue
        if net_name not in net_name_to_file:
            net_name_to_file[net_name] = net
        weighted_errors_for_net = []
        for j, err_dir in enumerate(opts.error_subdirs):
            if j == 0:
                split_path[0] += '\\'
            error_record = os.path.join(*split_path[:-2], err_dir, f"error_results_{net_name}.txt")
            assert os.path.isfile(
                error_record
            ), f"Could not find error record for {error_record}"
        # how to find the best nets?
        # need to calculate the weighted error for each one.
            err_manager = ErrorManager([os.path.join(split_path[0], err_dir)], analyze_by_wt_avg=True)
            err_manager.process_single_error(0, error_record, None)
            weighted_errors_for_net.append(list(err_manager.avgErrsByNet[0].values())[0])
        # print("net name:", net_name)
        weighted_errors_by_net[net_name] = np.mean(weighted_errors_for_net)
    sorted_nets = dict(sorted(weighted_errors_by_net.items(), key=lambda item: item[1]))
    # print("sorted nets:", sorted_nets)
    sorted_net_ks = list(sorted_nets.keys())
    if i % 2 == 0:
        group = sorted_net_ks[:]#int(len(sorted_net_ks) / 2):]
        key = "top_50_pct"
    else:
        group = sorted_net_ks[:3]
        key = "top_3"
    # input()
    # tally stats for the top three nets first.

    predictions = {}
    for i, res_file in enumerate([net_name_to_file[net] for net in group]):
        with open(os.path.abspath(res_file)) as f:
            results = json.load(f)
            for k in results:
                if len(ground_truth_counts) < len(results):
                    ground_truth_counts[k] = results[k]["true"]
                if k in predictions:
                    predictions[k].append(results[k]["predicted"])
                else:
                    predictions[k] = [results[k]["predicted"]]
    abs_errors = []
    rel_errors = []
    rel_errors_0to10 = []
    rel_errors_11to40 = []
    rel_errors_41plus = []
    for k in predictions:
        predictions[k] = np.mean(predictions[k])
        abs_errors.append(abs(predictions[k] - ground_truth_counts[k]))
        rel_error = abs_errors[-1] / ground_truth_counts[k]
        if rel_error == np.infty:
            continue
        rel_errors.append(rel_error)
        if ground_truth_counts[k] < 11:
            rel_errors_0to10.append(rel_error)
        elif ground_truth_counts[k] < 41:
            rel_errors_11to40.append(rel_error)
        else:
            rel_errors_41plus.append(rel_error)
    errors[key]["abs"]["mean"].append(np.mean(abs_errors))
    errors[key]["abs"]["max"].append(max(abs_errors))
    errors[key]["rel"]["mean"].append(np.mean(rel_errors))
    errors[key]["rel"]["mean0to10"].append(np.mean(rel_errors_0to10))
    errors[key]["rel"]["mean11to40"].append(np.mean(rel_errors_11to40))
    errors[key]["rel"]["mean41plus"].append(np.mean(rel_errors_41plus))
    # print("at end, here are errors:", errors)
# create a single CSV file
def write_stats_rows(f, key):
    for i, val in enumerate(errors[key]["abs"]["mean"]):
        f.write(
            f",{val:.3f},"
            f"{errors[key]['abs']['max'][i]:.3f},"
            f"{errors[key]['rel']['mean'][i]:.3f},"
            f"{errors[key]['rel']['mean0to10'][i]:.3f},"
            f"{errors[key]['rel']['mean11to40'][i]:.3f},"
            f"{errors[key]['rel']['mean41plus'][i]:.3f}\n"
        )
    # still need to write the confidence interval

    f.write("means:,")
    for key1, key2 in error_keys:
        mci = meanConfInt(errors[key][key1][key2], asDelta=True)
        f.write(f"{mci[0]:.3f} +/- {mci[1]:.3f},")
    f.write("\n")
    f.write("standard errors:,")
    for key1, key2 in error_keys:
        data = np.asarray(errors[key][key1][key2])
        mean = np.nanmean(data)
        std_err = np.sqrt(
            (1 / (len(data) - 1)) * np.nansum(np.power(np.subtract(data, mean), 2))
        )
        f.write(f"{std_err:.3f},")
    f.write("\n")


# need to convert to a pandas DataFrame
# first, need to choose one type of error, and then stack the observations
# from across the two methods.
for i, keys in enumerate(error_keys):
    key1, key2 = keys
    top_3_errs = errors["top_3"][key1][key2]
    top_50_errs = errors["top_50_pct"][key1][key2]
    stacked_errs = np.hstack((top_3_errs, top_50_errs))
    print("stacked_errs:", stacked_errs)
    err_df = pd.DataFrame(
        {
            "Error": stacked_errs,
            "Group": ["Top 3 nets"] * len(top_3_errs)
            + ["All nets"] * len(top_50_errs),
        },
        columns=["Group", "Error"],
    )
    print("dataframe?\n", err_df)
    dabest_df = pd.DataFrame(
        np.asarray(
            [
                err_df["Error"][: int(len(err_df["Error"]) / 2)],
                err_df["Error"][int(len(err_df["Error"]) / 2) :],
            ]
        ).T,
        columns=["Top 3 nets", "All nets"],
    )
    print("dabest df?", dabest_df)
    print(
        "data inputs:",
        [
            err_df["Error"][: int(len(err_df["Error"]) / 2)],
            err_df["Error"][int(len(err_df["Error"]) / 2) :],
        ],
    )
    print("end")

    error_dabest = dabest.load(dabest_df, idx=("Top 3 nets", "All nets"))
    fig = error_dabest.mean_diff.plot(fig_size=(9, 6))
    mean_diff_dist = plt.subplot(132)
    # print("err dist", mean_diff_dist)
    plt.subplot(131)

    plt.suptitle(f"Difference in {error_titles[i]}")
    # fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    # plt.axes(ax[0])
    sinaplot(x="Group", y="Error", data=err_df)
    fig.delaxes(fig.axes[1])
    fig.axes[0].change_geometry(1, 2, 2)
    fig.axes[1].change_geometry(1, 2, 1)
    # plt.tight_layout()
    # fig = plt.subplot(121)
    # plt.axes(ax[1])
    # copy_figure(mean_diff_dist)
    plt.show()


with open("bootstrap_results.csv", "w") as f:
    f.write(",mean abs,max abs,mean rel,mean rel 0-10,mean rel 11-40,mean rel 41+\n")
    f.write("top 3\n")
    write_stats_rows(f, "top_3")
    f.write("top 50%\n")
    write_stats_rows(f, "top_50_pct")
