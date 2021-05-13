import argparse
import calendar
from enum import Enum, auto
from glob import glob
import json
import os

class filename_styles(Enum):
    old = auto()
    new = auto()

from util import meanConfInt

p = argparse.ArgumentParser(
    description="Create HTML table of results from" " a learning experiment."
)
p.add_argument(
    "error_dir", help="Path to folder containing error files in JSON format."
)
opts = p.parse_args()
error_files = glob(os.path.join(opts.error_dir, "error_results*"))
outlier_filename = os.path.join(opts.error_dir, "outliers.txt")
if os.path.basename(opts.error_dir).startswith('batch'):
    if int(opts.error_dir.split('batch')[1].split('-')[0]) > 33:
        namestyle = filename_styles.new
    else:
        namestyle = filename_styles.old
else:
    namestyle = filename_styles.new
if os.path.isfile(outlier_filename):
    with open(outlier_filename, "r") as f:
        outliers_for_dir = ["%s.txt" % line for line in f.read().splitlines()]

headers = (
    "name",
    "mean abs. error",
    "max abs. error",
    "mean rel. error",
    "mean rel. error, 0-10 eggs",
    "mean rel. error, 11-40 eggs",
    "mean rel. error, 41+ eggs",
)
error_type_keys = (
    "mean absolute error",
    "maximum error",
    "mean relative error",
    "mean relative error, 0-10 eggs",
    "mean relative error, 11-40 eggs",
    "mean relative error, 41+ eggs",
)
months = [m.lower() for m in calendar.month_abbr[1:]]


def generate_name_str(filename):
    num_epochs = filename.split("epochs")[-2].split("_")[-1]
    date_split_idx = -2 if namestyle == filename_styles.new else -1
    date, time = filename.split("_")[date_split_idx].split(" ")
    split_date = date.split("-")
    date_num = split_date[-1]
    month = months[int(split_date[1]) - 1]
    year = split_date[0]
    date_str = f"{date_num}-{month}-{year}"
    host = filename.split("_")[-3].split("Yang-Lab-")[-1]
    return f"{num_epochs} epochs, {date_str}, {':'.join(time.split('-')[:2])}, {host}"


errors_by_type = {k: [] for k in error_type_keys}

with open("error_table.html", "w") as f:
    f.write('<table style="width: 100%;" border="0">\n')
    f.write("<tbody><tr>")
    for header in headers:
        f.write(f"<td><strong>{header}</strong></td>\n")
    f.write("<tr>\n")
    for fpath in error_files:
        f.write("<tr>")
        f.write(f"<td>{generate_name_str(fpath)}</td>\n")
        with open(fpath) as f_errs:
            tds = {k: "" for k in error_type_keys}
            errs = f_errs.read().splitlines()
            for err_line in errs:
                err_type = err_line.split(":")[0]
                if "(" in err_line:
                    error_val = float(err_line.split("(")[-1].split(")")[0])
                else:
                    error_val = float(err_line.split(":")[-1])
                if "outliers_for_dir" not in locals() or (
                    "outliers_for_dir" in locals()
                    and os.path.basename(fpath) not in outliers_for_dir
                ):
                    errors_by_type[err_type].append(error_val)
                tds[err_type] = f"<td>{error_val:.3f}</td>\n"
            for k in error_type_keys:
                f.write(tds[k])
        f.write("</tr>")
    f.write("</tbody>\n")
    f.write("</table>")

print("Means of errors by type:")
for k in errors_by_type:
    print(f"{k}:")
    print(meanConfInt(errors_by_type[k], asDelta=True))
