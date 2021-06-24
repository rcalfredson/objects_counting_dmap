from glob import glob
import os


class ErrorManager:
    def __init__(
        self,
        error_dirs: list,
        paired_name_map=None,
        analyze_by_wt_avg=False,
        best_by_min_cat=False,
        model_list=None
    ):
        self.error_categories = (
            "mean absolute error",
            "mean relative error",
            "mean relative error, 0-10 eggs",
            "mean relative error, 11-40 eggs",
            "mean relative error, 41+ eggs",
            "maximum error",
        )
        self.error_dirs = error_dirs
        self.analyze_by_wt_avg = analyze_by_wt_avg
        self.best_by_min_cat = best_by_min_cat
        if analyze_by_wt_avg:
            self.error_weights = (3, 2, 4, 1, 1, 4)
            self.wt_sum = sum(self.error_weights)
            self.error_weights = [w / self.wt_sum for w in self.error_weights]
        self.n_experiments = len(error_dirs)
        self.model_list = model_list
        if best_by_min_cat:
            self.best_nets = [{} for _ in range(self.n_experiments)]
            self.net_names = [[] for _ in range(self.n_experiments)]
        self.errors = [
            {k: [] for k in self.error_categories} for _ in range(self.n_experiments)
        ]
        self.paired_name_map = paired_name_map
        if self.paired_name_map:
            self.pairing_order = []
        self.avgErrsByNet = [{} for _ in range(self.n_experiments)]

    def process_single_error(self, i, error_filename, outliers_for_dir):
        if outliers_for_dir and os.path.basename(error_filename) in outliers_for_dir:
            return
        with open(os.path.abspath(error_filename), "r") as f:
            single_net_errors = f.read().splitlines()
            if self.analyze_by_wt_avg:
                self.categorized_errors = {k: 0 for k in self.error_categories}
            for err_line in single_net_errors:
                err_type = err_line.split(":")[0]
                if "(" in err_line:
                    parsed_error = float(err_line.split("(")[-1].split(")")[0])
                else:
                    parsed_error = float(err_line.split(":")[-1])
                self.update_error_structures(i, err_type, parsed_error)
            if self.analyze_by_wt_avg or self.best_by_min_cat:
                net_id = (
                    os.path.basename(error_filename)
                    .split("error_results_")[1]
                    .split(".pth.txt")[0]
                )
            if self.model_list and net_id not in self.model_list:
                return
            if self.analyze_by_wt_avg:
                self.avgErrsByNet[i][net_id] = self.calc_weighted_avg_error()
            if self.best_by_min_cat:
                self.net_names[i].append(net_id)

    def find_best_nets_for_exp(self, i):
        for cat in self.error_categories:
            index_of_min = self.errors[i][cat].index(min(self.errors[i][cat]))
            net_name = self.net_names[i][index_of_min]
            if net_name in self.best_nets[i]:
                self.best_nets[i][net_name]['cats'][cat] = self.errors[i][cat][index_of_min]
            else:
                self.best_nets[i][net_name] = {'cats':{cat: self.errors[i][cat][index_of_min]}}
            if self.analyze_by_wt_avg:
                self.best_nets[i][net_name]['wt_avg'] = self.avgErrsByNet[i][net_name]

    def update_error_structures(self, i, err_type, parsed_error):
        self.errors[i][err_type].append(parsed_error)
        if self.analyze_by_wt_avg:
            self.categorized_errors[err_type] = parsed_error

    def calc_weighted_avg_error(self):
        ret_val = sum(
            [
                self.categorized_errors[cat] * self.error_weights[i]
                for i, cat in enumerate(self.error_categories)
            ]
        )
        return ret_val

    def parse_error_for_dir(self, dir_name, index):
        error_files = glob(os.path.join(dir_name, "error_results*"))
        outlier_filename = os.path.join(dir_name, "outliers.txt")
        if not os.path.isfile(outlier_filename):
            outlier_filename = os.path.join(dir_name, '../outliers.txt')
        if os.path.isfile(outlier_filename):
            with open(outlier_filename, "r") as f:
                outliers_for_dir = ["%s.txt" % line for line in f.read().splitlines()]
        else:
            outliers_for_dir = None
        if not self.paired_name_map or self.paired_name_map and index == 0:
            for error_file in error_files:
                self.process_single_error(index, error_file, outliers_for_dir)
                if self.paired_name_map:
                    self.pairing_order.append(
                        error_file.split("error_results_")[1].split(".txt")[0]
                    )
        elif self.paired_name_map and index > 0:
            for net_name in self.pairing_order:
                error_file = "error_results_%s.txt" % (self.paired_name_map[net_name])
                self.process_single_error(
                    index, os.path.join(dir_name, error_file), outliers_for_dir
                )
        if self.best_by_min_cat:
            self.find_best_nets_for_exp(index)