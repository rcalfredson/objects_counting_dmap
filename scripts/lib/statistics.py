import numpy as np
import os
import random
import scipy.stats as st
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import p2stars

WELCH = True

# returns t, p, na, nb
def ttest_rel(a, b, msg=None, min_n=2):
    return ttest(a, b, True, msg, min_n)


def ttest_ind(a, b, msg=None, min_n=2):
    return ttest(a, b, False, msg, min_n)


def ttest(a, b, paired, msg=None, min_n=2):
    if paired:
        abFinite = np.isfinite(a) & np.isfinite(b)
    a, b = (x[abFinite if paired else np.isfinite(x)] for x in (a, b))
    na, nb = len(a), len(b)
    if min(na, nb) < min_n:
        return np.nan, np.nan, na, nb
    with np.errstate(all="ignore"):
        t, p = st.ttest_rel(a, b) if paired else st.ttest_ind(a, b, equal_var=not WELCH)
    if msg:
        print("%spaired t-test -- %s:" % ("" if paired else "un", msg))
        print(
            "  n = %s means: %.3g, %.3g; t-test: p = %.5f, t = %.3f"
            % (
                "%d," % na if paired else "%d, %d;" % (na, nb),
                np.mean(a),
                np.mean(b),
                p,
                t,
            )
        )
        print("copyable output:")
        print("means: %.3g, %.3g" % (np.mean(a), np.mean(b)))
        print("p: %.5f; %s" % (p, p2stars(p)))
    return t, p, na, nb


# a = np.array(
#     [
#       1.412,
# 1.919,#
# 1.812,#
# 1.913,#
# 1.49,#
# 1.659,#
# 3.09,#
# 1.436,#
# 1.422,#
# 3.471,#
# 1.423,#
# 1.374,#
# 2.227,#
# 2.287,#
# 1.469,#
# 2.124,#
# 1.315,#
# 1.985,#

#     ]
# )

# b = np.array(
#     [
#   1.978,
# 1.516,
# 1.622,
# 1.638,
# 1.65,
# 2.148,
# 1.823,
# 1.693,
# 2.7,
# 1.577,
# 1.316,
# 1.742,
# 2.225,
# 1.853,

#     ]
# )

# print("Quick test of ttest_rel")
# list1 = [1.712, 1.505, 1.188, 1.451, 1.609, 2.035]
# list2 = [1.823, 2.056, 1.714, 2.056, 1.527, 1.435, 1.834, 1.838, 1.334, 2.360, 2.017, 2.016]
# print(ttest_rel(np.asarray(list1), np.asarray(list2)))
# print(ttest_ind(np.asarray(list1), np.asarray(list2), msg=True))
# random.shuffle(list1)
# random.shuffle(list2)
# print("new list2:", list2)
# print(ttest_rel(np.asarray(list1), np.asarray(list2)))
# print(ttest_ind(np.asarray(list1), np.asarray(list2)))

# res = ttest_ind(a, b, msg=True)
# print("  p: %.3f; %s" % (res[1], p2stars(res[1])))
