import numpy as np
from util import *

arr1 = np.array([3, 1, 11])
arr2 = np.array([7, 20, 3])
arr3 = np.array([14, 3, 5])
arr4 = np.array([1, 9, 0])
unitedArr = np.vstack((arr1, arr2, arr3, arr4))
print('numpy mean:', np.mean(unitedArr, axis=0))
for i in range(unitedArr.shape[1]):
    print('using meanConfInt:', meanConfInt(unitedArr[:, i], asDelta=True))
print('before zip:', list((meanConfInt(unitedArr[:, i], asDelta=True)[:2] for i in \
    range(unitedArr.shape[1]))))
means, confInts = zip(*[meanConfInt(unitedArr[:, i], asDelta=True)[:2] for i in \
    range(unitedArr.shape[1])])
print('means:', means)
print('confInts:', confInts)
# maxDiffs = np.abs(np.max(unitedArr, axis=0) - np.min(unitedArr, axis=0))