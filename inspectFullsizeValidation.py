import cv2
import h5py

h5Test = h5py.File('eggTemp-fullsize/valid.h5', 'r')
print('items:', len(list(h5Test)))
print(list(h5Test))
# print('num patches:', len(h5Test['images']))
for k in h5Test:
    if "_dots" in k: continue
    cv2.imshow('debug1', h5Test[k][0].T)
    cv2.imshow('debug2', h5Test["%s_dots"%k][0].T)
    cv2.waitKey(0)
# print(h5Test['jul10_6right_0_1_WWO07'][0])
# h5Test = h5py.File('egg-fullsize/train.h5', 'r')
# print('items:', list(h5Test))
# print(h5Test['jul14_11left_0_3_KLB08'][0])
# h5Test = h5py.File('egg-eval/valid.h5', 'r')
# print(h5Test['jul14_11left_1_0_DSAZ8'][0])
