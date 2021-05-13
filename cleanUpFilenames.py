from glob import glob
from os.path import isfile, exists
from shutil import copy

glob_str = r'P:\Robert\objects_counting_dmap\batch40-fcrn-small-dataset-mae-300epchs-elastic-deform-redo\complete_nets\*'
files = glob(glob_str)
print('files:', files)
for fname in files:
    new_fname = ''.join(fname.split('retrain___'))
    if isfile(fname) and not exists(new_fname):
        copy(fname, new_fname)
