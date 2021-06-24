import cv2
from glob import glob
import numpy as np
import os
from PIL import Image
import platform
import sys
import torch
sys.path.append(os.path.abspath("./"))

from model import FCRN_A
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = FCRN_A(input_filters=3, N=2).to(device)
network.train(False)
sys_os = platform.system()

if sys_os == 'Windows':
    model_path = (r"P:\Robert\objects_counting_dmap_experiments"
    r"\batch56-fcrn-small-dataset-pixel-based-loss-redo-lr-0.0025"
    r"\complete_nets\egg-fullsize-pt-presample-compare-2021-03-23"
    "_FCRN_A_Yang-Lab-Dell3_2021-06-08 00-25-29.900685_300epochs.pth")
    img_folder = 'P:/Robert/objects_counting_dmap/egg_source/combined_robert_uli_temp'
elif sys_os == 'Linux':
    model_path = '/media/Synology3/Robert/objects_counting_dmap_experiments/batch56-fcrn-small-dataset-pixel-based-loss-redo-lr-0.0025/complete_nets/egg-fullsize-pt-presample-compare-2021-03-23' +\
    "_FCRN_A_Yang-Lab-Dell3_2021-06-08 00-25-29.900685_300epochs.pth"
    img_folder = '/media/Synology3/Robert/objects_counting_dmap/egg_source/combined_robert_uli_temp'
img_paths = glob(os.path.join(img_folder, '*.jpg'))
network = torch.nn.DataParallel(network)
network.load_state_dict(torch.load(model_path))
img_paths = [os.path.join(img_folder, '2020-12-02_img_0007_0_1_left_6A9RT.jpg')]
for pth in img_paths:
    for y_cropping in range(300):
        img_orig = Image.open(pth)
        half_crop = y_cropping / 2
        img_orig = img_orig.crop((0, np.floor(half_crop), img_orig.width, img_orig.height -np.ceil( half_crop)))
        img = torch.from_numpy(
                    (1 / 255) * np.expand_dims(np.moveaxis(np.array(img_orig, dtype=np.float32), 2, 0), 0)
                )

        img = img.to(device, dtype=torch.float)
        # print('some values of img:')
        # print(img)
        # print(img.shape)
        # print('and type:', type(img))
        # input()
        result = network(img)
        dmap_to_show = result.cpu().detach().numpy()[0].T
        img_orig = np.array(img_orig)
        img_orig = img_orig[:, :, ::-1].copy()
        cv2.imshow('input', img_orig)
        cv2.imshow('output', dmap_to_show)
        print(f'Viewing {os.path.basename(pth)}')
        print('Image shape:', img_orig.shape)
        print('Total Y cropping on either side:', y_cropping)
        cv2.waitKey(0)
