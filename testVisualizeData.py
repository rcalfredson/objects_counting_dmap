import cv2
import torch

from data_loader import H5Dataset, MultiDimH5Dataset, padImagesBasedOnMaxSize

dataset = H5Dataset(r"P:\Robert\objects_counting_dmap\eggTemp-multiple_folders\splineDistCompare.h5")
dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=4)

for images, labels in dataloader:
    for image, label in zip(images, labels):
        cv2.imshow('imageOrig', image.cpu().numpy()[0].T)
        cv2.imshow('labels', 255*label.cpu().detach().numpy()[0].T)
        cv2.waitKey(0)