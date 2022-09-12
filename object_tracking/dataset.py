import glob
import os

import cv2
import numpy as np


class VOT14Reader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.all_samples_path = glob.glob(os.path.join(self.dataset_path, "*/"))

    def __len__(self):
        return len(self.all_dataset_path)

    def __getitem__(self, idx):
        all_images_path = glob.glob(os.path.join(self.all_samples_path[idx], "images", "*"))
        all_images_path = sorted(all_images_path)   # sorted by file name index
        gt_path = os.path.join(self.all_samples_path[idx], "annotation", "groundtruth.txt")

        # read gt_path
        with open(gt_path, "r") as f:
            groundtruth = f.readlines()
            groundtruth = [line.replace('\n', '') for line in groundtruth]

        assert len(groundtruth) == len(all_images_path), "Total images path not equal to total groundtruth!"

        all_images = []
        all_images_name = []
        for image_path in all_images_path:
            image = cv2.imread(image_path)
            all_images.append(image)
            all_images_name.append(image_path)

        groundtruth = [np.array([float(pts) for pts in polygon.split(",")], dtype=np.int32).reshape(-1, 1, 2) \
                            for polygon in groundtruth]

        return all_images, groundtruth, all_images_name

