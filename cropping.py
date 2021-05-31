from cv2 import cv2
import numpy as np
import os

def res(images_dir, masks_dir,flag):
    dirs = {
        1:"test_",
        2:"train_",
        3:"val_"
    }
    ids = os.listdir(images_dir)
    mask_ids = os.listdir(masks_dir)
    images_fps = [os.path.join(images_dir, image_id) for image_id in ids]
    masks_fps = [os.path.join(masks_dir, image_id) for image_id in mask_ids]
    for i in range(len(images_fps)):
        im = cv2.imread(images_fps[i])
        im2 = cv2.imread(masks_fps[i])
        #print(im2,masks_fps[i])
        resized1 = cv2.resize(im,(512,512))
        """cv2.imshow("Resized image", resized1)
        cv2.waitKey(0)"""
        resized2 = cv2.resize(im2,(512,512))
        cv2.imwrite(f"cropped_data/{dirs[flag]}data/{i}.jpg",resized1)
        cv2.imwrite(f"cropped_data/{dirs[flag]}mask/{i}.png",resized2)
        print(i)

res("cropped_data/val_data_full","cropped_data/val_mask_full",3)