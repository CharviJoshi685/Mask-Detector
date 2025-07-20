# dataset_loader.py – optional helper for loading mask dataset

import os
import cv2
import numpy as np

def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    labels = []
    class_map = {"with_mask": 0, "without_mask": 1}

    for label in ["with_mask", "without_mask"]:
        class_folder = os.path.join(folder_path, label)
        for file in os.listdir(class_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = img.astype("float32") / 255.0
                images.append(img)
                labels.append(class_map[label])

    return np.array(images), np.array(labels)

# Example usage
# X, y = load_images_from_folder("dataset/")
