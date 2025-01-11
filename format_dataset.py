import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def convert_images_to_png(path):
    if not os.path.exists(path):
        print("Path doesn't exists")
        return
    
    images = os.listdir(path)

    for _, img in enumerate(tqdm(images, desc="Converting images")):
        if img.lower().endswith(('.jpeg', '.jpg')):
            image_path = os.path.join(path, img)
            image = cv2.imread(image_path)

            base_filename = os.path.splitext(img)[0]
            new_filename = f"{base_filename}.png"
            new_path = os.path.join(path, new_filename)

            cv2.imwrite(new_path, image)

            os.remove(image_path)


def get_images(path) -> np.array:
    images = os.listdir(path)
    array_images = []
    for _, img in enumerate(tqdm(images, desc="Getting images")):
        image = cv2.imread(os.path.join(path, img))
        image = cv2.resize(image, (32, 32))
        array_images.append(image.flatten())
    return array_images

def get_metrics(test_label, predicted_label):
    metrics = f"accuracy_score={accuracy_score(test_label, predicted_label)}\n"\
    f"f1_score={f1_score(test_label, predicted_label)}\n"\
    f"precision={precision_score(test_label, predicted_label)}\n"\
    f"recall={recall_score(test_label, predicted_label)}"
    return metrics