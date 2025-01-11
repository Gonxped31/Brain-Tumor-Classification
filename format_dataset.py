import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from kernels import (
    SHARPEN,
    SOBEL_X,
    SOBEL_Y,
    PREWITT_X,
    PREWITT_Y,
    LAPLACIAN,
    apply_kernel
)

import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

def preprocess_image(image, combo) -> np.array:
    for func in combo:
        image = func(image)
    return image

def get_images(path, combo):
    images_names = os.listdir(path)
    images = []
    # for _, img in enumerate(tqdm(images_names, desc="Getting images")):
    for img in images_names:
        image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32))
        image = preprocess_image(image, combo)
        image = image.flatten()
        images.append(image)
    return images

def train_and_evaluate(train_images, train_labels, test_images, test_labels):
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(train_images, train_labels)
    predictions = model.predict(test_images)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    return accuracy, precision, recall, f1, model


def best_filter_search(yes_images_path, no_images_path):
    kernels = {
        'Sharpen': lambda img: apply_kernel(img, SHARPEN),
        'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        'Sobel X': lambda img: apply_kernel(img, SOBEL_X),
        'Sobel Y': lambda img: apply_kernel(img, SOBEL_Y),
        'Prewitt X': lambda img: apply_kernel(img, PREWITT_X),
        'Prewitt Y': lambda img: apply_kernel(img, PREWITT_Y),
        'Laplacian': lambda img: apply_kernel(img, LAPLACIAN)
    }

    best_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }

    best_combo = None

    best_model = None

    for _, r in enumerate(tqdm(range(1, len(kernels) + 1), desc="Applying kernels")):
        for combo in itertools.combinations(kernels.values(), r):
            yes_images = get_images(yes_images_path, combo)
            no_images = get_images(no_images_path, combo)
            labels = np.array([1]*len(yes_images) + [0]*len(no_images))
            train_images = np.array(yes_images + no_images).astype('float32')

            train_sample, test_sample, train_label, test_label = train_test_split(train_images, labels, test_size=0.2, random_state=0)

            accuracy, precision, recall, f1, model = train_and_evaluate(train_sample, train_label, test_sample, test_label)
            if best_combo != None or best_model != None:
                diff = int((accuracy > best_metrics['accuracy'])) + int((precision > best_metrics['precision'])) +\
                    int((recall > best_metrics['recall'])) + int((f1 > best_metrics['f1']))
            else:
                diff = 4

            if diff >= 3:
                best_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                best_combo = combo
                
                best_model = model

    best_combo_list = [name for name, func in kernels.items() if func in best_combo]

    return best_metrics, best_combo_list, model
            
