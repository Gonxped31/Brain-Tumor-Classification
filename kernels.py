import numpy as np
import cv2

SHARPEN = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

PREWITT_Y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

LAPLACIAN = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

def apply_kernel(image, kernel):
    return  cv2.filter2D(image, -1, kernel)