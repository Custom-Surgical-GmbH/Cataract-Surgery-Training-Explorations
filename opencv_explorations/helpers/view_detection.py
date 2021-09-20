
import cv2
import numpy as np
from .limbus_detection import detect_circle

def get_view_mask(grey, return_circle=False):
    mask = np.zeros(grey.shape, dtype=np.uint8)
    circle = detect_circle(grey, validation='inout', validation_mode='min')
    cv2.circle(mask, (round(circle[0]), round(circle[1])), round(circle[2]), 255, cv2.FILLED)

    if return_circle:
        return mask, circle
    else:
        return mask
