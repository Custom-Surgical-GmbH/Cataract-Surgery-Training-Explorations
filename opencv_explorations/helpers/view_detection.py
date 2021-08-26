
import cv2
import numpy as np
from .limbus_detection import detect_limbus

def get_view_mask(grey):
    mask = np.zeros(grey.shape, dtype=np.byte)
    circle = detect_limbus(grey, validation='inout', validation_mode='min')
    cv2.circle(mask, (round(circle[0]), round(circle[1])), round(circle[2]), 1, cv2.FILLED)

    return mask
