
import cv2
import numpy as np
from .limbus_detection import detect_circle

def get_view_mask(grey, radius_shrink=1.0, return_circle=False):
    assert 0.0 < radius_shrink <= 1.0, 'radius_shrink should be in the interval (0.0, 1.0]'

    circle = detect_circle(grey, validation='inout', validation_mode='min')
    if circle is None:
        return None

    mask = np.zeros(grey.shape, dtype=np.uint8)
    circle[2] *= radius_shrink
    cv2.circle(mask, (round(circle[0]), round(circle[1])), round(circle[2]), 255, cv2.FILLED)

    if return_circle:
        return mask, circle
    else:
        return mask
