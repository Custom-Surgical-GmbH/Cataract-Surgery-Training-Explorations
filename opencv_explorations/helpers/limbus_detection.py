
import cv2
import numpy as np

from .misc import get_in_out_intensity_diff

def detect_limbus(gray, return_all=False, validation='first', considered_ratio_s=0.05):
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(255 - gray, ksize=(0,0), sigmaX=2),
        cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=40,
        minRadius=gray.shape[0]//10, maxRadius=round(gray.shape[0]//1.5)
    )
    
    if circles is None:
        return None
    
    circles = circles[0]
    if return_all:
        return circles

    assert (validation in ('first', 'inout')), f'unknown \'{validation}\' validation method'
    if validation == 'first':
        return circles[0,:]
    elif validation == 'inout':
        considered_ratio = considered_ratio_s
        considered_circles = []
        while len(considered_circles) == 0:
            considered_circles = circles[:int(len(circles)*considered_ratio)]
            considered_ratio *= 2
        
        in_out_diff_intensities = np.zeros(len(considered_circles))
        for index, circle in enumerate(considered_circles):
            in_out_diff_intensities[index] = get_in_out_intensity_diff(
                gray,
                tuple(np.around(circle[:2]).astype('int')),
                np.round(circle[2]).astype('int')
            )
            
        best_circle_index = np.argmax(in_out_diff_intensities)
        return considered_circles[best_circle_index]

