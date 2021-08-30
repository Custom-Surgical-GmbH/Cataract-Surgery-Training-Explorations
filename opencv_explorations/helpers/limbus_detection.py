
import cv2
import numpy as np

from .misc import get_in_out_intensity_diff

def detect_limbus(gray, return_all=False, validation='first', considered_ratio_s=0.05, 
        validation_mode='max', validation_value_thresh=None, view_mask=None):
    assert validation_mode in ('min', 'max'), 'validation_mode \'%s\' is not supported' % validation_mode

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(255 - gray, ksize=(0,0), sigmaX=2),
        cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=40,
        minRadius=np.min(gray.shape)//40, maxRadius=np.max(gray.shape[0])
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
        while int(len(circles)*considered_ratio) == 0:
            considered_ratio *= 2

        considered_circles = circles[:int(len(circles)*considered_ratio)] 
        
        in_out_diff_intensities = np.zeros(len(considered_circles))
        for index, circle in enumerate(considered_circles):
            in_out_diff_intensities[index] = get_in_out_intensity_diff(
                gray,
                tuple(np.around(circle[:2]).astype('int')),
                np.round(circle[2]).astype('int'),
                view_mask=view_mask
            )
            
        # print('max_value ', np.max(in_out_diff_intensities))
        optimal_value = None
        if validation_mode == 'min':
            optimal_value = np.nanmin(in_out_diff_intensities)
            if validation_value_thresh is not None and optimal_value > validation_value_thresh:
                return None
            else:
                best_circle_index = np.argmin(in_out_diff_intensities)
        elif validation_mode == 'max':
            optimal_value = np.nanmax(in_out_diff_intensities)
            if validation_value_thresh is not None and optimal_value < validation_value_thresh:
                return None
            else:
                best_circle_index = np.argmax(in_out_diff_intensities)

        return considered_circles[best_circle_index]
