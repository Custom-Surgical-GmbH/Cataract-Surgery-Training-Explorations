
import cv2
import numpy as np


def get_avg_laplacian(laplacian, center, radius):
    mask = np.zeros(laplacian.shape, dtype=np.byte)
    cv2.circle(mask, center, radius, 1)
    return np.mean(np.abs(laplacian[mask == 1]))


def get_mean_intensity(grey, center, radius, width_to_radius_ratio=0.05, mode='out'):
    assert mode in ('in', 'out', 'filled'), 'mode %s is not supported' % mode
     
    mask = np.zeros(grey.shape, dtype=np.byte)
    width = int(width_to_radius_ratio*radius)
    
    if mode == 'in':
        cv2.circle(mask, center, radius - (width//2), 1, thickness=width)
    elif mode == 'out':
        cv2.circle(mask, center, radius + (width//2), 1, thickness=width)
    elif mode == 'filled':
        cv2.circle(mask, center, radius, 1, thickness=cv2.FILLED)
    return np.mean(grey[mask == 1])


def get_circle_in_strip_out_intensity_diff(grey, center, radius, strip_width_to_radius_ratio=0.04, metric='in_out'):
    strip_width = int(strip_width_to_radius_ratio*radius)

    mask = np.zeros(grey.shape, dtype=np.byte)
    cv2.circle(mask, center, radius, 1, thickness=cv2.FILLED)
    in_intensity = np.mean(grey[mask == 1])
    
    mask = np.zeros(grey.shape, dtype=np.byte)
    cv2.circle(mask, center, radius + (strip_width//2), 1, thickness=strip_width)
    out_intensity = np.mean(grey[mask == 1])
    
    assert metric in ('in_out', 'out_in'), 'metric %s is not supported' % metric
    
    if metric == 'in_out':
        return out_intensity - in_intensity
    elif metric == 'out_in':
        return in_intensity - out_intensity
    
    return None


CIRCLE_WIDTH_TO_RADIUS_RATIO = 0.04
def get_in_out_intensity_diff(grey, center, radius):
    mask = np.zeros(grey.shape, dtype=np.byte)
    circle_width = int(CIRCLE_WIDTH_TO_RADIUS_RATIO*radius)
    
    cv2.circle(mask, center, radius - (circle_width//2), 1, thickness=circle_width)
    in_intensity = np.mean(grey[mask == 1])
    
    mask = np.zeros(grey.shape, dtype=np.byte)
    cv2.circle(mask, center, radius + (circle_width//2), 1, thickness=circle_width)
    out_intensity = np.mean(grey[mask == 1])
    
    return out_intensity - in_intensity

# def jiggle_circle(grey, best_circle, mode='min', max_iter=5, return_intermediates=False):
#     new_best_circle = np.copy(best_circle)
#     for _ in range(max_iter):
#         mask = np.zeros(grey.shape, dtype=np.byte)
#         circle_width = round(CIRCLE_WIDTH_TO_RADIUS_RATIO*new_best_circle[2])
#         cv2.circle(mask, (new_best_circle[0], new_best_circle[1]), 
#             radius + circle_width//2, 1, thickness=circle_width)
        
