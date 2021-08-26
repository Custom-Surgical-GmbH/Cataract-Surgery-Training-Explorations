
import cv2
import numpy as np

def get_avg_laplacian(laplacian, center, radius):
    mask = np.zeros(laplacian.shape, dtype=np.byte)
    cv2.circle(mask, center, radius, 1)
    return np.mean(np.abs(laplacian[mask == 1]))


def get_mean_intensity(grey, center, radius, width_to_radius_ratio=0.05, mode='out'):
    assert mode in ('in', 'out', 'filled'), 'mode \'%s\' is not supported' % mode
     
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
    
    assert metric in ('in_out', 'out_in'), 'metric \'%s\' is not supported' % metric
    
    if metric == 'in_out':
        return out_intensity - in_intensity
    elif metric == 'out_in':
        return in_intensity - out_intensity
    
    return None


CIRCLE_WIDTH_TO_RADIUS_RATIO = 0.04
def get_in_out_intensity_diff(grey, center, radius, view_mask=None):
    if view_mask is not None:
        assert grey.shape == view_mask.shape, 'grey must have the same shape as view_mask'

    circle_width = int(CIRCLE_WIDTH_TO_RADIUS_RATIO*radius)
    
    mask = np.zeros(grey.shape, dtype=np.byte)
    cv2.circle(mask, center, radius - (circle_width//2), 1, thickness=circle_width)
    if view_mask is not None:
        mask = mask & view_mask
    in_intensity = np.mean(grey[mask == 1])
    
    mask = np.zeros(grey.shape, dtype=np.byte)
    cv2.circle(mask, center, radius + (circle_width//2), 1, thickness=circle_width)
    if view_mask is not None:
        mask = mask & view_mask
    out_intensity = np.mean(grey[mask == 1])
    
    print(out_intensity, in_intensity)

    return out_intensity - in_intensity

def jiggle_circle(grey, best_circle, mode='min', max_iter=30, alpha=0.2, strip_width_to_radius_ratio=0.04, 
        return_intermediates=False, view_mask=None):
    assert mode in ('min', 'max'), 'mode \'%s\' is not supported' % mode

    intermediates = []
    circle = np.copy(best_circle).astype('float32')
    strip_width = round(strip_width_to_radius_ratio*circle[2])
    grey = grey.copy()

    for _ in range(max_iter):
        # pre edge treatment
        mask = np.zeros(grey.shape, dtype=np.byte)
        cv2.circle(mask, (round(circle[0]), round(circle[1])), round(circle[2]) + strip_width//2, 1, strip_width)
        moments = cv2.moments(grey*mask, False)
        
        # edge treatment
        mean_value = moments['m00']/cv2.countNonZero(mask)
        if view_mask is not None:
            grey[view_mask == 0] = round(mean_value)
        padding = round(circle[2])*2 + strip_width*2
        im_border = cv2.copyMakeBorder(grey, padding, padding, padding, padding, 
                                    cv2.BORDER_CONSTANT, value=round(mean_value))
        mask = np.zeros(im_border.shape, dtype=np.byte)
        cv2.circle(mask, (round(circle[0]) + padding, round(circle[1]) + padding), round(circle[2]) + strip_width//2, 1, strip_width)
        moments = cv2.moments(im_border*mask, False)

        # post edge treatment
        direction = np.array((moments['m10']/moments['m00'] - circle[0] - padding,
                            moments['m01']/moments['m00'] - circle[1] - padding))
        
        if mode == 'min':
            circle[:2] -= alpha*direction
        elif mode == 'max':
            circle[:2] += alpha*direction
        
        if return_intermediates:
            im_new = grey.copy()
            cv2.circle(im_new, (round(circle[0]), round(circle[1])), round(circle[2]) + strip_width//2, 255, strip_width)
            intermediates.append(im_new)

    if return_intermediates:
        return circle, intermediates
    else:
        return circle
