
import cv2
import numpy as np

def get_avg_laplacian(laplacian, center, radius):
    mask = np.zeros(laplacian.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return np.mean(np.abs(laplacian[mask == 255]))


def get_mean_intensity(grey, center, radius, width_to_radius_ratio=0.05, mode='out'):
    assert mode in ('in', 'out', 'filled'), 'mode \'%s\' is not supported' % mode
     
    mask = np.zeros(grey.shape, dtype=np.uint8)
    width = int(width_to_radius_ratio*radius)
    
    if mode == 'in':
        cv2.circle(mask, center, radius - (width//2), 1, thickness=width)
    elif mode == 'out':
        cv2.circle(mask, center, radius + (width//2), 1, thickness=width)
    elif mode == 'filled':
        cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return np.mean(grey[mask == 255])


def get_circle_in_strip_out_intensity_diff(grey, center, radius, strip_width_to_radius_ratio=0.04, metric='in_out'):
    strip_width = int(strip_width_to_radius_ratio*radius)

    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    in_intensity = np.mean(grey[mask == 255])
    
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius + (strip_width//2), 255, thickness=strip_width)
    out_intensity = np.mean(grey[mask == 255])
    
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
    
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius - (circle_width//2), 255, thickness=circle_width)
    # print('1. mask before', cv2.countNonZero(mask))
    if view_mask is not None:
        mask = mask & view_mask
        # print(cv2.countNonZero(view_mask))
        # print('1. mask after', cv2.countNonZero(mask))
    in_intensity = np.mean(grey[mask == 255])
    
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius + (circle_width//2), 255, thickness=circle_width)
    # print('2. mask before', cv2.countNonZero(mask))
    if view_mask is not None:
        mask = mask & view_mask
        # print('2. mask after', cv2.countNonZero(mask))
    out_intensity = np.mean(grey[mask == 255])

    return out_intensity - in_intensity

def jiggle_circle(grey, best_circle, mode='min', max_iter=30, alpha=0.2, strip_width_to_radius_ratio=0.04, 
        min_change_norm=1.0, return_mean_value=False, return_intermediates=False, view_mask=None):
    assert mode in ('min', 'max'), 'mode \'%s\' is not supported' % mode

    intermediates = []
    circle = np.copy(best_circle).astype('float32')
    strip_width = round(strip_width_to_radius_ratio*circle[2])
    grey = grey.copy()

    # center
    for i in range(max_iter):
        # pre edge treatment
        mask = np.zeros(grey.shape, dtype=np.uint8)
        cv2.circle(mask, (round(circle[0]), round(circle[1])), round(circle[2]) + strip_width//2, 255, strip_width)
        moments = cv2.moments(grey*(mask == 255), False)
        
        # edge treatment
        mean_value = moments['m00']/cv2.countNonZero(mask)
        if view_mask is not None:
            grey[view_mask == 0] = round(mean_value)
        padding = round(circle[2])*2 + strip_width*2
        im_border = cv2.copyMakeBorder(grey, padding, padding, padding, padding, 
                                    cv2.BORDER_CONSTANT, value=round(mean_value))
        mask = np.zeros(im_border.shape, dtype=np.uint8)
        cv2.circle(mask, (round(circle[0]) + padding, round(circle[1]) + padding), round(circle[2]) + strip_width//2, 255, strip_width)
        moments = cv2.moments(im_border*(mask == 255), False)

        # post edge treatment
        direction = np.array((moments['m10']/moments['m00'] - circle[0] - padding,
                            moments['m01']/moments['m00'] - circle[1] - padding))

        if np.linalg.norm(alpha*direction) < min_change_norm:
            break
        
        if mode == 'min':
            circle[:2] -= alpha*direction
        elif mode == 'max':
            circle[:2] += alpha*direction
        
        if return_intermediates:
            im_new = grey.copy()
            cv2.circle(im_new, (round(circle[0]), round(circle[1])), round(circle[2]) + strip_width//2, 255, strip_width)
            intermediates.append(im_new)

    # print('jiggle_circle iters:', i)

    mean_value = moments['m00']/cv2.countNonZero(mask)
    if return_mean_value and not return_intermediates:
        return circle, mean_value
    elif not return_mean_value and return_intermediates:
        return circle, intermediates
    elif return_mean_value and return_intermediates:
        return circle, mean_value, intermediates
    else:
        return circle

def tighten_circle(grey, best_circle, mode='min', max_iter=30, alpha=0.2, beta=0.97, strip_width_to_radius_ratio=0.04,
        max_change_ratio=0.8, initial_bump_up=1.1, return_intermediates=False, view_mask=None):
    assert 0 < beta < 1.0, 'beta \'%f\' has a wrong value' % beta
    assert 0 < max_change_ratio < 1.0, 'max_change_ratio \'%f\' has a wrong value' % max_change_ratio

    intermediates = []
    grey = grey.copy()
    new_circle = np.copy(best_circle).astype('float32')
    new_circle[2] *= initial_bump_up
    circle = None
    last_value = None

    # center
    for i in range(max_iter + 1):
        if i > 0:
            new_circle[2] *= beta

        new_circle, new_value, jiggle_intermediates = jiggle_circle(
            grey, new_circle, mode=mode, alpha=alpha, 
            strip_width_to_radius_ratio=strip_width_to_radius_ratio, return_mean_value=True,
            return_intermediates=True, view_mask=view_mask)
        if jiggle_intermediates:
            intermediates.append(jiggle_intermediates[-1])
        
        if i > 0:
            if new_value < last_value*max_change_ratio or new_value > last_value*(2 - max_change_ratio):
                break

        circle = new_circle
        last_value = new_value
    
    # print('tighten_circle iters:', i)

    if return_intermediates:
        return circle, intermediates
    else:
        return circle

def repair_bbox(bbox, max_width, max_height, max_consider_square_ratio=1.1):
#     assert (bbox[0] != 0 or bbox[1] != 0), 'could not repair bbox: %s' % str(bbox)
#     assert (bbox[0] + bbox[2] < max_width or bbox[1] + bbox[3] < max_height), \
#         'could not repair bbox: %s' % str(bbox)

    if bbox[0] == 0 and bbox[1] == 0:
        return None
    if bbox[0] + bbox[2] == max_width and bbox[1] + bbox[3] == max_height:
        return None

    if max(bbox[2]/bbox[3], bbox[3]/bbox[2]) < max_consider_square_ratio:
        return bbox
    
    new_bbox = list(bbox)
    square_size = max(bbox[2], bbox[3])
    new_bbox[2] = square_size
    new_bbox[3] = square_size
    if bbox[0] == 0:
        new_bbox[0] = -square_size + bbox[2]
    if bbox[1] == 0:
        new_bbox[1] = -square_size + bbox[3]
        
    return tuple(new_bbox)
    
