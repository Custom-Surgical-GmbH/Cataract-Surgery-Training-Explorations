
import numpy as np
import cv2

#############
# CONSTANTS #
#############

RED_LLTH = np.array([150, 90, 60])
RED_LUTH = np.array([179, 240, 230])
RED_ULTH = np.array([0, 90, 60])
RED_UUTH = np.array([15, 240, 230])

GREEN_LTH = np.array([40, 40, 75])
GREEN_UTH = np.array([100, 240, 230])

# BLUE_LTH = np.array([110, 40, 25])
# BLUE_UTH = np.array([160, 240, 190])
BLUE_LTH = np.array([110, 25, 75])
BLUE_UTH = np.array([160, 165, 190])

BLACK_LTH = np.array([0, 0, 0])
BLACK_UTH = np.array([179, 40, 170])

SCLERA_TO_LIMBUS_RATIO = 1.7
PLANAR_SCLERA_TO_LIMBUS_RATIO = 1.25
ENTROPY_EPS = np.log(0.7)

MOMENT_DECIDER = 'mu02'

###########
# METHODS #
###########

def segment_color(hsv, color):
    if color == 'red':
        red_l = cv2.inRange(hsv, RED_LLTH, RED_LUTH)
        red_u = cv2.inRange(hsv, RED_ULTH, RED_UUTH)
        result = cv2.bitwise_or(red_l, red_u)        
    elif color == 'green':
        result = cv2.inRange(hsv, GREEN_LTH, GREEN_UTH)
    elif color == 'blue':
        result = cv2.inRange(hsv, BLUE_LTH, BLUE_UTH)
    elif color == 'black':
        result = cv2.inRange(hsv, BLACK_LTH, BLACK_UTH)
    else:
        raise ValueError('unknown color: %s' % color)
    
    return result

def stack_vertical(strips, split=0.25):
    width = strips[0].shape[1]
    height = strips[0].shape[0]
    result = np.zeros((height*len(strips),width))
    for i, strip in enumerate(strips):
        split_th = round((i*split % 1.0) * width)
        result[i*height:(i+1)*height,width-split_th:] = strip[:,0:split_th]
        result[i*height:(i+1)*height,:width-split_th] = strip[:,split_th:]
    
    return result

def polar_transform(img, limbus_center, limbus_radius):
    polar = cv2.warpPolar(
        img,
        (115, 720),
        tuple(limbus_center),
        limbus_radius*SCLERA_TO_LIMBUS_RATIO,
        cv2.WARP_POLAR_LINEAR+cv2.WARP_FILL_OUTLIERS
    )
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)
    polar = polar[70:]

    return polar


def detect_markers_entropy(hsv, limbus_center, limbus_radius, return_verbose=False):
    hsv_polar = polar_transform(hsv, limbus_center, limbus_radius)
    
    red_polar = segment_color(hsv_polar, 'red')
    green_polar = segment_color(hsv_polar, 'green')
    blue_polar = segment_color(hsv_polar, 'blue')
#     black_polar = segment_color(hsv_polar, 'black')
#     black_polar[:,:] = 0 # TODO: needs better color segmentation
    
    colors_stacked = stack_vertical((red_polar, blue_polar, green_polar))
#     colors_stacked = stack_vertical((
#         red_polar, blue_polar, green_polar, black_polar))

    window_width = round(10*colors_stacked.shape[1]/360)
    colors_stacked_aug = np.zeros((
        colors_stacked.shape[0],
        colors_stacked.shape[1]+window_width
    ))
    colors_stacked_aug[:,window_width//2:colors_stacked_aug.shape[1]-window_width//2] = colors_stacked
    colors_stacked_aug[:,:window_width//2] = colors_stacked[:,colors_stacked.shape[1]-window_width//2:]
    colors_stacked_aug[:,colors_stacked_aug.shape[1]-window_width//2:] = colors_stacked[:,:window_width//2]
    entropy = []
    for i in range(360):
        window_center = round(i*colors_stacked.shape[1]/360)
        window = colors_stacked_aug[:,(window_center-window_width//2):(window_center+window_width//2+1)]           
        
        values = np.sum(window, axis=1)
        values = values[values != 0]
        if values.shape[0] == 0:
            entropy.append(0)
            continue

        values = values / np.sum(values)
        entropy.append(np.sum(np.multiply(-values, np.log(values))))
        
    entropy = np.array(entropy)
    max_entropy = np.max(entropy)
    optimal_degs = np.where(entropy > (max_entropy + ENTROPY_EPS))
    optimal_deg = np.median(optimal_degs)
#     print(optimal_deg, optimal_degs)
    
    rad = 2*np.pi*optimal_deg/entropy.size
    loc = limbus_radius*np.array([np.cos(rad), -np.sin(rad)])
    loc += limbus_center
    
    if return_verbose:
        vis = cv2.cvtColor(colors_stacked.astype('uint8'), cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, 'red', (0,round(vis.shape[0]*0.00 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(vis, 'blue', (0,round(vis.shape[0]*0.33 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(vis, 'green', (0,round(vis.shape[0]*0.66 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        return loc, vis, (red_polar, blue_polar, green_polar), colors_stacked
    
    return loc


def detect_markers_hu_moments(hsv, limbus_center, limbus_radius, return_verbose=False):
    hsv_polar = cv2.warpPolar(
        hsv,
        (0, 0),
        tuple(limbus_center),
        limbus_radius*PLANAR_SCLERA_TO_LIMBUS_RATIO,
        cv2.WARP_POLAR_LINEAR+cv2.WARP_FILL_OUTLIERS
    )
    hsv_polar = cv2.rotate(hsv_polar, cv2.ROTATE_90_CLOCKWISE)
    hsv_polar = cv2.resize(hsv_polar, (0, 0), fx=2.0, fy=1.0)
    hsv_polar = hsv_polar[round(limbus_radius):,:]
    
    red_polar = segment_color(hsv_polar, 'red')
    green_polar = segment_color(hsv_polar, 'green')
    blue_polar = segment_color(hsv_polar, 'blue')
    black_polar = segment_color(hsv_polar, 'black')
    black_polar[:,:] = 0 # TODO: needs better color segmentation
    
    colors_stacked = stack_vertical((
        green_polar, black_polar, red_polar, blue_polar))
#     colors_stacked = stack_vertical((
#         red_polar, blue_polar, green_polar, black_polar))

    window_width = round(10*colors_stacked.shape[1]/360)
    moments = []
    for i in range(360):
        window_center = round(i*colors_stacked.shape[1]/360)
        window = colors_stacked[
            :,
            max(0, window_center-window_width//2):min(colors_stacked.shape[1],window_center+window_width//2)
        ]
        moments.append(cv2.moments(window, binaryImage=True))
        
    moments = np.array([
        moment[MOMENT_DECIDER] for moment in moments
    ])
    
    max_deg = np.argmax(moments)
    rad = 2*np.pi*max_deg/moments.size
    loc = limbus_radius*np.array([np.cos(rad), -np.sin(rad)])
    loc += limbus_center
    
    if return_verbose:
        vis = cv2.cvtColor(colors_stacked.astype('uint8'), cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, 'green', (0,round(vis.shape[0]*0.00 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(vis, 'black', (0,round(vis.shape[0]*0.25 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(vis, 'red', (0,round(vis.shape[0]*0.50 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(vis, 'blue', (0,round(vis.shape[0]*0.75 + 10)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        return loc, vis, (green_polar, black_polar, red_polar, blue_polar)
    
    return loc

