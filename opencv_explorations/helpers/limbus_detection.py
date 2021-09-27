
import cv2
import numpy as np

from .misc import get_in_out_intensity_diff

def detect_circle(gray, return_all=False, validation='first', considered_ratio_s=0.05, 
        validation_mode='max', validation_value_thresh=None, view_mask=None,
        min_radius_ratio=1/40, max_radius_ratio=1.0):
    assert validation_mode in ('min', 'max'), 'validation_mode \'%s\' is not supported' % validation_mode

    min_radius = np.min(gray.shape)*min_radius_ratio
    max_radius = np.max(gray.shape)*max_radius_ratio

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(255 - gray, ksize=(0,0), sigmaX=2),
        cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=40,
        minRadius=round(min_radius), maxRadius=round(max_radius)
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


def detect_pupil_thresh(pupil_thres_mask, pca_correction=False, pca_correction_ratio=1.0):
    # morphological processing
    kernel = np.ones((3,3),np.uint8) # could be automatically set based on image moments
    pupil_thres_mask = cv2.morphologyEx(pupil_thres_mask, cv2.MORPH_OPEN, kernel)
    pupil_thres_mask = cv2.morphologyEx(pupil_thres_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # pca
    points = np.array((np.where(pupil_thres_mask == 255)[1], np.where(pupil_thres_mask == 255)[0])).T
    points = points.astype(np.float32)
    if points.size == 0:
        return None

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
    mean = mean[0]

    # correction
    if pca_correction and eigenvalues[1][0]/eigenvalues[0][0] > pca_correction_ratio: 
        pca_correction = False

    if not pca_correction:
        radius = np.sum(np.sqrt(eigenvalues))
        return np.append(mean, radius)
    else:
        mean_shift_scale = 2*(np.sqrt(eigenvalues[0][0]) - np.sqrt(eigenvalues[1][0]))
        mean_shift = mean_shift_scale*eigenvectors[1]

        # determining correct sign
        image_center = np.array((pupil_thres_mask.shape[1], pupil_thres_mask.shape[0]), dtype=np.float32)/2
        mean_corrected1 = mean - mean_shift
        mean_corrected2 = mean + mean_shift

        mean_corrected = None
        if np.linalg.norm(mean_corrected1 - image_center) > np.linalg.norm(mean_corrected2 - image_center):
            mean_corrected = mean_corrected1
        else:
            mean_corrected = mean_corrected2
        
        radius = 2*np.sqrt(eigenvalues[0][0])
        return np.append(mean_corrected, radius)
