
import cv2
import numpy as np


def estimate_transform(frame1, frame2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.01)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    if des1 is None or des2 is None:
        return None

    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if not good:
        return None

    transform, inliers = cv2.estimateAffinePartial2D(
        np.array([kp1[m[0].queryIdx].pt for m in good]),
        np.array([kp2[m[0].trainIdx].pt for m in good])
    )

    # TODO: NOT PRETTY AT ALL
    try:
        if np.isnan(transform).any():
            return None
    except:
        return None

    return transform


def get_transform_info(transform, parameter='all'):
    assert parameter in ('all', 'scale', 'angle',
                         'translation'), 'unsupported parameter \'%s\'' % parameter

    a = transform[0, 0]
    b = transform[1, 0]
    alpha = np.arctan2(b, a)
    s = a / np.cos(alpha)

    if parameter == 'all':
        return s, alpha, transform[:, 2]
    elif parameter == 'scale':
        return s
    elif parameter == 'angle':
        return alpha
    elif parameter == 'translatioin':
        return transform[:, 2]
