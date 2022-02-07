
import numpy as np
import cv2


def get_transform_limbus(limbus_circle_from, limbus_circle_to):
    scale = limbus_circle_to[2] / limbus_circle_from[2]
    translation = (limbus_circle_to[:2] - scale * limbus_circle_from[:2])
    return np.array([
        [scale, 0, translation[0]],
        [0, scale, translation[1]]
    ])


def transform_keypoints(keypoints, transform):
    points = [
        [k.pt[0], k.pt[1]]
        for k in keypoints
    ]
    points = np.array(points).T
    points = np.concatenate((points, np.ones((1, len(keypoints)))))
    points = transform @ points

    keypoints_transformed = tuple(
        cv2.KeyPoint(x=points[0, i], y=points[1, i],
                     size=k.size, angle=k.angle,
                     response=k.response, octave=k.octave,
                     class_id=k.class_id)
        for i, k in enumerate(keypoints)
    )

    return keypoints_transformed


def get_transform_info(transform, verbose=False):
    a = transform[0, 0]
    b = transform[1, 0]
    alpha = np.arctan2(b, a)
    s = a / np.cos(alpha)

    scale = s
    translation = transform[:, 2]
    rotation = 180*alpha/np.pi

    if verbose:
        print(
            f'translation (px): {translation}, rotation angle (deg): {rotation}, scale: {scale}')

    return scale, translation, rotation


def estimate_transform_from_matches(keypoints_from, keypoints_to, matches):
    points_from = np.array([keypoints_from[m[0].queryIdx].pt for m in matches])
    points_to = np.array([keypoints_to[m[0].trainIdx].pt for m in matches])

    transform, inliers = cv2.estimateAffinePartial2D(
        np.array([keypoints_from[m[0].queryIdx].pt for m in matches]),
        np.array([keypoints_to[m[0].trainIdx].pt for m in matches])
    )

    return transform, inliers


def estimate_transformation_hybrid(keypoints_from, descriptors_from, limbus_circle_from,
                                   keypoints_to, descriptors_to, limbus_circle_to):

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_from, descriptors_to, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if not good:
        return None

    transform_no_rotation = get_transform_limbus(
        limbus_circle_from, limbus_circle_to)
    keypoints_from_transformed = transform_keypoints(
        keypoints_from, transform_no_rotation)
    transform_rotation, _ = estimate_transform_from_matches(
        keypoints_from_transformed, keypoints_to, good)

    last_row = np.expand_dims(np.array([0, 0, 1]), axis=0)
    transform_no_rotation = np.concatenate(
        (transform_no_rotation, last_row.copy()))
    transform_rotation = np.concatenate((transform_rotation, last_row.copy()))
    transform = transform_rotation @ transform_no_rotation

    return transform[:2, :]


def estimate_orientation_hybrid(keypoints_from, descriptors_from, limbus_circle_from,
                                keypoints_to, descriptors_to, limbus_circle_to, verbose=False):

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_from, descriptors_to, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if not good:
        return None

    transform_no_rotation = get_transform_limbus(
        limbus_circle_from, limbus_circle_to)
    keypoints_from_transformed = transform_keypoints(
        keypoints_from, transform_no_rotation)
    transform_rotation, _ = estimate_transform_from_matches(
        keypoints_from_transformed, keypoints_to, good)
    scale, translation, rotation = get_transform_info(transform_rotation)

    if verbose:
        return rotation, transform_rotation, transform_no_rotation
    else:
        return rotation


def transform_2d_matmul(t1, t2):
    last_row = np.expand_dims(np.array([0, 0, 1]), axis=0)
    t1 = np.concatenate((t1, last_row.copy()))
    t2 = np.concatenate((t2, last_row.copy()))
    ret = t1 @ t2

    return ret[:2,:]
