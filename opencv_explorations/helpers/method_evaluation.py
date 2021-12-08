
import os
import cv2
import json
import numpy as np


def evaluate_limbus(methods, data_dir, append_frame_n=False):
    image_filenames = []
    limbus_groundtruths = []

    for filename in sorted(os.listdir(data_dir)):
        if filename[-5:] != '.json':
            continue

        image_filenames.append(filename[:-5] + '.jpg')
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)
            limbus_shape = [shape for shape in data['shapes']
                            if shape['label'] == 'limbus']
            assert (len(limbus_shape) ==
                    1), f'no unique \'limbus\' label in the file \'{filename}\''

            limbus_shape = limbus_shape[0]
            groundtruth = limbus_shape['points'][0]
            groundtruth.append(np.linalg.norm(
                np.array(limbus_shape['points'][0]) -
                np.array(limbus_shape['points'][1])
            ))
            if append_frame_n:
                groundtruth.append(int(filename[:-5]))
            limbus_groundtruths.append(np.array(groundtruth))

    assert (len(image_filenames) == len(limbus_groundtruths)
            ), 'different number of image_filenames vs. limbus_groundtruths'

    methods_results = []
    for i in range(len(image_filenames)):
        img = cv2.imread(os.path.join(data_dir, image_filenames[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]

        img_methods_results = []
        img_methods_results.append(limbus_groundtruths[i])
        for method in methods:
            method_result = method(img)
            if append_frame_n:
                method_result = np.append(
                    method_result, limbus_groundtruths[i][-1])

            img_methods_results.append(method_result)

        methods_results.append(img_methods_results)

    return np.array(methods_results)


def circles_intersection_area(circle1, circle2):
    d = np.hypot(circle2[0] - circle1[0], circle2[1] - circle1[1])

    if d < (circle1[2] + circle2[2]):
        a = circle1[2]**2
        b = circle2[2]**2

        x = (a - b + d**2) / (2 * d)
        z = x**2
        y = np.sqrt(a - z)

        if d <= np.abs(circle2[2] - circle1[2]):
            return np.pi * min(a, b)

        return a * np.arcsin(y / circle1[2]) + b * np.arcsin(y / circle2[2]) - y * (x + np.sqrt(z + b - a))
    return 0


def circle_area(circle):
    return np.pi * circle[2]**2
