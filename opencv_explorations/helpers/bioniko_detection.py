
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

BIONIKO_HEIGHT_P = 0.18
BIONIKO_WIDTH_P = 0.14
BIONIKO_ASPECT = 5.7
BIONIKO_FONT_FACTOR = 1.35


def write_bioniko(height, color=0, background=255):
    bioniko = background*np.ones((height, round(height*BIONIKO_ASPECT)))
    pil_im = Image.fromarray(bioniko)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(
        'data/century-gothic-bold.ttf', size=round(height*BIONIKO_FONT_FACTOR))
    draw.text((0, 0), 'BIONIKO', color, font=font, anchor='lt')
    bioniko = np.array(pil_im).astype('uint8')

    return bioniko


def detect_bioniko(gray, limbus_center, limbus_radius, return_verbose=False):
    # TODO: use multiple polar coordinate images to avoid 'BIONIKO' being cut
    gray_polar = cv2.warpPolar(gray, (0, 0), tuple(
        limbus_center), limbus_radius, cv2.WARP_POLAR_LINEAR)
    gray_polar = cv2.rotate(gray_polar, cv2.ROTATE_90_CLOCKWISE)
    gray_polar = cv2.resize(gray_polar, (0, 0), fx=2.0, fy=1.0)

    bioniko_height = min(
        round(limbus_radius*BIONIKO_HEIGHT_P), gray_polar.shape[0])
    bioniko = write_bioniko(bioniko_height, background=255)
    ccoeff_normed = cv2.matchTemplate(
        gray_polar, bioniko, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(ccoeff_normed)

    cv2.drawMarker(ccoeff_normed, max_loc, 255)
    cv2.line(gray_polar, (max_loc[0], 0),
             (max_loc[0], gray_polar.shape[0]-1), 255, 1)
    gray_polar[max_loc[1]:(max_loc[1]+bioniko.shape[0]),
               max_loc[0]:(max_loc[0]+bioniko.shape[1])] = bioniko
    cv2.putText(gray_polar, ' %s; %s' % (max_loc, max_val),
                (0, gray_polar.shape[0]-1), cv2.FONT_HERSHEY_SIMPLEX, 1, 0)

    rad = 2*np.pi*max_loc[0]/gray_polar.shape[1]
    loc = limbus_radius*np.array([np.cos(rad), -np.sin(rad)])
    loc += limbus_center

    if return_verbose:
        return loc, gray_polar, ccoeff_normed

    return loc
