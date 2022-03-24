"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import LOAD_RGB

import cv2 as cv

# OpenCV trackbar example - https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
title_window = "Gamma Correction"
max_val = 200


def on_trackbar(val):
    val = val / 100
    print(val)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # the OpenCV reads the colors the opposite
    if rep == 1:
        rep = 2
    else:
        rep = 1
    img_in = cv.imread(img_path, rep)
    # check
    if img_in is None:
        print('Could not open or find the image: ', img_path)
        exit(0)
    # denormalize
    img_in = img_in / 255

    cv.namedWindow(title_window)
    trackbar_name = 'Gamma'

    # since the OpenCV trackbar has only integer values, we will use it by multiplying by 100,
    # and when calculating the gamma we will divide by 100
    cv.createTrackbar(trackbar_name, title_window, 100, max_val, on_trackbar)

    # create a copy, in order to calculate each gamma correction from the origin image
    img_out = img_in.copy()
    # then, run until exit the window
    while True:
        # get the wanted resolution
        gamma = cv.getTrackbarPos(trackbar_name, title_window)
        # divide by 100, as explained above
        gamma = gamma / 100
        # then make the correction
        img_in = np.power(img_out, gamma)
        # and show the image
        cv.imshow(title_window, img_in)
        cv.waitKey(1)


def main():
    # gammaDisplay('images/bac_con.png', LOAD_GRAY_SCALE)
    gammaDisplay('testImg2.jpg', LOAD_RGB)
    # gammaDisplay('testImg2.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
