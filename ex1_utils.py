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
from itertools import repeat
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211551601


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    # if the representation is given as 1, meaning Gray Scale we will convert the img from BGR to GRAY
    if representation == 1:
        img_rep = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.gray()
    # else if the representation is given as 2, meaning RGB we will convert the img from BGR to RGB
    else:
        img_rep = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # finally we will normalize the picture to be in the range [0,1]
    img_rep = img_rep / 255
    return img_rep


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # use the imReadAndConvert function to upload the picture then display it
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # set the matrix
    mult = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.275, -0.321],
                     [0.212, -0.523, 0.311]])
    # multiply every pixel in order to transpose to YIQ
    YIQ = imgRGB.dot(mult)
    # then return the new image
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # set the inverse matrix using numpy
    mult = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.275, -0.321],
                     [0.212, -0.523, 0.311]])
    mult = np.linalg.inv(mult)
    # multiply every pixel in order to transpose to RGB
    RGB = imgYIQ.dot(mult)
    # then return the new image
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass


if __name__ == '__main__':
    path = 'images/beach.jpg'
    img = imReadAndConvert(path, LOAD_RGB)
    # plt.imshow(img)
    # imDisplay(path, LOAD_RGB)
    img = transformRGB2YIQ(img)
    # plt.imshow(img)
    img = transformYIQ2RGB(img)
    plt.imshow(img)









    plt.show()
