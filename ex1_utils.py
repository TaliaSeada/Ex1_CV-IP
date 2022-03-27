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
# git of this project - https://github.com/TaliaSeada/Ex1_ComputerVisionAndImageProcessing
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

    # Checking that representation is 1 or 2
    if representation != 1 and representation != 2:
        print("Insert only 1 for GRAY_SCALE or 2 for RGB")
        exit(0)
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

    if filename is None:
        print('Could not open or find the file:', filename)
        exit(0)
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

    if imgRGB is None:
        print('Could not open or find the image:', imgRGB)
        exit(0)
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

    if imgYIQ is None:
        print('Could not open or find the image:', imgYIQ)
        exit(0)
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

    if imgOrig is None:
        print('Could not open or find the image:', imgOrig)
        exit(0)

    # if an RGB image is given we will use only the Y channel of the corresponding YIQ image
    flag = 0
    if imgOrig.ndim == 3:
        flag = 1
        imgOrigYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = imgOrigYIQ[:, :, 0]

    # calculate the image histogram (range = [0, 255])
    imgOrig = (imgOrig * 255).astype(int)
    histOrg, bin_edges = np.histogram(imgOrig, bins=256, range=(0, 255))
    # calculate the normalized Cumulative Sum
    cumSum = np.cumsum(histOrg)
    allPixels = cumSum.max()

    # Create a LookUpTable(LUT), such that for each intensity i, LUT[i] = ceiling(CumSum[i] / allPixels * 255)
    lut = np.ceil((cumSum / allPixels) * 255)

    # Replace each intensity i with LUT[i]
    imEq = imgOrig.copy()
    for i in range(256):
        imEq[imgOrig == i] = int(lut[i])

    # calculate the equalized histogram (range = [0, 255])
    histEq, bin_edges = np.histogram(imEq, bins=256, range=(0, 255))

    # if we converted the RBG to YIQ, we need to convert it back
    if flag == 1:
        # convert back
        imgOrigYIQ[:, :, 0] = imEq / 255
        imEq = transformYIQ2RGB(imgOrigYIQ)
    # else we just normalize again the picture
    else:
        imEq = imEq / 255

    return imEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    if imOrig is None:
        print('Could not open or find the image:', imOrig)
        exit(0)

    # if an RGB image is given we will use only the Y channel of the corresponding YIQ image
    flag = 0
    if imOrig.ndim == 3:
        flag = 1
        imgOrigYIQ = transformRGB2YIQ(imOrig)
        imOrig = imgOrigYIQ[:, :, 0]

    # send to quantization
    images, error = _QuanMain(imOrig, nQuant, nIter)

    # if we converted the RBG to YIQ, we need to convert it back for each image
    if flag == 1:
        for i in range(len(images)):
            imgOrigYIQ[:, :, 0] = images[i] / 255
            images[i] = transformYIQ2RGB(imgOrigYIQ)

    return images, error


def _QuanMain(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # Each iteration in the quantization process contains two steps:
    #  • Finding z - the borders which divide the histograms into segments.
    #   z is a vector containing nQuant + 1 elements.The first and last elements are 0 and 255 respectively.
    #  • Finding q - the values that each of the segments' intensities will map to. q is also a vector,
    #   however, containing nQuant elements.

    images = []
    error = []  # nIter elements (or less in case of converges)
    # set z (borders)
    z = []
    for i in range(nQuant + 1):
        # in order to know where the boundaries need to be, we will calculate by (255/nQuant)*i:
        # i is the num of border we placing
        # first boundary is at 0 and last boundary is at 255
        z.append(int((255 / nQuant) * i))

    # getting the histogram
    imflat = (imOrig.flatten() * 255).astype(int)
    hist, bin_edges = np.histogram(imflat, bins=256)

    # perform the two steps above nIter times
    for i in range(nIter):
        # create the new image
        newImg = np.zeros_like(imOrig)
        # create list of the weighted means
        means = []

        # calculate the average of each part
        for j in range(nQuant):
            p_z_i = hist[z[j]:z[j + 1]]
            z_i = np.array(range(z[j], z[j + 1]))
            weightedAvg = (z_i * p_z_i).sum() / p_z_i.sum()
            # add to list
            means.append(weightedAvg)

        # now change the intensity of the image
        for m in range(len(means)):
            # for each boundary change the pixels in it to the new color (the mean we calculated for each boundary)
            newImg[imOrig > z[m] / 255] = means[m]

        # improve boundaries, without changing the first and last ones
        # using z_i = (q_i + q_i+1) / 2
        for b in range(1, len(z) - 1):
            z[b] = int((means[b - 1] + means[b]) / 2)

        # calculate MSE
        mse = np.sqrt((imOrig * 255 - newImg) ** 2).mean()
        error.append(mse)

        # finally add the new image to the list
        images.append(newImg)

    # plt.plot(error)
    # plt.show()
    return images, error
