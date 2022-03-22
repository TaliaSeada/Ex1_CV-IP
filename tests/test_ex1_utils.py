from unittest import TestCase

from ex1_utils import imReadAndConvert, imDisplay

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

# img_path = 'images/dog.jpg'
# img_path = 'images/stich.jpg'
# img_path = 'images/bac_con.jpg'
# img_path = 'images/beach.jpg'
# img_path = 'images/water_bear.png'

class Test(TestCase):
    def test_imReadAndConvert_scales(self):
        img_path = '../images/dog.jpg'
        img_gray = imReadAndConvert(img_path, LOAD_GRAY_SCALE)
        img_rgb = imReadAndConvert(img_path, LOAD_RGB)

        # check the dimensions
        self.assertEqual(len(img_gray.shape), 2)
        self.assertEqual(len(img_rgb.shape), 3)

    def test_imReadAndConvert(self):
        img_path = '../images/dog.jpg'
        img = imReadAndConvert(img_path, LOAD_GRAY_SCALE)

        # check type
        self.assertEqual(img.dtype, "float64")
        # check class of matrix
        self.assertEqual(str(type(img)), "<class 'numpy.ndarray'>")
        # check if normalized
        for i in range(len(img)):
            for j in range(len(img[i])):
                self.assertTrue(img[i][j] <= 1)
                self.assertTrue(img[i][j] >= 0)

    def test_imDisplay_Gray(self):
        img_path = '../images/dog.jpg'
        imDisplay(img_path, 1) # Gray image

    def test_imDisplay_RGB(self):
        img_path = '../images/stich.jpg'
        imDisplay(img_path, 2) # RGB image



