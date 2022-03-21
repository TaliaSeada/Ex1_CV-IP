from unittest import TestCase

from ex1_utils import imReadAndConvert

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


class Test(TestCase):
    def test_imReadAndConvert(self):
        img_path = '../images/beach.jpg'
        img = imReadAndConvert(img_path, LOAD_GRAY_SCALE)

        # check dtype
        self.assertEqual(img.dtype, "float64")
        # check class of matrix
        self.assertEqual(str(type(img)), "<class 'numpy.ndarray'>")
        # check if normalized
        for i in range(len(img)):
            for j in range(len(img[i])):
                self.assertTrue(img[i][j] <= 1)
                self.assertTrue(img[i][j] >= 0)
