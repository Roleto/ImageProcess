from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy


def Negate(image):
    img = np.zeros([396, 400, 3])
    print('Processing .')
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            blue = image[x][y][0]
            green = image[x][y][1]
            red = image[x][y][2]

            image[x, y, 0] = 255 - red
            image[x, y, 1] = 255 - green
            image[x, y, 2] = 255 - blue

    print('Done')
    return image


def create_gamma_lut(image, gamma):
    """Gamma paraméter értéknek megfelelő 256 elemű keresőtábla generálása.
    A hatványozás miatt először [0, 1] tartományra kell konvertálni, majd utána vissza [0, 255] közé.
    """
    # lut = np.arange(0, 256, 1, np.float32)
    lut = image / 255.0
    lut = lut ** gamma
    lut = np.uint8(lut * 255.0)
    print('done')
    return lut


def Log(image):
    # c = 255 / np.log(1 + np.max(image))
    c = 35
    log_image = c * (np.log(image + 1.0))
    log_image = np.uint8(log_image)
    return log_image
