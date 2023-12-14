import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
from PIL import Image


image = [
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0]
]

og_image = [
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0]
]


def hugh_transformacio(img, step=10):
    thetas = np.deg2rad(np.arange(0, 180, step))

    width = len(img[0])
    height = len(img)
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    max = 0
    max_id = []

    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
            if (accumulator[rho, t_idx] > max):
                max = accumulator[rho, t_idx]
                max_id = []
            if (accumulator[rho, t_idx] == max):
                max_id.append((rho, t_idx))

    for rho, t_idx in max_id:
        a = cos_t[t_idx]  # cos téta
        b = sin_t[t_idx]  # sin téta
        x0 = int(a * rhos[rho])
        y0 = int(b * rhos[rho])
        x1 = int((x0 + 1000) * (-b))
        y1 = int((y0 + 1000) * (a))
        x2 = int((x0 - 1000) * (-b))
        y2 = int((y0 - 1000) * (a))
        y_d = 0 - y0
        if (x1 == x2):
            for y in range(y0, y0 - y0):
                image[y][x0] = 5
            for y in range(0, height - y0):
                image[y][x0] = 5
        if (y1 == y2):
            for x in range(x0, 0):
                image[y0][x] = 5
            for x in range(0, width - x0):
                image[y0][x] = 5

    plt.show()
    # for rho, t_idx, x, y in max_id:
    #     print("nem tudom")
    return accumulator, thetas, rhos, image


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    imgpath = 'imgs/binary_crosses.png'
    for row in image:
        print(row)
    accumulator, thetas, rhos, newimage = hugh_transformacio(image)
    print("-------------------------------")
    for row in newimage:
        print(row)
    # show_hough_line(image, accumulator, thetas, rhos)
