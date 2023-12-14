import math
import numpy as np
import cv2

img_el = [
    [64, 67, 83, 123, 146, 140, 142, 142],
    [66, 73, 96, 129, 146, 141, 139, 143],
    [68, 82, 101, 137, 147, 190, 147, 139],
    [70, 91, 135, 147, 175, 156, 151, 140],
    [70, 95, 137, 175, 156, 151, 140, 143],
    [69, 83, 151, 156, 160, 156, 146, 150],
    [68, 87, 123, 153, 163, 158, 152, 155],
    [68, 88, 126, 153, 164, 158, 155, 155]
]
gauss_mask = [
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
]
movesets = [
    (0, 1),  # 0
    (1, 1),  # 45
    (1, 0),  # 90
    (1, -1),  # 135
    (0, -1),  # 180
    (-1, -1),  # 225
    (-1, 0),  # 270
    (-1, 1),  # 315
]


def Eldeteckt(img, omega=1, threshold=0):
    mask_x = [[0, 0, 2],
              [-1, 0, 1],
              [-2, 0, 0]]
    mask_y = [[0, 1, 2],
              [0, 0, 0],
              [-2, -1, 0]]

    """
    mask_x = [[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]]
    mask_y = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]
    """

    w = len(img[0])
    h = len(img)

    width = len(img)
    height = len(img[0])
    image = np.array(img)
    filtered_image = GausFilter(img)

    imageX = [[0 for i in range(width)] for j in range(height)]
    imageY = [[0 for i in range(width)] for j in range(height)]

    image_rad = [[0.0 for i in range(width)] for j in range(height)]
    imagedist = [[0.0 for i in range(width)] for j in range(height)]

    non_max = [[0.0 for i in range(width)] for j in range(height)]
    # output = [[0.0 for i in range(width)] for j in range(height)]
    edges = [[0.0 for i in range(width)] for j in range(height)]
    max = 0
    for y in range(1, height):
        for x in range(1, width):
            imageX[y][x] = filtered_image[y][x] - filtered_image[y][x - 1]
            imageY[y][x] = filtered_image[y][x] - filtered_image[y-1][x]

            image_rad[y][x] = math.atan2(imageY[y][x], imageX[y][x])
            imagedist[y][x] = imageY[y][x] ** 2 + imageX[y][x] ** 2
            if imagedist[y][x] > max:
                max = imagedist[y][x]
    normalized_dist = [
        [(imagedist[j][i] / max) * 100 for i in range(width)] for j in range(height)]
    thetas = (np.arange(0, 360, 45))
    for h in range(height - 1):
        for w in range(width - 1):
            if (h == 7 and w == 7):
                print()
            if (image_rad[h][w] > 0):
                rad_idx = int(np.rad2deg(image_rad[h][w]) // 45)
                maradek = (image_rad[h][w] / 45) - rad_idx
                if maradek > 24/45:
                    if rad_idx != 7:
                        rad_idx += 1
                    else:
                        rad_idx = 0
            else:
                newDeg = 360.0 + image_rad[h][w]
                rad_idx = int(newDeg // 45)
                if (rad_idx == 8):
                    rad_idx = 0
                maradek = (newDeg / 45) - rad_idx
                if maradek > 24/45:
                    if rad_idx != 7:
                        rad_idx += 1
                    else:
                        rad_idx = 0
            if normalized_dist[h][w] > threshold:
                if (normalized_dist[h][w] > normalized_dist[h + movesets[rad_idx][0]][w + movesets[rad_idx][1]]
                        and normalized_dist[h][w] > normalized_dist[h + movesets[rad_idx - 4][0]][w + movesets[rad_idx - 4][1]]):
                    non_max[h][w] = normalized_dist[h][w]
                    edges[h][w] = 1

            print(thetas[rad_idx], movesets[rad_idx])

        # for y in range(height):
        #     for x in range(width):
        #         if (normalized_dist[y][x] > threshold):
        #             edges[y][x] = 1
        #         else:
        #             edges[y][x] = 0

    #     for x in range(1, w-1): jó csak át kell irni
    #         x1 = (img[y - 1][x - 1] * mask_x[0][0])
    #         x2 = (img[y - 1][x + 1] * mask_x[0][2])
    #         x3 = (img[y][x - 1] * mask_x[1][0])
    #         x4 = (img[y][x + 1] * mask_x[1][2])
    #         x5 = (img[y + 1][x - 1] * mask_x[2][0])
    #         x6 = (img[y + 1][x + 1] * mask_x[2][2])
    #         x_value = x1 + x2 + x3 + x4 + x5 + x6

    #         y1 = (img[y - 1][x - 1] * mask_y[0][0])
    #         y2 = (img[y - 1][x] * mask_y[0][1])
    #         y3 = (img[y - 1][x + 1] * mask_y[0][2])
    #         y4 = (img[y + 1][x - 1] * mask_y[2][0])
    #         y5 = (img[y + 1][x] * mask_y[2][1])
    #         y6 = (img[y + 1][x + 1] * mask_y[2][2])
    #         y_value = y1 + y2 + y3 + y4 + y5 + y6
    #         magnitude = x_value ** 2 + y_value ** 2
    #         if magnitude > 70000:
    #             helpmatrix[0].append(x_value)
    #             helpmatrix[1].append(y_value)
    #         else:
    #             helpmatrix[0].append(0)
    #             helpmatrix[1].append(0)
    #         """
    #         valamiért nem talált 45 fokos
    #                     if x_value == y_value: #45 fok e?
    #             helpmatrix[0].append(x_value)
    #             helpmatrix[1].append(y_value)
    #         else:
    #             helpmatrix[0].append(0)
    #             helpmatrix[1].append(0)
    #         """

    #     helpmatrix[0].append(0)
    #     helpmatrix[1].append(0)
    #     imageX.append(helpmatrix[0])
    #     imageY.append(helpmatrix[1])
    # imageX.append([0] * 10)
    # imageY.append([0] * 10)
    # rad = math.atan2(45, 45)

    # print(np.rad2deg(rad))

    # for y in range(len(imageX)):
    #     helparray = []
    #     for x in range(len(imageX[0])):
    #         rad = math.atan2(imageY[y][x], imageX[y][x])
    #         helparray.append(np.rad2deg(rad))
    #     imagegrad.append(helparray)

    print("----------------------------------------------------------------------------")
    print("Image x=")
    for row in imageX:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(value))
        print(str_value)
    print("-------------------")

    print("Image y=")
    for row in imageY:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(value))
        print(str_value)
    print("-------------------")
    print("Image degs=")
    for row in image_rad:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(np.rad2deg(value)))
        print(str_value)
    print("-------------------")
    print("Image dist=")
    for row in imagedist:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(value))
        print(str_value)
    print("-------------------")
    print("Image non_max dist=")
    for row in non_max:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(value))
        print(str_value)
    print("-------------------")
    print("Image edges=")
    for row in edges:
        str_value = ""
        for value in row:
            str_value += "{},".format(int(value))
        print(str_value)
    print("----------------------------------------------------------------------------")


def avgfilter(img):
    width = len(img)
    height = len(img[0])
    output = [[0 for i in range(width)] for j in range(height)]
    for h in range(1, height - 1):
        for w in range(1, width-1):
            value = 0
            for i in range(-1, 2):
                for y in range(-1, 2):
                    value += img[h + i][w + y]
            output[h][w] = int(value / 9)
    return output


def GausFilter(img):
    width = len(img)
    height = len(img[0])
    output = [[0 for i in range(width)] for j in range(height)]
    for h in range(height):
        for w in range(width):
            if (h == 0 or h == height - 1):
                output[h][w] = img[h][w]
                continue
            if (w == 0 or w == width - 1):
                output[h][w] = img[h][w]
                continue
            value = 0
            for i in range(-1, 2):
                for y in range(-1, 2):
                    value += img[h + i][w + y] * gauss_mask[1 + i][1 + y]
            output[h][w] = value
    return output


if __name__ == '__main__':
    Eldeteckt(img_el)
