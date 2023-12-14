import math
import numpy as np
image = [
    [131, 133, 145, 153, 163, 173],
    [88, 90, 99, 109, 115, 124],
    [101, 106, 89, 86, 81, 83],
    [148, 162, 106, 43, 23, 29],
    [154, 158, 108, 38, 35, 30],
    [143, 153, 109, 59, 64, 56]
]


def AVG_filter(img):
    heigt = len(img)
    width = len(img[0])
    output = [[0]*width]
    mins = []
    for y in range(1, width - 1):
        row = [0]
        for x in range(1, heigt - 1):
            x0 = x-1
            y0 = y-1
            mask = []
            for i in range(y0, y0+3):
                mask.append(img[i][x0:x0+3])
            AVG = np.array(mask) / 9
            eigen = np.linalg.eigh(AVG)
            # mask = img[y0:y0+3][x0:x0+3]
            min = eigen.eigenvalues.min()
            if (abs(min) > 7.0):
                mins.append((min, x, y))
                row.extend([1])
            else:
                row.extend([0])
                output
        row.extend([0])
        output.append(row)
    output.append([[0]*width])
    return output


points = AVG_filter(image)
for row in points:
    print(row)
