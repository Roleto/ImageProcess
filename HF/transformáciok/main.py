import math
import numpy as np
triangel = [1.0, 1.0, 1.0]
homo_triangle = [
    [1.0],
    [1.0],
    [1.0],
    [1.0]
]


def mov(vector=[2, 0, 0]):
    output = [triangel[0] + vector[0], triangel[1] +
              vector[1], triangel[2] + vector[2]]
    return output


def rotate(input, deg=45, axis='z'):
    cos = math.cos(np.deg2rad(deg))
    sin = math.sin(np.deg2rad(deg))
    x_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, cos, (-sin)],
        [0.0, sin, cos]
    ]
    y_matrix = [
        [cos, 0.0, sin],
        [0.0, 1.0, 0.0],
        [(-sin), 0.0, cos]
    ]
    z_matrix = [
        [cos, (-sin), 0.0],
        [sin, cos, 0.0],
        [0.0, 0.0, 1.0]
    ]
    if axis == 'x':
        transformation = x_matrix

    if axis == 'y':
        transformation = y_matrix
    if axis == 'z':
        transformation = z_matrix
    output = []
    for x in range(0, 3):
        value = 0
        for y in range(0, 3):
            value += input[x] * transformation[x][y]
        output.append(value)
    return output


triangle_trans = mov()
triangle_trans = rotate(triangle_trans, deg=180, axis='y')
triangle_trans = rotate(triangle_trans)
print('')
