import math
import numpy as np
inputImage = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
              ]
mask_x = [[1, 0, -1],
          [2, 0, -2],
          [1, 0, -1]]
mask_y = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]

imageX = [[0]*10]
imageY = [[0]*10]
imagegrad = []

for y in range(1, 9):
    helpmatrix = [[0], [0]]
    for x in range(1, 9):
        x1 = (inputImage[y-1][x-1]*mask_x[0][0])
        x2 = (inputImage[y-1][x+1]*mask_x[0][2])
        x3 = (inputImage[y][x-1]*mask_x[1][0])
        x4 = (inputImage[y][x+1]*mask_x[1][2])
        x5 = (inputImage[y+1][x-1]*mask_x[2][0])
        x6 = (inputImage[y+1][x+1]*mask_x[2][2])
        helpmatrix[0].append(x1 + x2 + x3 + x4 + x5 + x6)

        y1 = (inputImage[y-1][x-1]*mask_y[0][0])
        y2 = (inputImage[y-1][x]*mask_y[0][1])
        y3 = (inputImage[y-1][x+1]*mask_y[0][2])
        y4 = (inputImage[y+1][x-1]*mask_y[2][0])
        y5 = (inputImage[y+1][x]*mask_y[2][1])
        y6 = (inputImage[y+1][x+1]*mask_y[2][2])
        helpmatrix[1].append(y1 + y2 + y3 + y4 + y5 + y6)
    helpmatrix[0].append(0)
    helpmatrix[1].append(0)
    imageX.append(helpmatrix[0])
    imageY.append(helpmatrix[1])
imageX.append([0]*10)
imageY.append([0]*10)

# nem értettem mire gondolt a gradiens irányának ábrázolásával de ki számitottam őket
for y in range(10):
    helparray = []
    for x in range(10):
        rad = math.atan2(imageY[y][x], imageX[y][x])
        helparray.append(np.rad2deg(rad))
    imagegrad.append(helparray)


print(np.matrix(imageX))
print("-------------------")
print(np.matrix(imageY))
print("-------------------")
print(np.matrix(imagegrad))
print("-------------------")

euk = []
man = []

for y in range(10):
    helpmatrix = [[], []]
    for x in range(10):
        dist = abs(imageX[y][x]) + abs(imageY[y][x])
        if dist > 3:
            helpmatrix[0].append(inputImage[y][x])
        else:
            helpmatrix[0].append(0)
        dist = imageX[y][x] ** 2 + imageY[y][x]**2
        if dist > 9:
            helpmatrix[1].append(inputImage[y][x])
        else:
            helpmatrix[1].append(0)
    man.append(helpmatrix[0])
    euk.append(helpmatrix[1])

print(np.matrix(euk))
print()
print(np.matrix(man))

# plt.imshow(inputImage, cmap='gray')
# plt.show()
