import numpy as np

valueArray = [0, 1, 2, 3, 4, 5, 6, 7]

image = []
processedImage = []

for y in range(8):
    helparray = []
    for x in range(8):
        helparray.append(abs(valueArray[x] - valueArray[y]))
    image.append(helparray)
    processedImage.append(helparray)

for y in range(1, 7):
    for x in range(1, 7):
        mask = []
        for i in range(3):
            for j in range(3):
                mask.append(image[y-1+i][x-1+j])
        mask.sort()
        processedImage[y][x] = mask[3]

print(np.matrix(image))
print("---------------------------------------")
print(np.matrix(processedImage))
