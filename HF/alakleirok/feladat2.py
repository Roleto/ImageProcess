import numpy as np
image = [
    [64, 67, 83, 122, 146, 140, 142, 142],
    [66, 73, 96, 129, 146, 141, 139, 143],
    [68, 82, 101, 137, 147, 140, 147, 139],
    [70, 91, 135, 147, 155, 156, 151, 140],
    [70, 95, 137, 155, 156, 151, 140, 143],
    [69, 83, 141, 156, 160, 156, 146, 150],
    [68, 87, 123, 153, 163, 158, 152, 155],
    [68, 88, 126, 156, 164, 158, 155, 155]
]


def kuszoboles(img=[], t=0.3, epst=0.01):
    norm = np.array(img) / 255
    while True:
        mL = norm[norm <= t].mean()  # átlag
        mH = norm[norm > t].mean()
        t_new = (mL + mH) / 2
        if (abs(t-t_new) < epst):
            break
        t = t_new
    return t


def niblackTH(img, n=5, k=-0.2):
    dx = len(img)
    dy = len(img[0])
    temparray = np.array(img)
    imgN = [[0]*dy]*dx
    w = int((n-1) / 2)
    for x in range(w+1, dx-w):
        for y in range(w+1, dy-w):
            block = np.array(temparray[x-w:x+w, y-w:y+w])
            wBmn = block.mean()  # atlag
            wBstd = block.std()  # szorás
            wBTH = (wBmn + k * wBstd)
            if (img[x][y] > wBTH):
                imgN[x][y] = 255
    return imgN


# kuszoboles(image)
for row in niblackTH(image):
    print(row)
