import matplotlib.pyplot as plt
import numpy as np
inputImage = [[0, 1, 5, 5, 5, 5, 5, 0],
              [1, 1, 1, 4, 5, 5, 5, 4],
              [1, 0, 0, 1, 6, 6, 6, 4],
              [1, 0, 0, 3, 6, 6, 6, 4],
              [1, 0, 3, 3, 6, 4, 4, 7],
              [0, 0, 3, 2, 6, 4, 4, 4],
              [0, 2, 2, 2, 6, 4, 4, 4],
              [0, 2, 2, 2, 4, 6, 4, 0],
              [0, 2, 2, 2, 4, 6, 4, 0]
              ]
hist = [0, 0, 0, 0, 0, 0, 0, 0]
for y in range(len(inputImage)):
    for x in range(len(inputImage[0])):
        hist[inputImage[y][x]] += 1

n_hist = []
for i in range(len(hist)):
    n_hist.append(hist[i]/8)

# plot
fig, ax = plt.subplots()

ax.bar(range(8), hist, width=1, edgecolor="white")
plt.show()


def sumValues(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


cs = sumValues(hist)

nj = (cs - cs.min()) * 7
N = cs.max() - cs.min()

cs = nj / N

cs = cs.astype('uint8')

processedImage = cs[inputImage]
print(np.matrix(processedImage))


fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.add_subplot(1, 2, 1)
plt.imshow(inputImage, cmap='gray')
fig.add_subplot(1, 2, 2)
plt.imshow(processedImage, cmap='gray')
plt.show()
