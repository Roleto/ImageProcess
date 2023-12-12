import cv2
hog = cv2.HOGDescriptor()
im = cv2.imread("DashCam/ImageProccesings/aauto.jpg")
h = hog.compute(im)
print(h)
cv2.imshow('original', im)
cv2.waitKey(0)
# cv2.imshow('hog', h)
# cv2.waitKey(0)
