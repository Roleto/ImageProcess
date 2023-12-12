import cv2
import numpy as np
img = cv2.imread('DashCam/ImageProccesings/autoroad_2.jpg')
dst = cv2.Mat(img)
median = cv2.medianBlur(img, 5)
cv2.imshow('canvasOutput', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi/180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])
cv2.imwrite('DashCam/ImageProccesings/detectedLines.png', img)
