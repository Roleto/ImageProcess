import cv2
import numpy as np


def Cropimage(path, startingnumber):
    img = cv2.imread(path)
    h = img.shape[0]  # Print image shape
    w = img.shape[1]  # Print image shape

    for y in range(0, h, 65):

        for x in range(0, w, 65):
            # Cropping an image
            if y + 65 > h:
                cropped_image = img[h-64:h, x:x+64, :]
            else:
                cropped_image = img[y:y+64, x:x+64, :]
    # Save the cropped image
            if startingnumber > 1000:
                crop_image_path = "DashCam/ImageProccesings/CroppedImages/image{}.png".format(
                    startingnumber)
            elif startingnumber > 100:
                crop_image_path = "DashCam/ImageProccesings/CroppedImages/image0{}.png".format(
                    startingnumber)
            elif startingnumber > 10:
                crop_image_path = "DashCam/ImageProccesings/CroppedImages/image00{}.png".format(
                    startingnumber)
            else:
                crop_image_path = "DashCam/ImageProccesings/CroppedImages/image000{}.png".format(
                    startingnumber)

            if (cropped_image.shape[0] == 64 and cropped_image.shape[1] == 64):
                cv2.imwrite(crop_image_path, cropped_image)
                startingnumber += 1
            if x + 65 > w:
                cropped_image = img[y:y+64, w-64:w, :]
                break


if __name__ == "__main__":
    Cropimage('', 0)
    # Cropimage('Datasets/Images/myImages/myImage4.png', 999)
    # ennél tartok foolytasd még a 2,3,4 van hátra mégse nem mindegyik kép jó kezd előrol
