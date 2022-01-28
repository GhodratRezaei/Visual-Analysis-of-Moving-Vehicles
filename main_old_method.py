import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

images_filename_list = os.listdir("video_frames/")
images_filename_list.sort(key=lambda f: int(re.sub("\D", "", f)))

images_list = []

for filename in images_filename_list:
        image = cv2.imread("video_frames/" + filename)
        images_list.append(image)


for i in range(len(images_list) - 1):
        grayscale_image_1 = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2GRAY)
        grayscale_image_2 = cv2.cvtColor(images_list[i + 1], cv2.COLOR_BGR2GRAY)
        diff_image = cv2.absdiff(grayscale_image_2, grayscale_image_1)
        ret, thresholded_image = cv2.threshold(diff_image, 60, 255, cv2.THRESH_BINARY)

        dilation_kernel = np.ones((1,1), np.uint8)
        dilated_image = cv2.dilate(thresholded_image, dilation_kernel, iterations=1)

        plt.imshow(dilated_image, cmap="gray")
        plt.show()
        
        contours, hierarchy = cv2.findContours(thresholded_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        objects_centroid_list = []

        for j, object_centroid in enumerate(contours):
                x, y, w, h = cv2.boundingRect(object_centroid)
                if cv2.contourArea(object_centroid) >= 100:
                        objects_centroid_list.append(object_centroid)

        cv2.drawContours(images_list[i], objects_centroid_list, -1, (255, 0, 0), 2)
        # plt.imshow(images_list[i])
        plt.imshow(cv2.cvtColor(images_list[i], cv2.COLOR_BGR2RGB))
        plt.show()
        

# # count of discovered contours        
# len(valid_cntrs)

# dmy = images_list[13].copy()

# cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
# cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
# plt.imshow(dmy)
# plt.show()

