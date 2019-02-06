import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("H:\\AR-ExtendingOnlineGames\\my_board\\boards\\d2d0b33d-14e7-4a6d-ae5c-18f633bbf9f4.jpg")

scale_percent = 60 # percent of original size

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)

# clock wise from top left corner
corners = [
    (51.7124, 161.505),
    (479.469, 159.418),
    (487.815, 587.175),
    (55.8856, 593.43)
]


plt.imshow(resized)
plt.show()
#cv2.imshow("image", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

