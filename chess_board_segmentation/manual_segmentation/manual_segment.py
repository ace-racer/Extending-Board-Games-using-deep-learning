import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

SPLIT_IMAGES_LOCATION = "H:\\AR-ExtendingOnlineGames\\my_board\\split_images\\board1"
CHESS_BOARD_IMAGE_LOCATION = "H:\\AR-ExtendingOnlineGames\\my_board\\boards\\board1.jpg"
SQUARE_SIDE_LENGTH = 70

if not os.path.exists(SPLIT_IMAGES_LOCATION):
    print("Creating folder to store split images {0}".format(SPLIT_IMAGES_LOCATION))
    os.makedirs(SPLIT_IMAGES_LOCATION)

def split_board(img, create_files=True):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    row = "abcdefgh"
    arr = []
    sq_len = img.shape[0] // 8
    for i in range(8):
        for j in range(8):
            image = img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len]
            position = str(row[i]) + "_" + str(j+1)
            arr.append({"image": image, "position": position})

            if create_files:
                cv2.imwrite(os.path.join(SPLIT_IMAGES_LOCATION, position + ".jpg"), image)
    return arr

def four_point_transform(img, points, square_length=SQUARE_SIDE_LENGTH):
    board_length = square_length * 8
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, board_length], [board_length, board_length], [board_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (board_length, board_length))


img = cv2.imread(CHESS_BOARD_IMAGE_LOCATION)

scale_percent = 99 # percent of original size

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)

# check the image after resize to get the corners
#plt.imshow(resized)
#plt.show()
#exit(0)

# clock wise from top left corner
""" at scale 60
corners = [
    (51.7124, 161.505),
    (479.469, 159.418),
    (487.815, 587.175),
    (55.8856, 593.43)
]
"""
corners = [
    (84.25, 274.75),
    (792.031, 264.426),
    (804.974, 974.787),
    (94.586, 985.12)
]


corners = [(int(round(x[0])), int(round(x[1]))) for x in corners]
print(corners)

        
for point in corners:
    cv2.circle(resized, tuple(point), 5, (0,255,0), -1)

cv2.imwrite(os.path.join(SPLIT_IMAGES_LOCATION, 'corner_points.jpg'), resized)

transformed_board = four_point_transform(resized, corners)
cv2.imwrite(os.path.join(SPLIT_IMAGES_LOCATION, 'transformed_board.jpg'), transformed_board)

split_board(transformed_board)

"""
image_height = corners[2][1] - corners[0][1]
image_width = corners[2][0] - corners[0][0]

print("Height: " + str(image_height))
print("Width: " + str(image_width))

max_height = max(corners[2][1], corners[3][1])
max_width = max(corners[1][0], corners[2][0])

print("Max height: " + str(max_height))
print("Max width: " + str(max_width))

cropped_image = resized[corners[0][1]: max_height, corners[0][0] : max_width]
cv2.imwrite(os.path.join(SPLIT_IMAGES_LOCATION, "cropped_board.jpg"), cropped_image)
split_board(cropped_image)


#plt.imshow(cropped_image)
#plt.show()
"""