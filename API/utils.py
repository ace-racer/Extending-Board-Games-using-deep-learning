from itertools import product
import numpy as np
import random
import base64
import cv2

import configurations

cols = list("abcdefgh")
rows = list("12345678")
cartesian_product = list(product(cols, rows))

board_positions = [x[0] + x[1] for x in cartesian_product]

chess_pieces = list("P"*8 + "N"*2 + "R"*2 + "B"*2 + "K" + "Q")

black_pieces = ["b" + x for x in chess_pieces]
white_pieces = ["w" + x for x in chess_pieces]
all_pieces = black_pieces
all_pieces.extend(white_pieces)



def generate_dummy_board(num_pieces=32):
    chosen_positions = random.sample(board_positions, num_pieces)
    chosen_pieces = random.sample(all_pieces, num_pieces)
    return dict(zip(chosen_positions, chosen_pieces))


def simulate_move():
    chosen_positions = random.sample(board_positions, 2)
    chosen_pieces = random.sample(all_pieces, 1)
    moves = {}
    moves[chosen_positions[0]] = chosen_pieces[0]
    moves[chosen_positions[1]] = "empty"
    return moves


def base64_encode_image(image):
    """
    Ref: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/
    Usage: 
        d = {"id": k, "image": base64_encode_image(image)}
        db.set("ID123", json.dumps(d))
    """
    # ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
    image = image.copy(order="C")

    # base64 encode the input NumPy array
    return base64.b64encode(image).decode("utf-8")

def base64_decode_image(image_obj, shape, dtype="uint8"):
    """
    Ref: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/
    Usage: 
        item = json.loads(item.decode("utf-8"))
        image = base64_decode_image(item["image"], (200, 200))
    """

    image = bytes(image_obj, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data type and target shape
    image = np.frombuffer(base64.decodestring(image), dtype=dtype)
    print("decoded image shape")
    print(image.shape)

    num_channels = 3
    required_shape = (shape[0], shape[1], num_channels)
    image = image.reshape(required_shape)
    # resized_image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)

    # return the decoded image
    return image

def downsize_image(img, new_dimensions):
    return cv2.resize(img, new_dimensions, interpolation = cv2.INTER_AREA)

def upsize_image(img, new_dimensions):
    return cv2.resize(img, new_dimensions, interpolation = cv2.INTER_LINEAR)

def convert_to_grayscale_enhance_contrast(image):
    """
    Converts the image to grayscale and then enhances the contrast of the image
    """
    # downsize image
    resized_image = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)

    # convert to gray scale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # perform histogram equalization to enhance the contrast
    gray = cv2.equalizeHist(gray)
    
    # get the histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])

    print(hist.shape)

    # squeeze to get the required dimension
    return np.squeeze(hist)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def process_image_three_class_cnn(image):
    
    resized_image = resize_given_image(image, configurations.CHESS_BLOCK_IMAGE_SIZE)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray)
    #print(edges.shape)
    
    
    # assert(denoised != edges)
    weighted_sum = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
       
    return weighted_sum

def resize_given_image(image, dims):
    if image.shape[0] != dims[0]  or image.shape[1] != dims[1]:
        # dimension is reduced here
        resized_image = cv2.resize(image, dims, interpolation = cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image
