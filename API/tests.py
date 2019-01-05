from redisprovider import RedisProvider
import utils
import cv2
import json

def serialize_and_insert_image(full_image_location):
    image = cv2.imread(full_image_location)
    gameid = "12343"
    details_to_store = {"gameid": gameid, "image": utils.base64_encode_image(image)}
    details_to_store_json = json.dumps(details_to_store)
    r = RedisProvider()
    r.set_value_in_redis(gameid, details_to_store_json)
    print("Image inserted")
    return gameid

def retrieve_deserialize_image(key):
    r = RedisProvider()
    stored_details = r.get_value_in_redis(key)
    stored_details = json.loads(stored_details)
    image = stored_details["image"]
    image = utils.base64_decode_image(image, (1200, 1200))
    # cv2.imshow("Image", image)
    cv2.imwrite("H:\\AR-ExtendingOnlineGames\\data\\chess_board\\Chess board images\\board1.jpg", image)

if __name__ == "__main__":
    #key = serialize_and_insert_image("H:\\AR-ExtendingOnlineGames\\data\\chess_board\\image-7.png")
    key = "12343"
    retrieve_deserialize_image(key)
    

