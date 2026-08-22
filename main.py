import cv2
import mediapipe as mp
import time
import math
import numpy as np

MEME_NAMES = {
    "nerd": "nerdcat.jpg",
    "angry fist": "fist_cat.jpg",
    "shhh" : "shhcat.jpg",
    "peace": "peace_cat.jpg",
    "side eye": "cat-side-eye-cat.png",
    "idle": "idle_cat.jpg",
    "point laugh": "laugh and point .jpg",
    "absolute cinema": "absolute.jpg",
    "rub chin": "ponder.jpg",
    "cat waving": "cat_wave_hello.PNG",
    "hands in scheming": "cat_scheming.jpg",
}

# thresholds for giving certain memes (all positive, we wont care yet which way we turn)
REQUIRED_FRAMES = 10
SIDE_EYE_ANGLE = 20
FINGER_MIN_ANGLE = 155
HAND_NEAR_FACE = 0.08
MIN_MOUTH_RATIO_FOR_OPEN = 0.14
THUMB_NEAR_CHIN_THRESH = 0.14
INDEX_NEAR_CHIN_THRESH = 0.06
DISTANCE_THRESHOLD_FOR_TOUCHING_FINGERS = 0.08
# get a 3x3 rotation matrix we want to extract the angles
def get_rotation_angles(matrix):

    r00 = matrix[0, 0]

    r10 = matrix[1, 0]


    r20 = matrix[2, 0]
    r21 = matrix[2, 1]
    r22 = matrix[2, 2]

    pitch = np.arctan2(
        -r20,
        np.sqrt(r00**2 + r10**2)
    )

    yaw = np.arctan2(
        r10,
        r00
    )

    roll = np.arctan2(
        r21,
        r22
    )

    return (
        np.degrees(yaw),
        np.degrees(pitch),
        np.degrees(roll)
    )
def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )
def angle(a, b, c):
    ba = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(
        np.arccos(cosine)
    )


def is_finger_extended(hand, mcp_index, pip_index, dip_index):
    finger_angle = angle(
        hand[mcp_index],
        hand[pip_index],
        hand[dip_index]
    )

    return finger_angle > FINGER_MIN_ANGLE

def get_finger_states(hand):
    #we wont have thumb extended but we will track it's position
    index = is_finger_extended(hand, 5, 6, 7)
    middle = is_finger_extended(hand, 9, 10, 11)
    ring = is_finger_extended(hand, 13, 14, 15)
    pinky = is_finger_extended(hand, 17, 18, 19)
    thumb = is_finger_extended(hand, 2, 3, 4)
    #might be used later
    # index_angle = angle(hand[5], hand[6], hand[7])
    # middle_angle = angle(hand[9], hand[10], hand[11])
    # ring_angle = angle(hand[13], hand[14], hand[15])
    # pinky_angle = angle(hand[17], hand[18], hand[19])
    print("is thumb out: ", thumb)
    return {
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
        "thumb": thumb,

        "thumb_tip": hand[4],
        "index_tip": hand[8],
        "middle_tip": hand[12],
        "ring_tip": hand[16],
        "pinky_tip": hand[20]
    }
def is_pointing(hand):

    return (
        hand["index"]
        and not hand["middle"]
        and not hand["ring"]
        and not hand["pinky"]
    )
def is_peace(hand):
    return (
        hand["index"]
        and hand["middle"]
        and not hand["ring"]
        and not hand["pinky"]
    )
def is_fist(hand):
    return not (
        hand["index"]
        or hand["middle"]
        or hand["ring"]
        or hand["pinky"]
    )
def is_mouth_open(face):
    return face["mouth_open"] > MIN_MOUTH_RATIO_FOR_OPEN
def side_eye(face):
    return abs(face["pitch"]) > SIDE_EYE_ANGLE

def index_near_face(face, hand):
    dist_of_hand_from_lips = distance(face["upper_lip"], hand["index_tip"])
    return dist_of_hand_from_lips < HAND_NEAR_FACE
def hand_fully_spread(hand):
    #Todo: in the future make it check the distance in the tip to differentiate from spread and fingers close together
    return  (
        hand["index"]
        and hand["middle"]
        and hand["ring"]
        and hand["pinky"]
    )
def finger_are_close(tip1, tip2):
    #same comment as the below function about ordering the hands
    dist = distance(tip1, tip2)
    return dist < DISTANCE_THRESHOLD_FOR_TOUCHING_FINGERS
def hands_in_abs_cinema(hand1, hand2):
    #in reality i should make it pass the left hand and right hand specifically, however for this pose it doesn't matter
    #and I cant be bothered :)
    return hand_fully_spread(hand1) and hand_fully_spread(hand2)
def all_fingers_are_touching(hand1, hand2):
    indx1, indx2 = hand1["index_tip"], hand2["index_tip"]
    midl1, midl2 = hand1["middle_tip"], hand2["middle_tip"]
    ring1, ring2 = hand1["ring_tip"], hand2["ring_tip"]
    pinky1, pinky2 = hand1["pinky_tip"], hand2["pinky_tip"]
    print("idx dist: ", distance(indx1, indx2))
    return (finger_are_close(indx1, indx2) and finger_are_close(midl1, midl2) and
            finger_are_close(ring1, ring2) and finger_are_close(pinky1, pinky2))
def hand_rubs_chin(face, hand):
    thumb = hand["thumb_tip"]
    index = hand["index_tip"]
    chin = face["chin_cords"]
    chin_to_index_dist = distance(chin, index)
    chin_to_thumb_dist = distance(chin, thumb)
    print("chin to indx: ", chin_to_index_dist, "chin to thumb: ", chin_to_thumb_dist)
    return chin_to_thumb_dist < THUMB_NEAR_CHIN_THRESH and chin_to_index_dist < INDEX_NEAR_CHIN_THRESH
##################################################################################
##### ---------- this is the main function for classifying memes  ---------- #####
##################################################################################
def classify_whole_features(features):
    face = features["face"]
    if not features["hand_present"]:
        if side_eye(face):
            return MEME_NAMES["side eye"]
        return MEME_NAMES["idle"]
    hand_array = features["hand"]
    if len(hand_array) == 2:
        if all_fingers_are_touching(hand_array[0], hand_array[1]):
            return MEME_NAMES["hands in scheming"]
        if hands_in_abs_cinema(hand_array[0], hand_array[1]):
            return MEME_NAMES["absolute cinema"]
        return MEME_NAMES["idle"]
    #if only one present we wont care which one it is (yet)
    hand = hand_array[0]

    #categorized based on order-sensitive poses
    if hand_rubs_chin(face, hand):
        return MEME_NAMES["rub chin"]
    if is_pointing(hand) and index_near_face(face, hand):
        return MEME_NAMES["shhh"]

    if (is_pointing(hand) or is_fist(hand)) and is_mouth_open(face):
        return MEME_NAMES["point laugh"]
    if is_pointing(hand) and not index_near_face(face, hand):
        return MEME_NAMES["nerd"]
    if is_fist(hand) and not side_eye(face):
        return MEME_NAMES["angry fist"]
    if is_peace(hand):
        return MEME_NAMES["peace"]
    if hand_fully_spread(hand):
        return MEME_NAMES["cat waving"]
    return MEME_NAMES["idle"]
def show_meme(meme):
    image = cv2.imread("memes/"+meme)

    if image is None:
        print(f"Could not load image: {meme}")
        return

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    cv2.imshow(IMAGE_WINDOW, image)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerTask = mp.tasks.vision.FaceLandmarker
HandLandmarkerTask = mp.tasks.vision.HandLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#meme window setup
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

IMAGE_WINDOW = "Meme"

cv2.namedWindow(
    IMAGE_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    IMAGE_WINDOW,
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

faceoptions = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    output_facial_transformation_matrixes=True
)
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

FaceLandmarker = FaceLandmarkerTask.create_from_options(faceoptions)
HandLandmarker = HandLandmarkerTask.create_from_options(hand_options)
video=cv2.VideoCapture(0)
ms=0


last_meme = None
meme_frames = 0
displayed_meme = None

#####################################################################################################
##### ---------- this is the main loop for putting the image and grabbing landmarks  ---------- #####
#####################################################################################################
while True:
    # basically boilerplate
    ret, image = video.read()
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = FaceLandmarker.detect_for_video(
        img,
        ms
    )
    hand_result = HandLandmarker.detect_for_video(
        img,
        ms
    )
    #we will only show memes if there is face due to the fact that everything is built on making the face :)
    if result.face_landmarks:
        #grabbing face landmarks,
        face  = result.face_landmarks[0]
        upper_lip = face[13]
        lower_lip = face[14]

        right_cheek = face[234]
        left_cheek = face[454]
        chin = face[152]
        mouth_opening = distance(
            upper_lip,
            lower_lip
        )
        face_width = distance(
            right_cheek,
            left_cheek
        )
        # will be used when doing tounge stuff
        mouth_ratio = mouth_opening / face_width

        rotation_matrix = result.facial_transformation_matrixes[0]
        # yaw rotating in an o shape, pitch rotate left to right, roll rotate up down
        # positive: yaw is right, pitch left, roll down
        yaw, pitch, roll = get_rotation_angles(rotation_matrix)
        face_features = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "mouth_open": mouth_ratio,
            "upper_lip": upper_lip,
            "chin_cords": chin,
        }
        # so pycharm wont be mad :)
        features = {}
        #check for extended index next to face
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            handedness = hand_result.handedness[0]
            # important points, 1-4-thumb tip (last num is tip), 5-8 - index tip, 9-12- middle tip, 13-16 ring tip, 17-20 pinky tip
            #which hand (doesnt really matter but might be useful), can also check which fingers are present by get_finger_state
            #print("hand:", index_tip, "which hand:", hand_result.handedness[0][0].category_name)
            #we'll calculate the distance of the index tip to the face if present
            fingers = get_finger_states(hand)
            features = {
                "face": face_features,
                "hand_present": True,
                "hand": [fingers | {"hand_name": handedness[0].category_name }]
            }
            if len(hand_result.hand_landmarks) >= 2:
                features["hand"].append(get_finger_states(hand_result.hand_landmarks[1])
                                        | {"hand_name":  hand_result.handedness[1][0].category_name})
        else:
            #put all the values we care about in one place
            features = {
                "face": face_features,
                "hand_present": False
            }
        meme = classify_whole_features(features)

        if meme == last_meme:
            meme_frames += 1
        else:
            last_meme = meme
            meme_frames = 1

        if meme_frames >= REQUIRED_FRAMES and meme != displayed_meme:
            displayed_meme = meme

        if displayed_meme is not None:
            show_meme(displayed_meme)
    else:
        print("NO FACE")
    ms += 33
    cv2.imshow("da face", image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

FaceLandmarker.close()
HandLandmarker.close()

