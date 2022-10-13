# V2 uses augmented videos in folders corresponding to the classes.
# this means the json file is no longer needed

import imp
import cv2
import os
import pickle
import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import applications

from holistic_feature_extractor import HolisticFeatureExtractor
from holistic_feature_data import HolisticData

feature_extractor = HolisticFeatureExtractor()
data = HolisticData()


import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# augmented10 contains the pre-cropped and pre-sized videos of 10 words
DATASET_PATH = "augmented_100_HO_cropped/"

# create 2 arrays, one with file ID and the other with gloss (i.e., class label)
video_labels = []
video_paths = []

IMG_SIZE = 350
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 172  # (126 hands, 46 pose)
# hands, 21 each with (x,y,z)
# pose, 33 each with (x,y). Can use visibility threshold to include or not.
# 21 * 2 * 3 = 126. 23 * 2 = 46. 126 + 46 = 172

# Check each folder in the dataset. Each one corresponds to a gloss (i.e., class label, aka category)
for category in os.listdir(DATASET_PATH):

    # for each category folder create folder path
    category_path = os.path.join(DATASET_PATH, category)

    # list the videos in the folder
    samples = os.listdir(category_path)

    # for each video in the category subset
    for video in samples:

        # Add the image name to the path and append the image path and label to their respective lists
        video_paths.append(os.path.join(category_path, video))
        video_labels.append(category)


# combine the lists into a pandas dataframe. First convert to pandas Series and then concatenate.
df = pd.concat(
    [
        pd.Series(video_paths, name="video_paths"),
        pd.Series(video_labels, name="video_labels"),
    ],
    axis=1,
)


# count the times each label (category) appears
print(df["video_labels"].unique())
print(df["video_labels"].value_counts())


# use 80% of the data for training. Within that training set, use 25% for validation
train_split = 0.9
val_split = 0.25

# # use scikit learn's train test split function to generate testing data
# intermediate_df, test_df = train_test_split(
#     df, train_size=train_split, shuffle=True, random_state=123
# )
# # from the remaining data, generate the training and validation sets
# train_df, valid_df = train_test_split(
#     intermediate_df, train_size=1 - val_split, shuffle=True, random_state=123
# )

# use scikit learn's train test split function to generate testing data
train_df, valid_df = train_test_split(
    df, train_size=train_split, shuffle=True, random_state=123
)

# Print the number of samples in each set to confirm the intended proportions
print(f"Training samples: {len(train_df)}, Validation samples: {len(valid_df)}")


##----------------------------------------------

# https://keras.io/examples/vision/video_classification/

# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub


def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, :]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return frames


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["video_labels"])
)
class_vocab = label_processor.get_vocabulary()


# For static images:
def draw_points_on_video(frames):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
        refine_face_landmarks=False,
    ) as holistic:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        new_frames = []
        for image in frames:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            new_frames.append(image)

        return new_frames


# For static images:
def extract_holistic_coordinates_from_video(frames):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        # there are 40 points on each hand and 3 coordinates (x,y,z) for each point
        landmarks_list = np.zeros(shape=(len(frames), NUM_FEATURES), dtype="float32")
        left_hand_normalized = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        right_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        poses = pd.DataFrame(np.zeros((23, 2)), columns=["x", "y"])
        bounding_box = np.zeros((len(frames), 4))
        features_list_temp = []

        for fr_num, image in enumerate(frames):
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Get the size of the frame
            y_size, x_size, _ = image.shape

            # check each category of landmarks and assign to correct location in landmark list
            # add the landmarks to the correct place in the landmark list.
            # left hand landmarks indexes 0 - 62,
            # right hand landmarks indexes 63 - 125,
            # pose landmarks indexes 126 - 169

            if results.left_hand_landmarks:
                for i, mark in enumerate(results.left_hand_landmarks.landmark):
                    left_hand_normalized.x[i] = mark.x
                    left_hand_normalized.y[i] = mark.y
                    left_hand_normalized.z[i] = mark.z

            if results.left_hand_landmarks:
                for i, mark in enumerate(results.left_hand_landmarks.landmark):
                    left_hand.x[i] = mark.x * x_size
                    left_hand.y[i] = mark.y * y_size
                    left_hand.z[i] = mark.z * x_size

            # right_hand.
            if results.right_hand_landmarks:
                for j, mark in enumerate(results.right_hand_landmarks.landmark):
                    right_hand.x[j] = mark.x * x_size
                    right_hand.y[j] = mark.y * y_size
                    right_hand.z[j] = mark.z * x_size

            # check the pose points 0-22. Points 23-33 are for the waist and legs, they are not needed

            if results.pose_landmarks:
                for k, mark in enumerate(results.pose_landmarks.landmark[:23]):
                    if mark.visibility > 0.5:
                        poses.x[k] = mark.x * x_size
                        poses.y[k] = mark.y * y_size

            ##-----------------
            # get the min and max values
            # must convert to dataframes to ensure that nan are not returned
            x_min = pd.DataFrame(
                {
                    left_hand.x[left_hand.x > 0].min(),
                    right_hand.x[right_hand.x > 0].min(),
                    poses.x[poses.x > 0].min(),
                }
            ).min()[0]

            x_max = pd.DataFrame(
                {left_hand.x.max(), right_hand.x.max(), poses.x.max()}
            ).max()[0]

            y_min = pd.DataFrame(
                {
                    left_hand.y[left_hand.y > 0].min(),
                    right_hand.y[right_hand.y > 0].min(),
                    poses.y[poses.y > 0].min(),
                }
            ).min()[0]

            y_max = pd.DataFrame(
                {left_hand.y.max(), right_hand.y.max(), poses.y.max()}
            ).max()[0]

            # the bounding box for this frame is defined as
            bounding_box[fr_num][:] = [x_min, y_min, x_max, y_max]

            LH_temp = left_hand.to_numpy().flatten()
            RH_temp = right_hand.to_numpy().flatten()
            Pose_temp = poses.to_numpy().flatten()

            # print(left_hand_normalized.to_numpy().flatten())

            features_list_temp.append([left_hand, right_hand, poses])

        # -------------------------------------

        # Crop resize the features to match the cropped frame

        # ---------------------------------------

        # After looping through all video frames

        # find the largest bounding box
        x1_min, y1_min = np.round(np.min(bounding_box, axis=0)[0:2])
        x2_max, y2_max = np.round(np.max(bounding_box, axis=0)[2:4])
        new_size_x = x2_max - x1_min
        new_size_y = y2_max - y1_min

        # check for limits
        x1_min = 0 if x1_min < 0 or np.isnan(x1_min) else x1_min
        y1_min = 0 if y1_min < 0 or np.isnan(y1_min) else y1_min

        x2_max = frames[0].shape[1] if x2_max > frames[0].shape[1] else x2_max
        y2_max = frames[0].shape[0] if y2_max > frames[0].shape[0] else y2_max

        for fr_idx in range(len(frames)):

            # Unpack the landmarks to separate variables
            left_hand = features_list_temp[fr_idx][0]
            right_hand = features_list_temp[fr_idx][1]
            poses = features_list_temp[fr_idx][2]

            # Left hand
            left_hand.x[left_hand.x > 0] = (
                left_hand.x[left_hand.x > 0] - x1_min
            ) / new_size_x

            left_hand.y[left_hand.y > 0] = (
                left_hand.y[left_hand.y > 0] - y1_min
            ) / new_size_y

            # left_hand.z[left_hand.z > 0] = (
            #     left_hand.z[left_hand.z > 0] - x1_min
            # ) / new_size_x

            # Right hand
            right_hand.x[right_hand.x > 0] = (
                right_hand.x[right_hand.x > 0] - x1_min
            ) / new_size_x
            right_hand.y[right_hand.y > 0] = (
                right_hand.y[right_hand.y > 0] - y1_min
            ) / new_size_y
            # right_hand.z[right_hand.z > 0] = (
            #     right_hand.z[right_hand.z > 0] - x1_min
            # ) / new_size_x

            # Pose coordinates
            poses.x[poses.x > 0] = poses.x[poses.x > 0] - x1_min
            # / new_size_x
            poses.y[poses.y > 0] = (poses.y[poses.y > 0] - y1_min) / new_size_y

            LH_temp = left_hand.to_numpy().flatten()
            RH_temp = right_hand.to_numpy().flatten()
            Pose_temp = poses.to_numpy().flatten()

            # print(LH_temp)

            landmarks_list[fr_idx] = np.concatenate(
                (LH_temp, RH_temp, Pose_temp), axis=0
            )

        return landmarks_list


def extract_holistic_coordinates(image, holistic, left_hand, right_hand, poses):

    # there are 40 points on each hand and 3 coordinates (x,y,z) for each point
    # landmarks_list = np.zeros(shape=(len(frames), NUM_FEATURES), dtype="float32")

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Get the size of the frame
    y_size, x_size, _ = image.shape

    if results.left_hand_landmarks:
        for i, mark in enumerate(results.left_hand_landmarks.landmark):
            left_hand.x[i] = mark.x * x_size
            left_hand.y[i] = mark.y * y_size
            left_hand.z[i] = mark.z * x_size
            # print(mark.z)

    # right_hand.
    if results.right_hand_landmarks:
        for j, mark in enumerate(results.right_hand_landmarks.landmark):
            right_hand.x[j] = mark.x * x_size
            right_hand.y[j] = mark.y * y_size
            right_hand.z[j] = mark.z * x_size

    # check the pose points 0-22. Points 23-33 are for the waist and legs, they are not needed

    if results.pose_landmarks:
        for k, mark in enumerate(results.pose_landmarks.landmark[:23]):
            if mark.visibility > 0.5:
                poses.x[k] = mark.x * x_size
                poses.y[k] = mark.y * y_size
    # get the min and max values
    # must convert to dataframes to ensure that nan are not returned
    x_min = pd.DataFrame(
        {
            left_hand.x[left_hand.x > 0].min(),
            right_hand.x[right_hand.x > 0].min(),
            poses.x[poses.x > 0].min(),
        }
    ).min()[0]

    x_max = pd.DataFrame({left_hand.x.max(), right_hand.x.max(), poses.x.max()}).max()[
        0
    ]

    y_min = pd.DataFrame(
        {
            left_hand.y[left_hand.y > 0].min(),
            right_hand.y[right_hand.y > 0].min(),
            poses.y[poses.y > 0].min(),
        }
    ).min()[0]

    y_max = pd.DataFrame({left_hand.y.max(), right_hand.y.max(), poses.y.max()}).max()[
        0
    ]

    # the bounding box for this frame is defined as
    bounding_box = [x_min, y_min, x_max, y_max]

    return bounding_box, image, left_hand, right_hand, poses


def prepare_all_videos(df):
    num_samples = len(df)
    video_paths = df["video_paths"].values.tolist()
    labels = df["video_labels"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to the sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.

    # initialize the mask and feature arrays
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.

    for idx, path in enumerate(video_paths):

        print(f"Processing video {idx} / {len(video_paths)}")
        # Gather all its frames and add a batch dimension.
        frames = load_video(
            path,
            max_frames=MAX_SEQ_LENGTH,
        )
        # frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        video_length = len(frames)
        length = min(MAX_SEQ_LENGTH, video_length)
        # ------ Save the video
        # video_w_points = draw_points_on_video(frames)
        # curent_directory = os.getcwd()
        # new_video_path = os.path.join(curent_directory, "Holistic_tracking_video1.mp4")
        # write_video_to_file(new_video_path, np.asarray(video_w_points))
        # ------

        frame_features[
            idx, :length, :
        ] = feature_extractor.run_feature_extraction_on_clip(np.asarray(frames))

        # frame_features[idx, :length, :] = extract_holistic_coordinates_from_video(
        #     frames
        # )

        # for j in range(length):
        #     temp_frame_features[1, j, :] = feature_extractor.predict(frames[None, j, :])
        temp_frame_mask[0, :length] = 1  # 1 = not masked, 0 = masked

        # frame_features[
        #     idx,
        # ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


def write_video_to_file(save_path, frames):

    print(f"Saving: {save_path}")
    # by default the fps or all videos is 25
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (350, 350),  # assuming a quare image
    )

    for frame in frames:
        out.write(np.asarray(frame))

    out.release()


print("Extracting features")
train_data, train_labels = prepare_all_videos(train_df)
print("Training set finished")
val_data, val_labels = prepare_all_videos(valid_df)
print("Validation set finished")
# test_data, test_labels = prepare_all_videos(test_df)
# print("Test set finished")


# Saving the objects:
with open(
    "preprocessed_videos_10_holistic_coordinates_cropped_OOP2.pkl", "wb"
) as f:  # Python 3: open(..., 'wb')
    pickle.dump(
        [
            train_data,
            train_labels,
            val_data,
            val_labels,
            class_vocab,
        ],
        f,
    )
