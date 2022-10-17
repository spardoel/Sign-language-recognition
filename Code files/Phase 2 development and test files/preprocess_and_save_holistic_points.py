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
DATASET_PATH = "augmented_100/"

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

# use scikit learn's train test split function to generate testing data
intermediate_df, test_df = train_test_split(
    df, train_size=train_split, shuffle=True, random_state=123
)
# from the remaining data, generate the training and validation sets
train_df, valid_df = train_test_split(
    intermediate_df, train_size=1 - val_split, shuffle=True, random_state=123
)

# Print the number of samples in each set to confirm the intended proportions
print(
    f"Training samples: {len(train_df)}, Test samples: {len(test_df)}, Validation samples: {len(valid_df)}"
)


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


def extract_holistic_coordinates(frames):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        landmarks_list = np.zeros(shape=(len(frames), NUM_FEATURES), dtype="float32")

        for fr_num, image in enumerate(frames):
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # check each category of landmarks and assign to correct location in landmark list
            # add the landmarks to the correct place in the landmark list.
            # left hand landmarks indexes 0 - 62,
            # right hand landmarks indexes 63 - 125,
            # pose landmarks indexes 126 - 169

            # check the left hand points
            left_hand_temp = []
            if results.left_hand_landmarks:
                for mark in results.left_hand_landmarks.landmark:
                    left_hand_temp.append(mark.x)
                    left_hand_temp.append(mark.y)
                    left_hand_temp.append(mark.z)

                landmarks_list[fr_num, : len(left_hand_temp)] = left_hand_temp

            # Check the right hand points
            right_hand_temp = []
            if results.right_hand_landmarks:
                for mark in results.right_hand_landmarks.landmark:
                    right_hand_temp.append(mark.x)
                    right_hand_temp.append(mark.y)
                    right_hand_temp.append(mark.z)

                landmarks_list[fr_num, 63 : 63 + len(right_hand_temp)] = right_hand_temp

            # check the pose points 0-22. Points 23-33 are for the waist and legs, they are not needed
            pose_temp = []
            if results.pose_landmarks:
                for mark in results.pose_landmarks.landmark[:23]:
                    if mark.visibility > 0.5:
                        pose_temp.append(mark.x)
                        pose_temp.append(mark.y)
                    else:
                        pose_temp.append(0)
                        pose_temp.append(0)
                        # pad with zeros if the point is not visible

                landmarks_list[fr_num, 126 : 126 + len(pose_temp)] = pose_temp

        return landmarks_list


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

        frame_features[idx, :length, :] = extract_holistic_coordinates(frames)
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
test_data, test_labels = prepare_all_videos(test_df)
print("Test set finished")


# Saving the objects:
with open(
    "preprocessed_videos_augmented100_holistic_coordinates.pkl", "wb"
) as f:  # Python 3: open(..., 'wb')
    pickle.dump(
        [
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
            class_vocab,
        ],
        f,
    )
