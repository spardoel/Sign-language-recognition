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

# augmented10 contains the pre-cropped and pre-sized videos of 10 words
DATASET_PATH = "augmented10/"

# create 2 arrays, one with file ID and the other with gloss (i.e., class label)
video_labels = []
video_paths = []


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

IMG_SIZE = 350
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 1280


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
    return np.asarray(frames).astype(dtype=np.int32)


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["video_labels"])
)
class_vocab = label_processor.get_vocabulary()

# build feature extractor
def build_feature_extractor():
    feature_extractor = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.efficientnet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")
    # build feature extractor


feature_extractor = build_feature_extractor()


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data_2 = np.squeeze(data)
    plt.imshow(data_2[5, :, :, :])
    plt.show()
    # plt.waitforbuttonpress()


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
        # Gather all its frames and add a batch dimension.
        frames = load_video(
            path,
            max_frames=MAX_SEQ_LENGTH,
        )
        frames = frames[None, ...]

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
        for i, batch in enumerate(frames):
            print(i)
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


print("Extracting features")
train_data, train_labels = prepare_all_videos(train_df)
val_data, val_labels = prepare_all_videos(valid_df)
test_data, test_labels = prepare_all_videos(test_df)


# Saving the objects:
with open(
    "preprocessed_videos_augmented10.pkl", "wb"
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
