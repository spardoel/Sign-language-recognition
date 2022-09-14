import json
import os
import cv2
import numpy as np
import pandas as pd
import keras
import pickle

from sklearn.model_selection import train_test_split
from keras import applications

# the path to the dataset
DATASET_PATH = "data/"

# the json file with the dataset details
labels_file_path = "WLASL_v0.3.json"

# open the json file and load its content
with open(labels_file_path) as ipf:
    content = json.load(ipf)


# create the empty variables
labels = []  # for the gloss, aka video label
video_file_names = []  # for the video name, i.e. identification number
box_location_x1y1x2y2 = []  # stores the box made by yolov3 to identify person
frame_start = []  # the video frame corresponding to the sign start
frame_end = []  # the video frame corresponding to the sign end (-1 means end of video)

# list all videos in the dataset
available_videos = os.listdir(DATASET_PATH)

# choose the number of labels to use
num_classes = 15

# loop through the first x glosses in the json file
for ent in content[:num_classes]:

    # loop through all samples in the json file for this label
    for inst in ent["instances"]:

        # construct the video name from the json file
        vid_id = inst["video_id"]
        vid_file_name = vid_id + ".mp4"

        # check if the video is in the dataset folder
        if vid_file_name in available_videos:

            # if the video is in the dataset, save the desired details
            labels.append(ent["gloss"])
            video_file_names.append(os.path.join(DATASET_PATH, vid_file_name))
            box_location_x1y1x2y2.append(inst["bbox"])
            frame_start.append(inst["frame_start"])
            frame_end.append(inst["frame_end"])


# combine the lists into a pandas dataframe. First convert to pandas Series and then concatenate.
df = pd.concat(
    [
        pd.Series(video_file_names, name="video_paths"),
        pd.Series(labels, name="video_labels"),
        pd.Series(box_location_x1y1x2y2, name="box_x1y1x2y2"),
        pd.Series(frame_start, name="frame_start"),
        pd.Series(frame_end, name="frame_end"),
    ],
    axis=1,
)

# count the times each label (category) appears
print(df["video_labels"].value_counts())


## ---------- Train test split ------------------


# use 80% of the data for training. Within that training set, use 20% for validation and 20% for testing.
train_split = 0.8
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


##------------------------- Prepare videos -------------------

# https://keras.io/examples/vision/video_classification/

# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

# prepare the video cropping function
IMG_SIZE = 350
MAX_SEQ_LENGTH = 50  # frame rate is 25, so 50 frames is 2 seconds of video

# Define the crop function
def crop_frame(frame, box_coordinates):
    x1, y1, x2, y2 = box_coordinates  # unpack the coordinates

    return frame[y1:y2, x1:x2]


# define the function that loads the video from the provided path then crops and resizes the video.
def load_video(
    path,
    max_frames=0,
    resize=(IMG_SIZE, IMG_SIZE),
    crop_box=(0, 0, 0, 0),
    start_end_frames=(1, -1),
):
    cap = cv2.VideoCapture(path)

    frames = []
    counter = 1
    # use try / finally to automatically close the VideoCapture
    try:
        while True:
            # while new frames are available, load the next frame
            ret, frame = cap.read()

            if not ret:
                # if no frames are available break the loop
                break

            # if the current frame number is larger than the start frame
            if counter >= start_end_frames[0]:
                # prepare the frame
                frame = crop_frame(frame, crop_box)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :]
                # add the frame to the list
                frames.append(frame)
                counter += 1  # increment the frame counter

            # if the length of 'frames' is equal to the desired max number of frames break the loop.
            # Also break the loop with the frame counter is larger than the end of the desired clip, but only if the end of the desired clip is larger than -1
            # -1 indicated the end of the video, so only stop the loop early if the sign clip ends before the end of the video
            if len(frames) == max_frames or (
                counter >= start_end_frames[1] and start_end_frames[1] > 1
            ):
                break
    finally:
        cap.release()

    # return video as a numpy array
    return np.asarray(frames).astype(dtype=np.int32)


# build feature extractor
def build_feature_extractor():
    # Select the pre-trained model to used form the Keras applications
    feature_extractor = applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = applications.efficientnet.preprocess_input

    # define the input size
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))

    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    # return the feature extraction model
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
NUM_FEATURES = 1280


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["video_labels"])
)

class_vocab = label_processor.get_vocabulary()


def prepare_all_videos(df):
    num_samples = len(df)
    video_paths = df["video_paths"].values.tolist()
    video_crop_boxes = df["box_x1y1x2y2"].values.tolist()
    labels = df["video_labels"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed the sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.

    # initialise the mask and features arrays
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
            resize=(IMG_SIZE, IMG_SIZE),
            crop_box=video_crop_boxes[idx],
        )
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):

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
with open("preprocessed_videos.pkl", "wb") as f:
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
