# This script is to open the videos, crop them then save in a folder with the label name.
# This is done to restructure the dataset and allow for video ganerators with data augmentation to tbe used


from ntpath import join
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import utils
import os
import json
from keras import layers
from sklearn.model_selection import train_test_split
from pathlib import Path


DATASET_PATH = "data/"

labels_file_path = "WLASL_v0.3.json"

# Open the json file with the video labels and details
with open(labels_file_path) as ipf:
    content = json.load(ipf)

# create 2 arrays, one with file ID and the other with gloss
labels = []
video_file_names = []
available_videos = os.listdir("data")
box_location_x1y1x2y2 = []  # stores the box made by yolov3 to identify person
frame_start = []
frame_end = []

num_classes = 30

# loop through the first 100 glosses
for ent in content[:num_classes]:

    # loop through all samples for this label
    for inst in ent["instances"]:

        vid_id = inst["video_id"]

        vid_file_name = vid_id + ".mp4"

        if vid_file_name in available_videos:

            # add the gloss to the labels array
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


# prepare the video cropping function
IMG_SIZE = 350
MAX_SEQ_LENGTH = 50  # frame rate is 25, so 50 frames is 2 seconds of video

# Define the crop function
def crop_center_square(frame, box_coordinates):
    x1, y1, x2, y2 = box_coordinates

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
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if counter >= start_end_frames[0]:
                frame = crop_center_square(frame, crop_box)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :]
                frames.append(frame)
                counter = counter + 1

            # -1 indicated the end of the video
            if len(frames) == max_frames or (
                counter >= start_end_frames[1] and start_end_frames[1] > 1
            ):
                break
    finally:
        cap.release()
    return frames


# Main loop
# loop through all uniqu video labels
# Create a folder with using the label name
# for each video with that label
# Open, crop, resize the video
# Save the video in the folder


# Create function to apply to each row of the dataframe.
# Get the path, label, and crop box
# crop and resize the video
# Check if a exists whose name matches the label, if not create it.
# if the folder exists, save the video in this folder.


def process_and_save_video(df_row):
    # accept a row from the main dataframe
    video_path = df_row["video_paths"]
    video_crop_boxe = df_row["box_x1y1x2y2"]
    label = df_row["video_labels"]
    start_end_frms = (df_row["frame_start"], df_row["frame_end"])

    # load the video, also crop and resize
    frames = load_video(
        video_path,
        max_frames=MAX_SEQ_LENGTH,
        resize=(IMG_SIZE, IMG_SIZE),
        crop_box=video_crop_boxe,
        start_end_frames=start_end_frms,
    )

    # Get the current working directory
    curent_directory = os.getcwd()

    # Check if the destination folder apready exists
    new_folder_path = os.path.join(curent_directory, "data_folders2", label)
    try:
        os.mkdir(new_folder_path)
    except:
        print("destination folder already exists")

    # get the video file name
    video_file = os.path.basename(video_path)  # get the video name
    
    # join the file name to the new save path
    new_save_path = os.path.join(new_folder_path, video_file)
    print(f"Saving: {new_save_path}")

    # Set up the video writter. By default the fps or all videos is 25.
    out = cv2.VideoWriter(
        new_save_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (IMG_SIZE, IMG_SIZE)
    )

    # loop through each video frame and write to file.
    for frame in frames:
        out.write(frame)
        # print(np.shape(frames[0]))

    # Release the video writer
    out.release()


# run the main function on each row of the dataframe.
df.apply(process_and_save_video, "columns")
