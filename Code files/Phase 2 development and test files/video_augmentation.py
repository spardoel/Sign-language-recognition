# https://github.com/okankop/vidaug/blob/master/README.md


import cv2
import os
import random
import numpy as np
from PIL import Image
from vidaug import augmentors as va


def get_augmentor(augmentation):

    if augmentation == "HorizontalFlip":
        seq = va.Sequential([va.HorizontalFlip()])
    elif augmentation == "Rotate":
        seq = va.Sequential([va.RandomRotate(degrees=10)])
    elif augmentation == "Translate":
        seq = va.Sequential([va.RandomTranslate(x=60, y=60)])
    elif augmentation == "Add":
        add_amount = random.randint(10, 60)
        seq = va.Sequential([va.Add(add_amount)])
    elif augmentation == "Subtract":
        add_amount = random.randint(-60, -10)
        seq = va.Sequential([va.Add(add_amount)])
    elif augmentation == "Salt":
        salt_amount = random.randint(75, 120)
        seq = va.Sequential([va.Salt(salt_amount)])
    elif augmentation == "Pepper":
        salt_amount = random.randint(75, 120)
        seq = va.Sequential([va.Pepper(salt_amount)])

    return seq


DATASET_LOCATION = "data_folder_100_not_cropped/"

dataset_path = os.listdir(DATASET_LOCATION)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            PIL_image = Image.fromarray(frame)
            # must use the PIL Image format. Otherwise the image rotation doesn't work and corrupts the video.
            # (the flip works with numpy arrays though)

            frames.append(PIL_image)

    finally:
        cap.release()
    return frames


def create_save_path(input_path, folder, suffix=""):
    # get current directory
    curent_directory = os.getcwd()

    # Check if the destination folder apready exists
    new_folder_path = os.path.join(curent_directory, "augmented_100_HO_cropped", folder)
    try:
        os.mkdir(new_folder_path)
    except:
        print("destination folder already exists")

    # get the name of the video file
    video_file = os.path.basename(input_path)

    # Add the transformation type to the name
    new_save_path = os.path.join(
        new_folder_path, video_file[:-4] + "_" + suffix + ".mp4"
    )

    return new_save_path


def write_video_to_file(save_path, frames):
    width = frames[0].width
    height = frames[0].height

    print(f"Saving: {save_path}")
    # by default the fps or all videos is 25
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (width, height),  # assuming a quare image
    )

    for frame in frames:
        out.write(np.asarray(frame))

    out.release()


transformations = [
    "HorizontalFlip",
    "Rotate",
    "Translate",
    "Add",
    "Subtract",
    "Salt",
    "Pepper",
]


for class_folder in dataset_path:
    class_path = os.path.join(DATASET_LOCATION, class_folder)

    videos_in_class = os.listdir(class_path)

    for video_file in videos_in_class:
        video_path = os.path.join(class_path, video_file)

        video = load_video(video_path)

        # for each type of augmentation

        for augment in transformations:

            seq = get_augmentor(augment)

            augmented_video = seq(video)

            save_path = create_save_path(video_path, class_folder, suffix=augment)

            write_video_to_file(save_path, augmented_video)
