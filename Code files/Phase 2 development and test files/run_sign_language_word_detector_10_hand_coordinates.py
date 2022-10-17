# Copy 2 normalizes the cropping to smooth out jumps between adjacent frames.

from ctypes import resize
import cv2
import torch
import numpy as np
import json
import os
from utilities.circular_queue import CircularQueue

from keras.models import model_from_json
import keras
from keras import applications


import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

CLASS_LIST = [
    "before",
    "book",
    "candy",
    "chair",
    "clothes",
    "computer",
    "cousin",
    "drink",
    "go",
    "who",
]


# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5n")

font = cv2.FONT_HERSHEY_SIMPLEX

img_size = 350

model_weights_file = "model_weights_aug10_hand_features.h5"
model_json_file = "model_aug10_hand_features.json"


# load model from JSON file
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into the new model
    loaded_model.load_weights(model_weights_file)
    loaded_model.make_predict_function()


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


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()  # returns camera frames along with bounding boxes and predictions

    def get_frame(self):
        _, fr = self.video.read()

        # crop and resize the image before returning
        results = model(fr)  # identify the person

        dataframe = results.pandas()  # convert to pandas dataframe
        detected_objects = dataframe.xyxy[0].name
        for idx, object in enumerate(detected_objects):
            if object == "person":

                x1 = round(dataframe.xyxy[0].xmin[idx])
                y1 = round(dataframe.xyxy[0].ymin[idx])

                x2 = round(dataframe.xyxy[0].xmax[idx])
                y2 = round(dataframe.xyxy[0].ymax[idx])

                bounding_box = [x1, y1, x2, y2]

        return bounding_box, fr

    def process_clip(self, frames, bounding_boxes):

        # find the largest bounding box
        x1_min, y1_min = np.min(bounding_boxes, axis=0)[0:2]
        x2_max, y2_max = np.max(bounding_boxes, axis=0)[2:4]

        # crop all frames to max and resize
        clip_cropped = frames[:, y1_min:y2_max, x1_min:x2_max, :]
        clip_resized = []
        for fr in clip_cropped:
            clip_resized.append(cv2.resize(fr, (350, 350)))

        clip = np.asarray(clip_resized)

        num_frames = len(frames)

        # initialize the mask and feature arrays
        frame_masks = np.ones(shape=(num_frames), dtype="bool")
        frame_features = np.zeros(shape=(num_frames, NUM_FEATURES), dtype="float32")

        frame_features = extract_hand_coordinates(clip)
        # frame_features = feature_extractor.predict(clip)
        new_clip = []

        return (frame_features, frame_masks, new_clip)


IMG_SIZE = 350
NUM_FEATURES = 126
# build feature extractor

# For static images:
def extract_hand_coordinates(frames):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        # there are 40 points on each hand and 3 coordinates (x,y,z) for each point
        landmarks_list = np.zeros(shape=(len(frames), NUM_FEATURES), dtype="float32")

        for fr_num, image in enumerate(frames):
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks_list_temp = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    for mark in hand_landmarks.landmark:
                        landmarks_list_temp.append(mark.x)
                        landmarks_list_temp.append(mark.y)
                        landmarks_list_temp.append(mark.z)

                landmarks_list[fr_num, : len(landmarks_list_temp)] = landmarks_list_temp
        return landmarks_list


def main(camera):

    predicted_sign = "Nothing"

    # Create the two queues
    q_frames = CircularQueue(50)
    q_boxes = CircularQueue(50)

    while True:
        # Get the next video frame from the camera
        bounding_box, frame = camera.get_frame()

        # add the frame
        q_frames.enQueue(frame)
        q_boxes.enQueue(bounding_box)

        if q_frames.tail == 49:

            # get the bounding boxes and video from the queues
            clip = q_frames.getQueue()
            clip_boxes = q_boxes.getQueue()

            # process the video and crop to largest bounding box in 'clip_box'
            frame_features, frame_mask, new_clip = camera.process_clip(
                np.asarray(clip), np.asarray(clip_boxes)
            )

            # curent_directory = os.getcwd()
            # new_video_path = os.path.join(
            #     curent_directory, "Cropped_video_clip_deaf2.mp4"
            # )
            # write_video_to_file(new_video_path, np.asarray(new_clip))

            # reshape the masks and feature variables
            frame_mask2 = frame_mask[np.newaxis, :]
            frame_features2 = frame_features[np.newaxis, :, :]

            # predict the label of the video
            pred = loaded_model.predict([frame_features2, frame_mask2])[0]

            # Get the string of the predicted label
            predicted_sign = CLASS_LIST[np.argmax(pred)]
            print(predicted_sign)

        cv2.putText(frame, predicted_sign, (25, 25), font, 1, (255, 255, 0), 2)

        cv2.imshow("Sign language recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


main(VideoCamera())
