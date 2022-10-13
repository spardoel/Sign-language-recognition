# this class is used to extract the holistic features from videos.

from random import random
from re import S
from tracemalloc import start
import pandas as pd
import numpy as np

import cv2
import os
import mediapipe as mp

from time import time

from utilities.holistic_feature_data import HolisticData

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


MAX_INPUT_FRAME_WIDTH = 1920
MAX_INPUT_FRAME_HEIGHT = 1080


class HolisticFeatureExtractor:
    def __init__(self):

        # create the holistic feature extraction model
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        self.right_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        self.pose = pd.DataFrame(np.zeros((23, 2)), columns=["x", "y"])

        self.bounding_box = pd.DataFrame(
            np.zeros((1, 4)), columns=["x_min", "y_min", "x_max", "y_max"]
        ).astype(int)

        self.frame_width = 350
        self.frame_height = 350
        self.frame_width = 640
        self.frame_height = 480
        self.num_features = 172

        self.max_video_length = 50

    def run_feature_extractor_single_frame(self, image, data_structure):

        # reset the variables to zeros
        self.left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        self.right_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
        self.pose = pd.DataFrame(np.zeros((23, 2)), columns=["x", "y"])

        # get the shape of the input image
        self.input_image_size_y, self.input_image_size_x, _ = image.shape

        # run the feature extractor model on the image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # unpack the results
        self.__extract_coordinates_from_results(results)
        self.__get_single_frame_bounding_box()

        # optionally, draw the landmark coordinates on the image
        # image = self.__draw_all_points_on_image(image, results)

        # add the frame data to the queues in the Data() class
        data_structure.add_new_frame(
            image, self.bounding_box, self.left_hand, self.right_hand, self.pose
        )

        return image

    def run_feature_extraction_on_clip(self, video):

        data_structure = HolisticData()
        new_clip = []
        for frame in video:
            # process the frame
            self.run_feature_extractor_single_frame(frame, data_structure)

        # process the video and crop to largest bounding box in 'clip_box'
        frame_features, _, _ = self.process_clip(data_structure)

        return frame_features

    def process_clip(self, data_structure):

        (
            frames,
            boxes,
            LH_coordinates,
            RH_coordinates,
            P_coordinates,
        ) = data_structure.get_clip()

        # Find the bounding box min and max values
        self.__get_bounding_box_min_max(boxes)
        self.__check_bounding_box_validity(frames[0])

        # Crop the clip to the bounding box
        clip_cropped = frames[
            :,
            int(self.bounding_box.y_min[0]) : int(self.bounding_box.y_max[0]),
            int(self.bounding_box.x_min[0]) : int(self.bounding_box.x_max[0]),
            :,
        ]

        # define the output variables
        self.output_features = np.zeros((self.max_video_length, self.num_features))
        ouput_clip = []

        # loop through the frames
        for fr_idx, frame in enumerate(clip_cropped):

            frame = cv2.resize(frame, (350, 350))

            # set the variables to zeros to prevent cary over from previous frame
            self.left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
            self.right_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])
            self.pose = pd.DataFrame(np.zeros((23, 2)), columns=["x", "y"])

            # Re-normalize the features
            self.__map_coordinates_to_resized_image(
                fr_idx,
                LH_coordinates[fr_idx],
                RH_coordinates[fr_idx],
                P_coordinates[fr_idx],
            )

            ouput_clip.append(frame)

        # create the mask output variable
        frame_masks = np.zeros(shape=(self.max_video_length), dtype="bool")
        frame_masks[: len(frames)] = 1  # 1 = not masked, 0 = masked

        # reset the bounding box to zeros after processing the clip.
        self.bounding_box = pd.DataFrame(
            np.zeros((1, 4)), columns=["x_min", "y_min", "x_max", "y_max"]
        ).astype(int)

        return self.output_features, frame_masks, ouput_clip

    def __extract_coordinates_from_results(self, results):

        # pulls the desired coordinates values out of the Results variable retunred by the holistic model.

        # converts the normalized coordinate values to pixel values by multiplying by the iamge size
        if results.left_hand_landmarks:

            LH_x = np.asarray(self.left_hand.x)
            LH_y = np.asarray(self.left_hand.y)
            LH_z = np.asarray(self.left_hand.z)
            for i, mark in enumerate(results.left_hand_landmarks.landmark):
                LH_x[i] = mark.x * self.input_image_size_x
                LH_y[i] = mark.y * self.input_image_size_y
                LH_z[i] = mark.z * self.input_image_size_x
                # print(f"point after calculation {self.left_hand.x[i]}")
            self.left_hand.x[:] = LH_x[:]
            self.left_hand.y[:] = LH_y[:]
            self.left_hand.z[:] = LH_z[:]
        else:  # pad with zero
            self.left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])

        # right_hand.
        if results.right_hand_landmarks:
            RH_x = np.asarray(self.right_hand.x)
            RH_y = np.asarray(self.right_hand.y)
            RH_z = np.asarray(self.right_hand.z)

            for j, mark in enumerate(results.right_hand_landmarks.landmark):
                RH_x[j] = mark.x * self.input_image_size_x
                RH_y[j] = mark.y * self.input_image_size_y
                RH_z[j] = mark.z * self.input_image_size_x

            self.right_hand.x[:] = RH_x[:]
            self.right_hand.y[:] = RH_y[:]
            self.right_hand.z[:] = RH_z[:]
        else:
            self.right_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])

        # check the pose points 0-22. Points 23-33 are for the waist and legs, they are not needed

        if results.pose_landmarks:

            P_x = np.asarray(self.pose.x)
            P_y = np.asarray(self.pose.y)

            for k, mark in enumerate(results.pose_landmarks.landmark[:23]):
                if mark.visibility > 0.5:
                    self.pose.x[k] = mark.x * self.input_image_size_x
                    self.pose.y[k] = mark.y * self.input_image_size_y
                else:
                    self.pose.x[k] = 0
                    self.pose.y[k] = 0

            self.pose.x[:] = P_x[:]
            self.pose.y[:] = P_y[:]

    def draw_bounding_box(self, image):
        # print(
        #     f"({self.bounding_box.x_min[0]},{self.bounding_box.y_min[0] }),({self.bounding_box.x_max[0]},{self.bounding_box.y_max[0]})"
        # )
        cv2.rectangle(
            image,
            (self.bounding_box.x_min[0], self.bounding_box.y_min[0]),
            (self.bounding_box.x_max[0], self.bounding_box.y_max[0]),
            (255, 255, 255),
            4,
        )
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        return image

    def __get_single_frame_bounding_box(self):
        # use the coordinates in the video to find the smallest box that includes all coordinates

        # The coordinate system has its origin in the upper left corner.
        # The x_min, y_min correspond to the upper left corner of the bounding box
        # The x_max, y_max correspond to the bottom right corner of the bounding box

        # get the min and max values
        # must convert to dataframes to ensure that nan are not returned

        x_min = pd.DataFrame(
            {
                self.left_hand.x[self.left_hand.x > 0].min(),
                self.right_hand.x[self.right_hand.x > 0].min(),
                self.pose.x[self.pose.x > 0].min(),
            }
        ).min()[0]

        x_max = pd.DataFrame(
            {self.left_hand.x.max(), self.right_hand.x.max(), self.pose.x.max()}
        ).max()[0]

        y_min = pd.DataFrame(
            {
                self.left_hand.y[self.left_hand.y > 0].min(),
                self.right_hand.y[self.right_hand.y > 0].min(),
                self.pose.y[self.pose.y > 0].min(),
            }
        ).min()[0]

        y_max = pd.DataFrame(
            {self.left_hand.y.max(), self.right_hand.y.max(), self.pose.y.max()}
        ).max()[0]

        self.__check_new_bounding_box_values(x_min, y_min, x_max, y_max)

    def __get_bounding_box_min_max(self, boxes):
        # use the coordinates in the video to find the smallest box that includes all coordinates

        # The coordinate system has its origin in the upper left corner.
        # The x_min, y_min correspond to the upper left corner of the bounding box
        # The x_max, y_max correspond to the bottom right corner of the bounding box

        # get the min and max values
        # must convert to dataframes to ensure that nan are not returned

        x_min = pd.DataFrame(
            {
                self.left_hand.x[self.left_hand.x > 0].min(),
                self.right_hand.x[self.right_hand.x > 0].min(),
                self.pose.x[self.pose.x > 0].min(),
            }
        ).min()[0]

        x_max = pd.DataFrame(
            {self.left_hand.x.max(), self.right_hand.x.max(), self.pose.x.max()}
        ).max()[0]

        y_min = pd.DataFrame(
            {
                self.left_hand.y[self.left_hand.y > 0].min(),
                self.right_hand.y[self.right_hand.y > 0].min(),
                self.pose.y[self.pose.y > 0].min(),
            }
        ).min()[0]

        y_max = pd.DataFrame(
            {self.left_hand.y.max(), self.right_hand.y.max(), self.pose.y.max()}
        ).max()[0]

        self.__check_new_bounding_box_values(x_min, y_min, x_max, y_max)

    def __check_bounding_box_validity(self, image):
        # the default CV video is 640 pixels by 480 pixels. Because the holistic model estimates landmarks that are not within the frame, it is possible that the bounding box is outside the actual frame.
        # check that the maximum x value is between 0 and 640, and y value is between 0 and 480

        max_height, max_width, _ = image.shape

        if self.bounding_box.x_min[0] < 0:
            self.bounding_box.x_min = 0

        if self.bounding_box.y_min[0] < 0:
            self.bounding_box.y_min = 0

        if self.bounding_box.x_max[0] > max_width:
            self.bounding_box.x_max = max_width

        if self.bounding_box.y_max[0] > max_height:
            self.bounding_box.y_max = max_height

        # after the validity is confirmed, extract the size of the frame delimited by the bounding box
        self.new_size_x = self.bounding_box.x_max[0] - self.bounding_box.x_min[0]
        self.new_size_y = self.bounding_box.y_max[0] - self.bounding_box.y_min[0]

        # If the bounding box is smaller than 350, by 350, this function increased moves the sided outward until the frame is 350 by 350
        # self.__resize_bounding_box_x()

    def __resize_bounding_box_x(self):

        # print(
        #     f"recursive function called. X value is {self.new_size_x}, Y value is {self.new_size_y}"
        # )

        # Check that the cropped frame is not smaller than the desired 350 by 350 pixels
        if self.new_size_x < 350:

            if self.bounding_box.x_min[0] - 1 > 0:
                self.bounding_box.x_min[0] -= 1

            self.new_size_x = self.bounding_box.x_max[0] - self.bounding_box.x_min[0]

            if (
                self.new_size_x < 350
                and self.bounding_box.x_max[0] + 1 < MAX_INPUT_FRAME_WIDTH
            ):
                self.bounding_box.x_max[0] += 1

            self.new_size_x = self.bounding_box.x_max[0] - self.bounding_box.x_min[0]
            self.__resize_bounding_box_x()

        if self.new_size_y < 350:

            if self.bounding_box.y_min[0] - 1 > 0:
                self.bounding_box.y_min[0] -= 1

            self.new_size_y = self.bounding_box.y_max[0] - self.bounding_box.y_min[0]

            if (
                self.new_size_y < 350
                and self.bounding_box.y_max[0] + 1 < MAX_INPUT_FRAME_HEIGHT
            ):
                self.bounding_box.y_max[0] += 1

            self.new_size_y = self.bounding_box.y_max[0] - self.bounding_box.y_min[0]

            # self.__resize_bounding_box_x()

    def __check_new_bounding_box_values(self, x_min, y_min, x_max, y_max):

        # check if the vlaues are nan
        if np.isnan(x_min):
            x_min = 0

        if np.isnan(y_min):
            y_min = 0

        self.bounding_box.x_min = int(x_min)
        self.bounding_box.y_min = int(y_min)
        self.bounding_box.x_max = int(x_max)
        self.bounding_box.y_max = int(y_max)

    def __map_coordinates_to_resized_image(self, fr_idx, left_hand, right_hand, pose):

        # Values are converted to numppy arrays for faster processing.

        # Left hand points
        LH_x = np.asarray(left_hand.x)
        LH_y = np.asarray(left_hand.y)
        LH_z = np.asarray(left_hand.z)

        for loop_idx in range(len(LH_x)):
            if LH_x[loop_idx] > 0:
                LH_x[loop_idx] = (
                    LH_x[loop_idx] - self.bounding_box.x_min[0]
                ) / self.new_size_x
                LH_z[loop_idx] = (
                    LH_z[loop_idx] - self.bounding_box.x_min[0]
                ) / self.new_size_x
            if LH_y[loop_idx] > 0:
                LH_y[loop_idx] = (
                    LH_y[loop_idx] - self.bounding_box.y_min[0]
                ) / self.new_size_y

        # Right hand points
        RH_x = np.asarray(right_hand.x)
        RH_y = np.asarray(right_hand.y)
        RH_z = np.asarray(right_hand.z)

        for loop_idx in range(len(RH_x)):
            if RH_x[loop_idx] > 0:
                RH_x[loop_idx] = (
                    RH_x[loop_idx] - self.bounding_box.x_min[0]
                ) / self.new_size_x
                RH_z[loop_idx] = (
                    RH_z[loop_idx] - self.bounding_box.x_min[0]
                ) / self.new_size_x
            if RH_y[loop_idx] > 0:
                RH_y[loop_idx] = (
                    RH_y[loop_idx] - self.bounding_box.y_min[0]
                ) / self.new_size_y

        # Pose points
        P_x = np.asarray(pose.x)
        P_y = np.asarray(pose.y)

        for loop_idx in range(len(P_x)):
            if P_x[loop_idx] > 0:
                P_x[loop_idx] = (
                    P_x[loop_idx] - self.bounding_box.x_min[0]
                ) / self.new_size_x
            if P_y[loop_idx] > 0:
                P_y[loop_idx] = (
                    P_y[loop_idx] - self.bounding_box.y_min[0]
                ) / self.new_size_y

        # flatten the data in the order x,y,z. Store in new, 1-D arrays
        LH_output = np.zeros(shape=(len(LH_x) * 3))
        for loop_idx in range(len(LH_x)):
            LH_output[3 * loop_idx] = LH_x[loop_idx]
            LH_output[3 * loop_idx + 1] = LH_y[loop_idx]
            LH_output[3 * loop_idx + 2] = LH_z[loop_idx]

        RH_output = np.zeros(shape=(len(RH_x) * 3))
        for loop_idx in range(len(RH_x)):
            RH_output[3 * loop_idx] = RH_x[loop_idx]
            RH_output[3 * loop_idx + 1] = RH_y[loop_idx]
            RH_output[3 * loop_idx + 2] = RH_z[loop_idx]

        P_output = np.zeros(shape=(len(P_x) * 2))
        for loop_idx in range(len(P_x)):
            P_output[2 * loop_idx] = P_x[loop_idx]
            P_output[2 * loop_idx + 1] = P_y[loop_idx]

        # Concatenate the flattened arrays into a single 1-D array
        self.output_features[fr_idx] = np.concatenate(
            (LH_output, RH_output, P_output), axis=0
        )

        # replace any nan values with zeros
        self.output_features[fr_idx][np.isnan(self.output_features[fr_idx])] = 0

    def __draw_point_on_frame(self, frame, point_x, point_y):

        height, width, _ = frame.shape
        # print(
        #     f"The second finger coordinates are {int(point_x*width)}, {int(point_y*height)}"
        # )

        # imgplot = plt.imshow(frame)
        # plt.show()

        frame = cv2.circle(
            frame,
            (int(point_x * width), int(point_y * height)),
            radius=4,
            color=(255, 0, 255),
            thickness=2,
        )

        # imgplot = plt.imshow(frame)
        # plt.show()

        return frame

    def draw_point_on_frame_pixels(self, frame, point_x, point_y):

        height, width, _ = frame.shape
        print(f"The finger coordinates are {int(point_x)}, {int(point_y)}")

        # imgplot = plt.imshow(frame)
        # plt.show()

        frame = cv2.circle(
            frame,
            (int(point_x), int(point_y)),
            radius=4,
            color=(255, 0, 0),
            thickness=2,
        )

        imgplot = plt.imshow(frame)
        plt.show()

        return frame

    def __draw_all_points_on_image(self, image, results):
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     mp_holistic.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        # )
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

        return image
