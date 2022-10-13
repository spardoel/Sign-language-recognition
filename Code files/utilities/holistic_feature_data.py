# this class is used to extract the holistic features from videos.

import pandas as pd
import numpy as np

import cv2
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from utilities.circular_queue import CircularQueue


class HolisticData:
    def __init__(self, number_of_frames):

        # create the queues
        self.q_frames = CircularQueue(number_of_frames)
        self.q_boxes = CircularQueue(number_of_frames)
        self.q_LH_coords = CircularQueue(number_of_frames)
        self.q_RH_coords = CircularQueue(number_of_frames)
        self.q_P_coords = CircularQueue(number_of_frames)

    def add_new_frame(self, frame, box, left_hand, right_hand, pose):
        # add the frame
        self.q_frames.enqueue(frame)
        self.q_boxes.enqueue(box)
        self.q_LH_coords.enqueue(left_hand)
        self.q_RH_coords.enqueue(right_hand)
        self.q_P_coords.enqueue(pose)

    def get_queue_tail(self):

        return self.q_frames.tail

    def get_clip(self):

        return (
            np.asarray(self.q_frames.get_queue()),
            self.q_boxes.get_queue(),
            self.q_LH_coords.get_queue(),
            self.q_RH_coords.get_queue(),
            self.q_P_coords.get_queue(),
        )

    def save_video(self, save_path, marked_clip=False):
        pass

    def __identify_bounding_box(self):
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
                self.poses.x[self.poses.x > 0].min(),
            }
        ).min()[0]

        x_max = pd.DataFrame(
            {self.left_hand.x.max(), self.right_hand.x.max(), self.poses.x.max()}
        ).max()[0]

        y_min = pd.DataFrame(
            {
                self.left_hand.y[self.left_hand.y > 0].min(),
                self.right_hand.y[self.right_hand.y > 0].min(),
                self.poses.y[self.poses.y > 0].min(),
            }
        ).min()[0]

        y_max = pd.DataFrame(
            {self.left_hand.y.max(), self.right_hand.y.max(), self.poses.y.max()}
        ).max()[0]

        self.__check_new_bounding_box_values(x_min, y_min, x_max, y_max)

    def __check_bounding_box_validity(self):
        # the default CV video is 640 pixels by 480 pixels. Because the holistic model estimates landmarks that are not within the frame, it is possible that the bounding box is outside the actual frame.
        # check that the maximum x value is between 0 and 640, and y value is between 0 and 480

        if self.bounding_box.x_min[0] < 0:
            self.bounding_box.x_min = 0

        if self.bounding_box.y_min[0] < 0:
            self.bounding_box.y_min = 0

        if self.bounding_box.x_max[0] > 640:
            self.bounding_box.x_max = 640

        if self.bounding_box.y_max[0] > 480:
            self.bounding_box.y_max = 480

        # after the validity is confirmed, extract the size of the frame delimited by the bounding box
        self.new_size_x = self.bounding_box.x_max[0] - self.bounding_box.x_min[0]
        self.new_size_y = self.bounding_box.y_max[0] - self.bounding_box.y_min[0]

        # If the bounding box is smaller than 350, by 350, this function increased moves the sided outward until the frame is 350 by 350
        self.__resize_bounding_box_x()

    def __resize_bounding_box_x(self):

        # print(
        #     f"recursive function called. X value is {self.new_size_x}, Y value is {self.new_size_y}"
        # )

        # Check that the cropped frame is not smaller than the desired 350 by 350 pixels
        if self.new_size_x < 350:

            if self.bounding_box.x_min[0] - 1 > 0:
                self.bounding_box.x_min[0] -= 1
            if self.bounding_box.x_max[0] + 1 < 640:
                self.bounding_box.x_max[0] += 1

            self.new_size_x = self.bounding_box.x_max[0] - self.bounding_box.x_min[0]
            self.__resize_bounding_box_x()

        if self.new_size_y < 350:

            if self.bounding_box.y_min[0] - 1 > 0:
                self.bounding_box.y_min[0] -= 1
            if self.bounding_box.y_max[0] + 1 < 480:
                self.bounding_box.y_max[0] += 1

            self.new_size_y = self.bounding_box.y_max[0] - self.bounding_box.y_min[0]

            self.__resize_bounding_box_x()

    def __check_new_bounding_box_values(self, x_min, y_min, x_max, y_max):

        # the first time through
        if self.bounding_box.x_max.values[0] == 0:
            self.bounding_box.x_min = int(x_min)
            self.bounding_box.y_min = int(y_min)
            self.bounding_box.x_max = int(x_max)
            self.bounding_box.y_max = int(y_max)

        # Check if the current values are larger (or smaller) than the current ones
        if x_min < self.bounding_box.x_min[0]:
            self.bounding_box.x_min = int(x_min)

        if y_min < self.bounding_box.y_min[0]:
            self.bounding_box.y_min = int(y_min)

        if x_max > self.bounding_box.x_max[0]:
            self.bounding_box.x_max = int(x_max)

        if y_max > self.bounding_box.y_max[0]:
            self.bounding_box.y_max = int(y_max)

    def __map_coordinates_to_resized_image(self, idx):
        self.video_feature_compoenents[idx]

        self.left_hand = self.video_feature_compoenents[idx][0]
        self.right_hand = self.video_feature_compoenents[idx][1]
        self.poses = self.video_feature_compoenents[idx][2]
        # print(self.left_hand)
        # print(f"inside mapping function {self.left_hand.x[2]}")
        # print(f"first point is {self.left_hand.x[20]}")

        # Left hand
        self.left_hand.x[self.left_hand.x > 0] = (
            self.left_hand.x[self.left_hand.x > 0] - self.bounding_box.x_min[0]
        ) / self.new_size_x
        self.left_hand.y[self.left_hand.y > 0] = (
            self.left_hand.y[self.left_hand.y > 0] - self.bounding_box.y_min[0]
        ) / self.new_size_y
        self.left_hand.z[self.left_hand.x > 0] = (
            self.left_hand.z[self.left_hand.x > 0] - self.bounding_box.x_min[0]
        ) / self.new_size_x

        # print(f"inside mapping function2 {self.left_hand.x[2]}")
        # print(self.left_hand)

        # Right hand
        self.right_hand.x[self.right_hand.x > 0] = (
            self.right_hand.x[self.right_hand.x > 0] - self.bounding_box.x_min
        ) / self.new_size_x
        self.right_hand.y[self.right_hand.y > 0] = (
            self.right_hand.y[self.right_hand.y > 0] - self.bounding_box.y_min
        ) / self.new_size_y
        self.right_hand.z[self.right_hand.x > 0] = (
            self.right_hand.z[self.right_hand.x > 0] - self.bounding_box.x_min
        ) / self.new_size_x

        # Pose coordinates
        self.poses.x[self.poses.x > 0] = (
            self.poses.x[self.poses.x > 0] - self.bounding_box.x_min / self.new_size_x
        )
        self.poses.y[self.poses.y > 0] = (
            self.poses.y[self.poses.y > 0] - self.bounding_box.y_min
        ) / self.new_size_y

        # combine to form output variable
        LH_temp = self.left_hand.to_numpy().flatten()
        RH_temp = self.right_hand.to_numpy().flatten()
        Pose_temp = self.poses.to_numpy().flatten()

        # print(LH_temp)

        self.output_features[idx] = np.concatenate(
            (LH_temp, RH_temp, Pose_temp), axis=0
        )

        # print(f"first point is {self.left_hand.x[0]}")

    def __draw_point_on_frame(self, frame, point_x, point_y):

        height, width, _ = frame.shape
        print(
            f"The second finger coordinates are {int(point_x*width)}, {int(point_y*height)}"
        )

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

    def __draw_point_on_frame_pixels(self, frame, point_x, point_y):

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

        # imgplot = plt.imshow(frame)
        # plt.show()

        return frame

    def __draw_all_points_and_show_image(self, image, results):
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     mp_holistic.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        # )
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        # )
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        # )
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        # )

        # converts the normalized coordinate values to pixel values by multiplying by the iamge size
        if results.left_hand_landmarks:
            for i, mark in enumerate(results.left_hand_landmarks.landmark):

                self.left_hand.x[i] = mark.x
                self.left_hand.y[i] = mark.y
                self.left_hand.z[i] = mark.z
                # print(f"point after calculation {self.left_hand.x[i]}")
        else:  # pad with zero
            self.left_hand = pd.DataFrame(np.zeros((21, 3)), columns=["x", "y", "z"])

        image = self.__draw_point_on_frame(
            image, self.left_hand.x[8], self.left_hand.y[8]
        )

        # imgplot = plt.imshow(image)
        # plt.show()
        return image

    def write_video_to_file(self, frames, filename="Holistic_tracking_left_wrist.mp4"):

        frames2 = np.asarray(frames)

        curent_directory = os.getcwd()
        new_video_path = os.path.join(curent_directory, filename)

        height, width, _ = frames[0].shape

        print(f"Saving: {new_video_path}")
        # by default the fps or all videos is 25
        out = cv2.VideoWriter(
            new_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (width, height),
        )

        for frame in frames2:
            out.write(np.asarray(frame))

        out.release()
