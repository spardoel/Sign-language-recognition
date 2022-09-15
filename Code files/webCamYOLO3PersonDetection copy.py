import cv2
import torch
import numpy as np
import json
from utils import CircularQueue
from keras.models import model_from_json
import keras
from keras import applications

CLASS_LIST1 = [
    "drink",
    "go",
    "cousin",
    "walk",
    "book",
    "chair",
    "before",
    "candy",
    "who",
    "clothes",
    "computer",
]

CLASS_LIST = [
    "all",
    "before",
    "black",
    "book",
    "candy",
    "chair",
    "clothes",
    "computer",
    "cousin",
    "deaf",
    "drink",
    "fine",
    "finish",
    "like",
    "many",
    "mother",
    "no",
    "now",
    "orange",
    "table",
    "walk",
    "who",
    "year",
    "yes",
]

# Model
# model = torch.hub.load( "ultralytics/yolov3", "yolov3")  # or yolov3-spp, yolov3-tiny, custom

model = torch.hub.load("ultralytics/yolov5", "yolov5n")

font = cv2.FONT_HERSHEY_SIMPLEX

img_size = 350
q = CircularQueue(50)

model_weights_file = "model_weights_sug.h5"
model_json_file = "model_aug.json"

# load model from JSON file
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into the new model
    loaded_model.load_weights(model_weights_file)
    loaded_model.make_predict_function()


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()  # returns camera frames along with bounding boxes and predictions

    def get_frame(self):
        _, fr = self.video.read()

        # crop and resize the image before returning
        results = model(fr)  # identify the person
        sign_frame = fr
        dataframe = results.pandas()  # convert to pandas dataframe
        detected_objects = dataframe.xyxy[0].name
        for idx, object in enumerate(detected_objects):
            if object == "person":
                # cv2.putText(fr, dataframe.xyxy[0].name[idx], (25, 25), font, 1, (255, 255, 0), 2)
                # print(object)
                x1 = round(dataframe.xyxy[0].xmin[idx])
                y1 = round(dataframe.xyxy[0].ymin[idx])

                x2 = round(dataframe.xyxy[0].xmax[idx])
                y2 = round(dataframe.xyxy[0].ymax[idx])

                sign_frame = fr[y1:y2, x1:x2]

                sign_frame = cv2.resize(sign_frame, (350, 350))
                cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return sign_frame, fr

    def process_clip(self, video):

        pass
        # roi = cv2.resize(fr, (img_size, img_size))
        # temp = roi[np.newaxis, :, :, :]


IMG_SIZE = 350
NUM_FEATURES = 1280
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


feature_extractor = build_feature_extractor()


def prepare_videos(frames):

    MAX_SEQ_LENGTH = 50
    num_samples = 1

    frames = frames[None, ...]

    # initialize the mask and feature arrays
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

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
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    frame_features = temp_frame_features.squeeze()
    frame_masks = temp_frame_mask.squeeze()

    return (frame_features, frame_masks)


def gen(camera):
    rest_frames = 0
    predicted_sign = "Nothing"
    while True:
        sign_frame, frame = camera.get_frame()
        frame2 = frame
        q.enQueue(sign_frame)

        clip = q.getQueue()
        if len(clip) == 50:

            if rest_frames == 50:
                rest_frames = 0
                frame_features, frame_mask = prepare_videos(
                    np.asarray(clip).astype(dtype=np.int32)
                )

                frame_mask2 = frame_mask[np.newaxis, :]
                frame_features2 = frame_features[np.newaxis, :, :]
                pred = loaded_model.predict([frame_features2, frame_mask2])[0]

                predicted_sign = CLASS_LIST[np.argmax(pred)]
                print(predicted_sign)

            rest_frames += 1
            q.deQueue()  # remove the last frame to make room for the next one

        cv2.putText(
            frame,
            predicted_sign,
            (25, 25),
            font,
            1,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Sign language recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


gen(VideoCamera())
