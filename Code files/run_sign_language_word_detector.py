import cv2
import torch
import numpy as np
import json
from utilities.circular_queue import CircularQueue

from keras.models import model_from_json
import keras
from keras import applications

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
q = CircularQueue(50)

model_weights_file = "model_weights_sug.h5"
model_json_file = "model_aug copy.json"

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

        dataframe = results.pandas()  # convert to pandas dataframe
        detected_objects = dataframe.xyxy[0].name
        for idx, object in enumerate(detected_objects):
            if object == "person":

                x1 = round(dataframe.xyxy[0].xmin[idx])
                y1 = round(dataframe.xyxy[0].ymin[idx])

                x2 = round(dataframe.xyxy[0].xmax[idx])
                y2 = round(dataframe.xyxy[0].ymax[idx])

                sign_frame = fr[y1:y2, x1:x2]

                sign_frame = cv2.resize(sign_frame, (350, 350))
                cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return sign_frame, fr

    def process_clip(self, frames):

        num_frames = len(frames)

        # initialize the mask and feature arrays
        frame_masks = np.ones(shape=(num_frames), dtype="bool")
        frame_features = np.zeros(shape=(num_frames, NUM_FEATURES), dtype="float32")

        frame_features = feature_extractor.predict(frames)

        return (frame_features, frame_masks)


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


def main(camera):

    predicted_sign = "Nothing"
    while True:
        sign_frame, frame = camera.get_frame()

        q.enQueue(sign_frame)

        clip = q.getQueue()

        if q.tail == 49:

            frame_features, frame_mask = camera.process_clip(np.asarray(clip))

            frame_mask2 = frame_mask[np.newaxis, :]
            frame_features2 = frame_features[np.newaxis, :, :]
            pred = loaded_model.predict([frame_features2, frame_mask2])[0]

            predicted_sign = CLASS_LIST[np.argmax(pred)]
            print(predicted_sign)

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


feature_extractor = build_feature_extractor()
main(VideoCamera())
