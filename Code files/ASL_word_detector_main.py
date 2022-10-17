import cv2
import numpy as np
from keras.models import model_from_json
from utilities.holistic_feature_extractor import HolisticFeatureExtractor
from utilities.holistic_feature_data import HolisticData


# create the feature extractor and data handler classes
feature_extractor = HolisticFeatureExtractor()
data = HolisticData(26)


# the list of all words the classifier was trained to identify
CLASS_LIST = [
    "accident",
    "africa",
    "all",
    "apple",
    "basketball",
    "bed",
    "before",
    "bird",
    "birthday",
    "black",
    "blue",
    "book",
    "bowling",
    "brown",
    "but",
    "can",
    "candy",
    "chair",
    "change",
    "cheat",
    "city",
    "clothes",
    "color",
    "computer",
    "cook",
    "cool",
    "corn",
    "cousin",
    "cow",
    "dance",
    "dark",
    "deaf",
    "decide",
    "doctor",
    "dog",
    "drink",
    "eat",
    "enjoy",
    "family",
    "fine",
    "finish",
    "fish",
    "forget",
    "full",
    "give",
    "go",
    "graduate",
    "hat",
    "hearing",
    "help",
    "hot",
    "how",
    "jacket",
    "kiss",
    "language",
    "last",
    "later",
    "letter",
    "like",
    "man",
    "many",
    "medicine",
    "meet",
    "mother",
    "need",
    "no",
    "now",
    "orange",
    "paint",
    "paper",
    "pink",
    "pizza",
    "play",
    "pull",
    "purple",
    "right",
    "same",
    "school",
    "secretary",
    "shirt",
    "short",
    "son",
    "study",
    "table",
    "tall",
    "tell",
    "thanksgiving",
    "thin",
    "thursday",
    "time",
    "walk",
    "want",
    "what",
    "white",
    "who",
    "woman",
    "work",
    "wrong",
    "year",
    "yes",
]


font = cv2.FONT_HERSHEY_SIMPLEX

# the location of the classifier model and weights
model_weights_file = "model/model_weights_100_holistic_features_cropped_OOP2.h5"
model_json_file = "model/model_100_holistic_features_cropped_OOP2.json"

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
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()

        return fr


def main(camera):

    predicted_sign = "Nothing"

    while True:

        # Get the next video frame from the camera
        frame = camera.get_frame()

        # process the frame
        frame = feature_extractor.run_feature_extractor_single_frame(frame, data)
        # frame = feature_extractor.draw_bounding_box(frame)

        if data.get_queue_tail() == 25:

            # process the video and crop to largest bounding box in 'clip_box'
            frame_features, frame_mask, new_clip = feature_extractor.process_clip(data)

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
