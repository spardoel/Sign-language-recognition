from keras.models import model_from_json
import numpy as np
import cv2


class SignIdentificationModel(object):

    CLASS_LIST = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    def __init__(self, model_json_file, model_weights_file):

        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.trained_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.trained_model.load_weights(model_weights_file)
        self.trained_model.make_predict_function()

    def predict_sign(self, img):
        self.preds = self.trained_model.predict(img)
        return SignIdentificationModel.CLASS_LIST[np.argmax(self.preds)]


class WebCam(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.img_size = 50
        self.model = SignIdentificationModel(
            "model_alphabet.json", "model_weights_alphabet.h5"
        )

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray_frame, (self.img_size, self.img_size))
        pred = self.model.predict_sign(roi[np.newaxis, :, :])
        print(f"Letter is : {pred}")

        cv2.putText(frame, pred, (50, 50), self.font, 2, (255, 255, 50), 2)

        return frame


def main(camera):
    while True:
        next_frame = camera.get_frame()
        cv2.imshow("Sign language recognition", next_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


main(WebCam())
