# Developing a sign language interpreter using machine learning Part 3 - Testing the model

To this point I had loaded the dataset and trained a CNN model to identify images of the ASL sign language alphabet. 
But classifying images from a dataset is boring. In this post I will explain how I used Open CV and my webcam to turn the model into a real-time sign language interpreter. 

## Real-time sign identification  
Getting the model to run in real-time on images from my webcam was surprisingly easy. 
The code consisted of a few simple classes and some nice functions provided by OpenCV.
To start, let's talk about the SignIdentificationModel class.
This class uses a class variable to hold the string labels of the possible classes. 
In the init method, the pre-trained model is loaded and saved as a class property. 
The only other class method is the predict_sign method which runs on every frame of data. Here is the code for the class.

```
from tensorflow.keras.models import model_from_json
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
```
As you see, the init method uses model_from_json to load the model and update the weights. Then the predict_sign method receives an image, classifies it, and returns the label as a string. 

The next class is the one controlling the webcam. 
```
class WebCam(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.img_size = 50
        self.model = SignIdentificationModel("model_1.json", "model_weights_1.h5")

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray_frame, (self.img_size, self.img_size))
        pred = self.model.predict_sign(roi[np.newaxis, :, :])
        print(f"Letter is : {pred}")

        cv2.putText(frame, pred, (50, 50), self.font, 2, (255, 255, 50), 2)

        return frame

```
The init method creates the videoCapture object that gets the images from the webcam. The font for the prediction text overlay is set, the image size is set, and the class controlling the pre-trained model created. 
The get_frame method reads a frame from the videocapture object.
The frame is then converted to grayscale using cv2.cvtcolor(). Next, the image is resized to match the inputs used during model training. 
Then, the image dimensionality is increased by 1 and passed to the model for classification. The predicted result is plcaed on the frame using the cv2.putText command. Finally, the frame (with the text overlay) is returned.

Just a few more things.
```
def main(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow("Sign language recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
```
The main function is defined. It starts and infinite loop which gets the next frame from the camera and displays it using cv2.imshow. The loop breaks when q is pressed.
Then, as shown below, all there is left is to run the main function. 
```
main(WebCam())
```

## Real-time sign classification first try

I ran the program and started signing the alphabet. Here is a short video of the first test. I signed the letters A, B, C, D, E, F, A.

https://user-images.githubusercontent.com/102377660/188236875-24ed3ee4-1087-409a-b216-3d11b66e6971.mp4

As you can tell from the video, the model had no idea what was going on an was essentially useless. 
Eventhough the model had a decent validation accuracy (83%) the model is completely unable to identify the alphabet when I sign. 
So what now? 
Honestly, this result was not that surprising. After all, the model only used 100 samples from each of the 30+ classes, it's no wonder it isn't doing well. Plus, keep in mind that the model is relatively simple. 

## Wrap up
Ok, so the model basically sucked. But that's ok, cause I had a plan!
The next step was to adjust the model, re-train it and try again. 

