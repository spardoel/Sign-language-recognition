# Developing a sign language interpreter using machine learning Part 9 - Setting up the real-time video classification

The sign language word classification model was trained. During the training, it reached 97% validation accuracy. This was infinitely better than the first attempt, but I wasn't sure if it would work with totally new input videos. The data augmentation generated many 'new' data samples that were very similar to others. This meant that the performance of the model could be artificially high. To investigate further, I wanted to implement the model in a more realistic context. 

For this next part of the project, I set up the model to classify video being streamed from my webcam.

## Overview

In the first phase of this project, I set up a model to classify images of the sign language alphabet. 
I set up the model to accept frames coming from my live webcam. In that version, my hand was the only thing in the frame. 
In addition, I pointed the webcam at the ceiling to ensure that there was a plain, solid coloured background. But now, in phase 2 of the project, things were a bit different. 

In the sign language word dataset I used to develop the model, the videos were cropped to only include the person. This was done using the YOLO3 model. 
To give my model a fighting chance at success, I wanted to make the real-time input resemble the training input as much as possible. 
This meant that I needed to identify the person (me) in the frame and crop each frame of the video. In addition, the cropped video frames needed to be resized to match the input shape that the model was expecting. 
After that, the video needed to be run through the feature extraction model. This part was concerning. During the model training, feature extraction was by far the longest step. I was concerned that running the feature extraction in real-time might not be possible, but more on that later. 
Finally, after the data had gone through the first two models, it needed to be classified by my sign language word classification model. 

So I needed to do the following:

1. Identify the person in the frame.
2. Crop the frames and perform feature extraction.
3. Classify the video clip. 

There were many details to figure out, but for now, let's jump in. I'll explain along the way. 

## Object detection using YOLO (You Only Look Once)

I won't go into the theory behind YOLO models, but in essence, it is a simplification of previous multi-step object detection approaches. The dataset used YOLOV3 to identify the person within the frame, so that the video could be cropped to include only the person. Since I wanted to match the training data as much as possible, I logically looked into the YOLOV3 model. 

I downloaded a pre-trained model and started experimenting with object detection. I could pass an image to the YOLO model, and the model would show me the image with all the distinct objects nicely encircled. Well, encircled with rectangles that is. So I guess the objects were en-regtangled? I don't know, but it was great, exactly what I needed. But then, I noticed that YOLOV3 was not the most recent version. In fact, there was a YOLOV5 available. As we all know, newer is always better. So I traded in for the newer model. 

The shinny new model that caught my eye was provided by the good people at Ultralytics. Here is a link to their YOLOV5 github page https://github.com/ultralytics/yolov5.

Here is a super simple example of the model in use. 

```
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

img = "D:/Documents/Python projects/Sign language blog files/stock office.jpg"

# Inference
results = model(img)

# Results
results.print() 

results.show()
```
After importing pytorch, the model is loaded. Then the path to an image is provided. In this case I used a random stock image of a generic office.
The image is passed to the model, then displayed. Pretty easy right? Here is the output. 

![YOLOv5_example_image](https://user-images.githubusercontent.com/102377660/190526428-f3e7b62b-d64f-4ae8-aaf4-370766be7f55.jpg)

As you can see, the YOLO model identifies a variety of objects within the image, draws bounding boxes around them, provides a label, and a value representing the model's confidence in that label. Perfect. 

Great, so I had a way to find the person within the frame. Next, I needed to be able to access the list of detected objects, identify which one is the person doing the sign language, and crop the video. 

This part proved to be tricky. 

As it turned out, the 'results' variable returned by the YOLO model was somewhat complicated. I am not sure what I was expecting, but properties named 'predicted_classes' and 'bounding_boxes' would have been nice. 
Alas, that was not the case. Here is a picture of the 'results' variable paused in the VS Code debugger. 

![results from yolov5](https://user-images.githubusercontent.com/102377660/190527055-7b0a7b3e-fc86-41a4-b28e-a4ccc29ba9c8.JPG)

Having never used this model before, I didn't really know what I was looking at or how to get the information I wanted. I immediately saw the 'names' property but since there were no bicycles or trucks in this image I assume that is the dictionary of all the objects the model is capable of detecting and not the names of the objects in this particular image. There were some values labeled 's' and 't' and a few tensors labeled with a bunch of X's and Y's.

So what to do? Well, I had just used the results.print() and results.show() methods to display the image with the bounding boxes and labels. So those methods must be accessing the information I needed. If I could take a look at the code behind those functions I could use the same steps to access the data for my own nefarious purposes. So I hopped off to github to check the source code for those methods. 

To be honest, it took longer than I had hoped. But eventually I found something useful. 
In the "utils" folder I found some of the code I was looking for, take a look. 
```
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

```
These two methods describe the conversion of coordinates. It seemed that properties labeled with X's and Y's were different representations of the same data. So I should only need to access one of those properties. That narrowed things down somewhat. I could explore one of the XY properties and see what I could find. 

While going through the documentation and code on the Github page, I also learned that the results can be converted into pandas dataframes. I hoped this would make it easy to interpret and work with the results. So I converted the results to pandas using the following devilishly complicated command. 
```
 dataframe = results.pandas() 
```
This made the data somewhat easier to work with. After converting to a dataframe, I selected one of the XY properties, namely 'xyxy' and tried to print it to get a better understanding of how I can access the data. 
I ran the following code.
```
dataframe = results.pandas()
print(dataframe.xyxy)
```
Which produced this output.
```
[         xmin        ymin        xmax        ymax  confidence  class    name
0  136.110565   86.834450  239.048477  291.053192    0.876165      0  person
1   71.859322  224.772690  253.474701  348.138336    0.823238     63  laptop
2  262.835114   73.205490  410.364105  292.073090    0.745676      0  person
3  262.766724  137.355728  492.365814  404.668854    0.656045      0  person
4  325.229462  225.454422  519.091248  407.656555    0.581231     56   chair
5    0.000000  200.403870   51.203964  318.416046    0.572645     56   chair
6    0.255729  119.131889   47.150875  239.244247    0.499317     62      tv
7  459.049164  236.212570  520.255920  407.262390    0.366459     56   chair]
```
Hey! Now I was getting somewhere. After converting the results to a dataframe, and selecting the coordinate representation I wanted (xyxy) I could access the list of detections. 
Each detection had the X,Y coordinates of the bounding box as well as the class name. They were even sorted according to the model's confidence value. Perfect, I could work with that. 

I will talk a bit more about exactly how I implemented the YOLO model later on. For now, I want to talk about the different versions of the model that I had to choose from. 

### Which YOLO to use?

The Ultralytics github page lists several versions of the YOLOV5 model. Each one with its own pros and cons, but for me, speed was the most important. 

I planned to use the model to detect one person within a frame. That's it. I didn't need the most accurate model, I didn't need to be able to identify a variety of objects, I just needed to identify one person. Furthermore, the accuracy of detection wasn't even that important. 
I didn't care if the bounding box was slightly too big or slightly too small and cut off the top of the person's head. No, for me, speed was the most important. For this reason, I selected the smallest of the options: YOLOV5-nano. This model had the fewest parameters and the fastest classification time, which was exactly what I was looking for. Cool. Moving on.  

## Running the YOLO model in real time

After selecting the YOLO model, I wanted to test it with the live video feed from my webcam. 
Here is the test code I wrote. This code is available in "Code files/webcam_YOLO5_person_detection.py"

```
import cv2
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5n")

font = cv2.FONT_HERSHEY_SIMPLEX  # The font used when displaying text on the image frame

class VideoCamera(object):
    def __init__(self):
        # create the Open CV video capture object
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()  
        
    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):

        # Read the next frame from the camera
        _, fr = self.video.read()

        # Use the YOLO model to identify objects in the frame
        results = model(fr)

        # convert the results to pandas
        dataframe = results.pandas()

        # create an intermediate variable to hold the detected objects (for clarity)
        detected_objects = dataframe.xyxy[0].name

        # loop through each detected opbect
        for idx, object in enumerate(detected_objects):

            #  if the object is classified as a person
            if object == "person":

                # print the label to the frame
                cv2.putText(fr, dataframe.xyxy[0].name[idx], (25, 25), font, 1, (255, 255, 0), 2)

                # extract the integer coordinates of the bounding box
                x1 = round(dataframe.xyxy[0].xmin[idx])
                y1 = round(dataframe.xyxy[0].ymin[idx])

                x2 = round(dataframe.xyxy[0].xmax[idx])
                y2 = round(dataframe.xyxy[0].ymax[idx])

                # Draw the bounding box around the object
                cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # return the frame with the text and bounding box
        return fr


def gen(camera):
    while True:

        frame = camera.get_frame()

        cv2.imshow("Sign language recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# run the main function and pass in the camera object
gen(VideoCamera())
```
After the imports, the YOLO5-nano model is loaded. The font to be used when writing text on the video frames is set. Next, the VideoCamera class is defined. The __init__() method creates the video capture object. The __del__() method releases the object when the class is deleted. Next, let's talk about the get_frame() method. This method reads the next frame from the video capture object. The frame is passed to the YOLO5 model and the results are converted to a pandas dataframe. Then the list (pandas Series technically) of detected objects is extracted. This step was not strickly necessary, but I think it makes the code a bit easier to understand. The code then loops through each of the detected objects. If the detected object is a 'person' then the label (i.e., 'person') is written in the top left hand corner of the frame.  Next, the coordinates of the bounding box are extracted and rounded. It was not strictly necessary to create the (x1,y1), and (x2,y2) intermediate variables, but I think it helps make the code easier to understand. After generating the coordinates, the bounding box was drawn on the frame and the frame was returned. Next, the main() method is created. This method requests the next frame, displays the processed frame, and checks for an exit condition. That's it. 

Ok, so what happens when the code is run? Well, this happens.


https://user-images.githubusercontent.com/102377660/190858607-c3f0eb33-f8fd-4d3d-8fb2-a2e3cd42dcdc.mp4


As you can see, the YOLO model identifies me in the frame and the bounding box is drawn. The bounding box follows me as I move around in the frame. Perfect. 

With the basic testing of the YOLO model done, the 'Identify the person in the frame' part of the program was solved. 

## Feature extraction

When training the model, the feature extraction step took by far the longest. This was somewhat concerning. As a refresher, here is the feature extraction step in the previous code.

```
  # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(path,max_frames=MAX_SEQ_LENGTH)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1,MAX_SEQ_LENGTH,),dtype="bool",)
        temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES),dtype="float32")

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx] = temp_frame_features.squeeze()
        frame_masks[idx] = temp_frame_mask.squeeze()
```
Now, to be honest I hadn't gone through this code with a fine tooth comb. The feature extraction code was taken from the Keras documentation (https://keras.io/examples/vision/video_classification/). The code was working fine and I didn't see a reason to mess with it. Until now that is. So let's dive in. 

Specifically, let's go through this section. 

```
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
```
This is the part of the code that actually performs the feature extraction. I ran the code and paused it here in the debugger. Ok, so the 'frames' variable had a shape of (1,50,350,350,3). The 50 is the number of frames, the '350,350' are the width and height of the image (in pixels) and the 3 was the number of channels in the image (3 because it was a colour image). In the code section above, the first line uses enumerate(frames) to loop through the frames variable and keep track of the index. The 'frames' variable's first dimension was 1, so this loop only ran once. Interesting.

Inside the loop, the length of the video was extracted, this is the number of frames in the clip. Then the length of the video is compared to the pre-set maximum length of 50, and the smaller of the two is saved as 'length'. Now the code loops through each of the frames in the clip. The features are saved for each frame. Then, the masks variable is populated with ones where frames were available. Importantly, the variables used to store the features and masks were created as arrays of zeros. This means that if the video clip does not have the maximum number of frames, the features and mask variables are padded with zeros. 

This zero padding is needed to compensate for shorter videos. But in the case of real-time implementation, the zero padding is unecessary. When pulling frames from the camera, I can simply keep pulling frames until I reach the desired amount. In other words, I can ensure that the video is never shorter than 50 frames. No padding required. This means I can simplify the feature extraction code and run it in efficient batches of 50 frames.

Speaking of batches, I needed a way to collect a desired number of video frames to pass to the feature extractor. Then I wanted to be able to get a few more frames and pass those to the feature extractor. Gather a streaming input to process in batches? Sounds a bit like a buffer don't you think?

## Circular queue

As I alluded in the previous section, I needed some way of collecting batches of frames. I could have simply taken the first X frames, classified the video clip, then taken the next X frames and classified those. But I wanted to allow for more frequent processing. By that I mean I wanted to be able to collect overlapping video clips. For this, I needed a data structure that could store a set amount of frames and replace the oldest frames as new ones became available. Did somebody say circular queue? 

Circular queues are simple enough in theory. A set amount of space is allocated at creation and each new sample is added until the allocated space is full. Then, the queue starts over and replaces the oldest entries with the new ones. What makes my implmentation slightly different is that I also wanted to get the entire contents of the queue once it was full. Enough with the talking, why don't I show you what I mean, eh?

```
class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.head = self.tail = -1

    # Insert an element into the circular queue
    def enqueue(self, data):

        # if this is the first entry
        if self.head == -1:
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            # if not the first, increment the tail and add the data
            self.tail = (self.tail + 1) % self.max_size
            self.queue[self.tail] = data

            # if the tail has caught up with the head, increment the head
            if self.tail == self.head:
                self.head = (self.head + 1) % self.max_size

    # Get the contents of the entire queue
    def get_queue(self):
    # create the output variable
        data = []
        if self.head == -1:
            print("Queue is empty")

        pointer = self.head
        
        while True:
            data.append(self.queue[pointer])

            if pointer == self.tail:
                break

            pointer = (pointer + 1) % self.max_size

        return data

    # print the contents of the queue
    def printQueue(self):
        if self.head == -1:
            print("Queue is empty")

        for i in range(self.max_size):

            if self.queue[i] is not None:

                print(self.queue[i], end=" ")
        # new line
        print()
```
To write the circular queue, I created a new class with a few relatively simple methods. The constructor method creates the class object and sets the size of the queue as 'max_size'. An empty list of that size is then created as self.queue. Next, the indexes indicating the position of the start and end of the queue are created as self.head and self.tail, and both are set to -1.

The next method is called enqueue(). This method adds a new sample to the end of the queue, or, if the queue is full, it overwrites the oldest sample. First, this method checks if the head is -1. If yes, then this is the first element being added to the queue, set the head and tail to 0 and add the element to the queue. Otherwise, if this is not the first element being added, then increment the tail and add the element to the end of the queue. Notice that the tail is incremented using the modulo operator. This is the key to the function of a circular queue. The modulo operator returns the remainder of a division. So, 2 % 5 returns 2 because 5 goes into 2 zero times with a remainder of 2. Crucially, 5 % 5 returns 0 since 5 goes into 5 once with no remainder. This means that incrementing the index then dividing by the maximum length of the queue will increment the index normally until the end of the quue is reached, at which point the index will be reset to 0. This way, the index starts over and loops through the queue again. Hence the name 'Circular' queue. Right, back to the code. After the tail is incremented and the data is saved to the queue, the tail is compared to the head. If they are the same, the queue index has wrapped around and is overwiting the previous head value. In this case, increment the head to stay ahead of the tail (the head stays ahead of the tail, get it? A-Head? I'm sorry, moving on). 

The next method is the get_queue() method. This method unravels the queue and returns it in its entirety. For this, an index called 'pointer' is used. The pointer is set to equal the head. Then the element in the queue at location 'pointer' is appended to the 'data' variable, which is used as the output. The code then checks if pointer is equal to tail. If so, then the last element has beed taken from the queue, so break the while loop. Finally, return the data variable.

The last method in the CircularQueue class is the printQueue() method. This method is mainly used to test the other two methods. The printQueue() method simply loops through the queue from index 0 until the end and prints the value. 

To test the queue, I ran the following code. 
```
q = CircularQueue(4)

q.enqueue(0)
q.print_queue()
q.enqueue(1)
q.printq_ueue()
q.enqueue(2)
q.print_queue()
q.enqueue(3)
q.print_queue()

q.enqueue(4)
q.print_queue()

print(q.get_queue())

q.enqueue(5)
q.printQueue()

print(q.get_queue())
```
First, the CircularQueue is instantiated as q and the maximum length is set to 4. 
Then 4 values (0,1,2,3) are added to the queue, the contents of the queue are printed each time. Then, a fifth value is added to the queue. This value should overwrite the first element in the queue. 

Next, the get_queue() method is called. This should return the contents of the queue in the order we provided. So in this case the get_queue method should return [1,2,3,4]. Next, another value is added to the queue, this one should overwrite the next element in the queue. I think you get the picture. When the code is run, this is the output. 

```
0 
0 1
0 1 2
0 1 2 3
4 1 2 3
[1, 2, 3, 4]
4 5 2 3
[2, 3, 4, 5]
```
This is exactly what I was looking for. The queue fills up then starts overwriting previous values. In addition, when the get_queue() method is used, the result is returned in the correct order. Great. Queue finished. 

## Putting it all together

I had figured out how to run the YOLO model to identify the person withing the frame. I had also inspected the feature extraction section of the code and come up with a plan to speed it up. Lastly, a circular queue was written to handle the collection of new video frames. Now it was time to put it all together. 

The code I will go through in this section is available in "run_sign_language_word_detector.py".

Alright, as a broad overview, there are 3 main sections to this code. 
1. The VideoCamera() class that fetches and processes frames of video. 
2. The build_feature_extractor() function which creates and returns the feature extraction model. 
3. The main() function which contains the main loop and calls the other functions. 

As a quick note, I didn't spend much time thinking about the program architecture and where to place functions. This was just a test script and not the final program. So if you find yourself asking 'Why isn't that code in its own class?' or 'Wouldn't it make more sense to move that function into a method?' you are probably right. I'll look at all those things later, for now, I was only interested in getting the code to run to test the basic functionality. 

### Some basic setup

Before jumping into the code, there were a few things to setup. 

```
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

```
After the imports, the CLASS_LIST constant was created with the possible class labels. Then the YOLO model was loaded and the font used for printing the label to the screen is set. The Image size is set, and the queue is instantiated to hold 50 samples. Then, the previously trained classification model is loaded. 

### The VideoCamera class

This class had 4 methods, namely, the __init__() method, the __del__() method, the get_frame() method, and the process_clip() method. Here is the class. 
```
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

```

Starting with the __init__() method. This method simply instantiates the Open CV Video Capture object and sets it as the 'video' property. Next, the __del__() method releases the Video Capture object when the class is deleted. 

The get_frame() method is a bit more interesting. It is very similar to the code used to test the YOLO object detector. The method gets the next frame of video, uses the YOLO model to detect objects, and converts the results to a pandas dataframe. Then, the code loops through the detected objects looking for the 'person' label. When the person is found, the coordinates of the bounding box are extracted. The frame is cropped using the bounding box and saved as sign_frame. Now, at this point there are 2 video frames. The first is the original frame, this one will be used to show the bounding box and the classification label to the user. The other frame (i.e., sign_frame) is the cropped frame and is used for classification. This frame is never shown to the user. 
Ok, so there are 2 frames. Moving on. The sign_frame is then resized to 350 by 350. Meanwhile, the rectangle demonstrating the bounding box is drawn on the original frame. This method ends with both the original and the sign_frame being returned.

Finally there is the process_clip() method. This method accepts a clip of 50 frames and performs the feature extraction. First the length of the clip is determined and saved as 'num_frames'. Then the feature and mask variables are created. Note that the mask variable is created as ones and never changed. As I mentioned previously, the masks are not really needed anymore since the clips are all the same length. The frames are then passed to the feature extractor. Then the method returns the features and the masks, that's it. 

### The build_feature_extractor function

This function is pretty straight forward and hasn't changed since the previous versions of the code. Here it is. 
```
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
```
The size of the images and the number of features are set as constants. Then the Efficient Net is loaded. The input and output shapes are defined and the keras model is created. Like I said, nothing new here. 

### The main function

Alright, the main function for this test script mainly consists of a while loop. The loop gets the next frame from the camera, loads it into the queue and checks a condition. If the condition is true, pause the loop and classify the frames currently in the buffer. Here is the code.
```
def main(camera):
   
    predicted_sign = "Nothing"
    while True:
        sign_frame, frame = camera.get_frame()

        q.enqueue(sign_frame)

        if q.tail == 49:
        
            clip = q.get_queue()
            frame_features, frame_mask = camera.process_clip(np.asarray(clip))

            frame_mask2 = frame_mask[np.newaxis, :]
            frame_features2 = frame_features[np.newaxis, :, :]
            pred = loaded_model.predict([frame_features2, frame_mask2])[0]

            predicted_sign = CLASS_LIST[np.argmax(pred)]
            print(predicted_sign)

        cv2.putText(frame,predicted_sign,(25, 25),font,1,(255, 255, 0),2)
        cv2.imshow("Sign language recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


feature_extractor = build_feature_extractor()
main(VideoCamera())
```
Ok, the main function accepts the VideoCamera object. The prediction result is initially set to 'nothing'. Then the main loop starts. 
The loop gets the next frame and gives it to the queue (q is the instance of the Circular Queue). Then, check the position of the queue tail. If the tail is 49, then the queue is full (because the queue was created to hold 50 frames). When this condition is true, the clip contained in the queue is processed. First the video clip is extracted from the q using get_queue(). The clip is then processed to extract the features and masks. The masks and features are modified to add another dimension. This is to match the expected input for the model. The clip is then classified. The prediction is used as an index to get the appropriate label in the CLASS_LIST constant. This prediction is then printed to the frame and shown to the user. 

## Running the code

Time to run the code! How exciting. I ran the code, and started signing the words in CLASS_LIST. Here is an example.


https://user-images.githubusercontent.com/102377660/190913677-4aa13a54-15da-40bd-9680-6ef6fcd0bacb.mp4

Awesome! It worked. You may have noticed that the video freezes for a split second just before the guessed label in the top left corner is updated. This freezing corresponds to the code classifying the video clip. After the freeze, the code then runs until it has collected 50 new frames, which are then classified, and the label in the top left is updated. 

In the video (when I wasn't looking down to check my notes), I signed 'Chair', 'Clothes', 'Cousin', 'Drink', and 'Go'. The model classified all of them correctly! The reason I signed those 5 words and not the other 5 is because the words shown in the clip are the only ones that the model could reliably classify. In fact, the model was completely unable to classify some words. For instance, when I signed 'Candy' the model always guessed 'Cousin', and when I signed 'Who' the model always guessed 'Drink'. These mistakes were not all that surprising to me. I'll show you a few videos and hopefully you'll see why the model is struggling. 

Here is the sign for 'Candy'.




https://user-images.githubusercontent.com/102377660/191854346-2ee5d5c2-7200-4ef2-89b3-ad19d5c43860.mov



And here is the sign for 'Cousin'




https://user-images.githubusercontent.com/102377660/191854362-5b69395d-861e-4250-a1b6-d2bc91b87b82.mov



To you or I the signs are clearly different. But they are similar enough to confuse the model. I think it may have to do with the positioning of the hand next to the head. I'm not sure, but this seems like a forgiveable mistake. Let's look at a few more videos. 

Here is the sign for 'Who'




https://user-images.githubusercontent.com/102377660/191854388-bb0aeb58-0f60-4c43-a362-5c6bcfc6e2b7.mov




And here is the sign for 'Drink'




https://user-images.githubusercontent.com/102377660/191853576-f5565160-dc01-4ae4-8ff7-9c6b66db3ff9.mov




These signs are even more similar than 'Candy' and 'Cousin'. Both signs have the hand coming in front of the mouth with the fingers slighlty curled. Seeing these signs side by side, it is understandable that the model had trouble distinguishing between the two. 

Another thing I noticed was that 'drink' was sometimes mistakenly labeled as 'cousin'. This is also pretty understandable. Both signs have the hand in a 'C' shape and involve moving the hand near the face. After experimenting a little, I figured out that tilting my head back when signing 'drink' made the classification much more reliable. Phrased another way, accentuating a feature unique to the sign 'drink' helped the model identify the sign (pretty unsurprising when you think about it). 

## Wrap up

The test was a success! Well, mostly. The model could reliably identify 50% of the signs. I am pretty pleased with that result. Also, the test was perfomed in near real-time! Yes, there was a slight lag when a clip was being processed, but not bad at all.

The next steps were to expand the model vocabulary and try to improve classification accuracy. Expanding the model vocabulary would involve adding more videos with different labels. Additional training videos should help the model learn, but adding more classes also makes the classification more challenging. It will be interesting to see what happens. 

There were also some smaller tweaks I could make to improve the model speed and accuracy. I could improve the speed by using fewer features or a lighter feature extraction model. Or by having a better computer (my preferred option). There were also a few things I could do to improve the model performance. For instance, I could remove background obstacles to see if that helps, or I could record myself signing and add those videos to the training set. So I had a few options and things to play with. But more on that in the next post. Thanks for reading!


