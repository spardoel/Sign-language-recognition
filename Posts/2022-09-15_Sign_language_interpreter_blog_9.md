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

I downloaded a pre-trained model and started experimenting with object detection. I could pass an image to the YOLO model, and the model would show me the image with all the distinct objects nicely encircled. Weel, encircled with rectangles that is. So I guess the objects were en-regtangled? I don't know, but it was great, exactly what I needed. But then, I noticed that YOLOV3 was not the most recent version. In fact, there was a YOLOV5 available. As we all know, newer is always better. So I traded in for the newer model. 

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
After importing pytorch, the model is loaded. Then the path to an image is provided. In this case I used the a random stock image of a generic office.
The image is passed to the model, then displayed. Pretty easy right? Here is the output. 

![YOLOv5_example_image](https://user-images.githubusercontent.com/102377660/190526428-f3e7b62b-d64f-4ae8-aaf4-370766be7f55.jpg)

As you can see, the YOLO model identifies a variety of objects within the image, draws bounding boxes around them, provides a label, and a value representing the model's confidence in that label. Perfect. 

Great, so I had a way to find the person within the frame. Next, I needed to be able to access the list of detected objects, identify which one is the person doing the sign language, and crop the video. 

This part proved to be tricky. 

As it turned out, the 'results' variable returned by the YOLO model was somewhat complicated. I am not sure what I was expecting, but a property named 'predicted_classes' and 'bounding_boxes' would have been nice. 
Alas, that was not the case. Here is a picture of the 'results' variable paused in the VS Code debugger. 

![results from yolov5](https://user-images.githubusercontent.com/102377660/190527055-7b0a7b3e-fc86-41a4-b28e-a4ccc29ba9c8.JPG)

Having never used this model before, I didn't really know what I was looking at or how to get the information I wanted. I immediately saw the 'names' property but since there were no bicycles or trucks in this image I assume that is the dictionary of all the object the model is capable of detecting and not the names of the objects in this particular image. There were some values labeled 's' and 't' and a few tensors labeled with a bunch of X's and Y's.

So what to do? Well, I had just used the results.print() and results.show() methods to display the image with the bounding boxes and labels. So those methods must be accessing the information I needed. If I could take a look at the code behind those functions I could use the same steps to access the data for my own nefarious purposes. So I hopped off to github to checked the source code for those methods. 

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
Each detection had the X,Y coordinates of the bounding box as well as the class name. They were even sorted according to the model's confidence value. Perfect, I can work with that. 

I will talk a bit more about exactly how I implemented the YOLO model later on. For now, I want to talk about the different versions of the model that I had to choose from. 

### Which YOLO to use?

The Ultralytics github page lists several versions of the YOLOV5 model. Each one with its own pros and cons, but for me, speed was the most important. 

I planned to use the model to detect one person within a frame. That's it. I didn't need the most accurate model, I didn't need to be able to identify a variety of objects, I just needed to identify one person. Furthermore, the accuracy of detection wasn't even that important. 
I didn't care if the bounding box was slightly too big or slightly too small and cut off the top of the person's head. No, for me, speed was the most important. For this reason, I selected the smallest of the options: YOLOV5-nano. This model had the fewest parameters and the fastest classification time, which was exactly what I was looking for. Cool. Moving on.  

## Running the YOLO model in real time

After selecting the YOLO model I wanted to test it with the live video feed from my webcam. 
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
After the imports, the YOLO5-nano model is loaded. The font to be used when writing text on the video frames is set. Next, the VideoCamera class is defined. The __init__ method creates the video capture object. The __del__ method releases the object when the class is deleted. Next, the get_frame() method. This method reads the next frame from the video capture object. The frame is passed to the YOLO5 model and the results are converted to a pandas dataframe. Then the list (pandas Series technically) of detected objects is extracted. This step was not strickly necessary, but I think it makes the code a bit easier to understand. The code then loops through each of the detected objects. If the detected object is a 'person' then the label (i.e., 'person') is written in the top left hand corner of the frame.  Next, the coordinates of the bounding box are extracted and rounded. It was not strictly necessary to create the (x1,y1), and (x2,y2) intermediate variables, but I think it helps make the code easier to understand. After generating the coordinates, the bounding box was drawn on the frame and the frame was returned. Next, the main() method is created. This method requests the next frame, displays the processed frame, and checks for an exit condition. That's it. 

Ok, so what happens when the code is run? Well, this happens.


https://user-images.githubusercontent.com/102377660/190858607-c3f0eb33-f8fd-4d3d-8fb2-a2e3cd42dcdc.mp4


As you can see, the YOLO model identifies me in the frame and the bounding box is drawn. The bounding box follows me as I move around in the frame. Perfect. 

With the basic testing of the YOLO model, the 'Identify the person in the frame' part of the program was solved. 

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

This zero padding is needed to compensates for shorter videos. But in the case of real-time implementation, the zero padding is unecessary. When pulling frames from the camera, I can simply keep pulling frames until I reach the desired amount. In other words, I can ensure that the video is never shorter than 50 frames. No padding required. This means, I can simplify the feature extraction code and run it in efficient batches of 50 frames.

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
    def enQueue(self, data):

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
    def getQueue(self):
        data = []
        if self.head == -1:
            print("Queue is empty")

        pointer = self.head
        while True:
            # print(f"Head is {self.head}, tail is {self.tail}, pointer is {pointer}")
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
To write the circular queue I created a new class with a few relatively simple methods. The constructor method creates the class object and sets the size of the queue as 'max_size'. An empty list of that size is then created as self.queue. Next, the indexes indicating the position of the start and end of the queue are created as self.head and self.tail, and both are set to -1.

The next method is called enqueue(). This method adds a new sample to the end of the queue, or, if the queue is full, it overwrites the oldest sample. First, this method checks if the head is -1. If yes, then this is the first element being added to the queue, set the head and tail to 0 and add the element to the queue. If this is not the first element being added, then increment the tail and add the element to the end of the queue. Notice that the tail is incremented using the modulo operator. This is the key to the function of a circular queue. The modulo operator returns the remainder of a division. So, 2 % 5 returns 2 because 5 goes into 2 o times with a remainder of 2. Crucially, 5 % 5 returns 0 since 5 goes into 5 once with no remainder. This means that incrementing the index then dividing by the maximum length of the queue will increment the index normally until the end of the quue is reached, at which point the index qill be reset to 0. This way the index starts over and loops through the queue again. Right, back to the code. After the tail is incremented and the data is saved to the queue, the tail iscompared to the head. If they are the same, the queue index has wrapped around and is overwiting the previous head value. In this case, increment the head to stay ahead of the tail. 

The next method is the getQueue() method. This method will unravel the queue and return it in its entirety. For this, an index called 'pointer' is used. The pointer is set to equal the head. Then the element in the queue at location 'pointer' is appended to the 'data' variable which is used as the output. The code then checks if pointer is equal to tail. If so, then that was the last element in the queue, so break the while loop. Finally, return the data variable.

The last method in the CircularQueue class is the printQueue() method. This method is mainly used to test the other two methods. The printQueue() method simply loops through the queue from index 0 until the end and prints the value. 

To test the queue, I ran the following code. 
```
q = CircularQueue(4)

q.enQueue(0)
q.printQueue()
q.enQueue(1)
q.printQueue()
q.enQueue(2)
q.printQueue()
q.enQueue(3)
q.printQueue()

q.enQueue(4)
q.printQueue()

print(q.getQueue())

q.enQueue(5)
q.printQueue()

print(q.getQueue())
```
First, the CircularQueue is instantiated as q and the maximum length is set to 4. 
Then 4 values (0,1,2,3) are added to the queue, the contents of the queue are printed each time. Then, a fifth value is added to the queue. This value should overwrite the first element in the queue. 

Next, the getQueue() method is called. This should return the contents of the queue in the order we provided. So in this case the getQueue method should return [1,2,3,4]. Next, another value is added to the queue, this one should overwrite the next element in the queue. I think you get the picture. When the code is run, this is the output. 

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
This is exactly what I was looking for. The queue fills up thenstarts overwriting previous values. In addition, when the getQueue method is used, the result is returned in the correct order. Great. Queue, finished. 

## Putting it all together

I had figured out how to run the YOLO model to identify the person withing the frame. I had also inspected the feature extraction section of the code and come up with a plan to speed it up. Now it was time to put it all together. 



