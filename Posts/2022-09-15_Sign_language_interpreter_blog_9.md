# Developing a sign language interpreter using machine learning Part 9 - Setting up the real-time video classification

The sign language word classification model was trained. During the training it reached 97% validation accuracy. This was infinitely better than the first attempt but I wasn't sure if it would work with totally new input videos. 
For this next part of the project, I set up the model to classify video being streamed from my webcam. 

## Overview

In the first phase of this project, I set up a model to classify images of the sign language alphabet. 
I set up the model to accept frames coming from my live webcam. In that version, my hand was the only thing in the frame. 
In addition, I pointed the webcam at the ceiling to ensure that there was a plain, solid coloured background. Now, is phase 2 of the project things were a bit different. 
In the sign language word dataset I used to develop the model, the videos were cropped to only include the person. This was done using the YOLO3 model. 
To give my model a fighting chance I wanted to make the input resemble the training data as much as possible. 
This meant that I needed to run a classifier to identify me in the frame and crop each frame of the video. In addition, the cropped video frames needed to be resized to match the input shape that the model was expecting. 
After that, the video needed to be run through the feature extraction model. During the model training, this was by far the longest step. I was concerned that running the feature extraction in real-time might not be possible, but more on that later. 
Finally, after the data had gone through the first two models it needed to be classified by my sign classification model. There were many details to figure out along the way, but for now, let's jump in. I'll explain along the way. 

## Object detection using YOLO

The dataset used YOLOV3 to identify the person within the frame and crop the video around the person. Since I wanted to match the training data as much as possible, I logically looked into the YOLOV3 model. 
I downloaded a pre-trained model and started experimenting with object detection. I could pass an image to the YOLO model and the model would output a list of object within the image, complete with the coordinates of the bounding boxes. 
This was great, it was exactly what I needed. But then, I noticed that YOLOV3 was not the most recent version. In fact there was a YOLOV5 available. As we all know, newer is always better. So I traded in for the newer model. 

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
The image is passed to the model, then displayed. Here is the output. 

![YOLOv5_example_image](https://user-images.githubusercontent.com/102377660/190526428-f3e7b62b-d64f-4ae8-aaf4-370766be7f55.jpg)

As you can see, the YOLO model identifies a variety of objects within the iamge and draws bounding boxes around them. Perfect. 

Great, so I had a way to find the person within the frame. Next, I needed to be able to access the list of detected objects, identify which one is the person doing the sign language, and drop the video. 
This part proved to be tricky. 

As it turned out, the 'results' variable returned by the YOLO model was somewhat complicated. I am not sure what I was expecting, but a properties named 'predicted_classes' and 'bounding_boxes' would have been nice. 
Alas, it was not the case. Here is a picture of the 'results' variable paused in the SV code debugger. 

![results from yolov5](https://user-images.githubusercontent.com/102377660/190527055-7b0a7b3e-fc86-41a4-b28e-a4ccc29ba9c8.JPG)

Having never used this model before, I didn't really know what I was looking at or how to get the information I wanted. 
So what to do? Well, I had just used the results.print and results.show methods to display the iamge with the bounding boxes and labels. So I checked the source code for those methods. 
In the "utils" folder I found some of the code I was looking for. 
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
These two methods describe the conversion of coordinates. It seemed that properties were different representations of the same data. Ok, good to know. 
Next, I converted the results to pandas using 
```
 dataframe = results.pandas() 
```
This made the data somewhat easier to work with. After converting to a dataframe, I selected one of the properties, namely 'xyxy' and tried to print it to get a better understanding of how I can access the data. 
Running this code
```
dataframe = results.pandas()
print(dataframe.xyxy)
```
Gave the following output
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
Ok! Now I was getting somewhere. After converting the results to a dataframe, and selecting the coordinate representation I wanted (xyxy) I could access the list of detections. 
Each detection had the X,Y coordinates of the bounding box as well as the class name. Perfect, I can work with that. 

I will talk a bit more about exactly how I used implemented the YOLO model later on. For now I want to talk about the different version that I had to choose from. 
The Ultralytics github page list several version of the YOLOV5 model. Each one has its pros and cons, but for me, speed was the most important. I planned to use the model to detect one person within a frame. That's it. 
I didn't need the most accurate model, I didn't need to be able to identify a variety of objects, I just needed to identify one person. Furthermore, the accuracy of detection wasn't even that important. 
I didn't care if the bounding box was slightly too small and cut off the top of the person's head. No, for me, speed was the most important. For this reason, I selected the smallest of the options. 
The YOLOV5-nano. This model had the fewest parameters and the fastest classification time, which was exactly what I was looking for. 

## Writing the code

After figuring out how to do the person detection, the rest of the implementation was pretty straightforward. 
