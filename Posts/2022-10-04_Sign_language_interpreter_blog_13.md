# Developing a sign language interpreter using machine learning Part 13 - Refactoring to improve program speed

As a refresher, here is a video of the current sign language word identification program. 

https://user-images.githubusercontent.com/102377660/193940632-c7080bd4-08bb-46d8-9c6b-0f3fbec07b88.mov

As you can see, the video freezes for a few seconds before the next guess is printed in the top left corner of the frame. I wanted to eliminate this delay. 
This post will be about the process of refactoring the code and how I needed to generate a new dataset and train a whole new model in order to eliminate this delay. 

## What is taking so long? 

I wanted to speed up the code. Well that's great, but first I needed to identify which parts of the code were slowing down the program. I imported the time module and checked various sections. 

In the first draft of this section, I started copying out the main function that runs the sign language interpreter program. I then used the time module to evaluate the methods and follow the rabbit trail until I identified the part of the code that was taking the longest. After some thought, I removed all that. 

Here is what you need to know. For each frame, the YOLO model was used to identify the person in the frame. This was done as the frames were received. The YOLO model was fast enough to run on each frame in real-time. Therefore, the YOLO model was not the cause of the delay. As the program ran, it collected frames. Once 50 frames of video were ready, then the features were extracted. The features were then passed to the classification model. The classification model ran once per clip of 50 frames. The classification model could be simplified a bit, but compared to the YOLO and feature extraction models, the classification model was basically instantaneous. No, the YOLO and classification models were not the problem. The part of the code responsible for the several second delay before each new classifier decision was the feature extraction model.

### Improving the speed of the feature extraction

The feature extraction step was by far the slowest. Well darn, what to do about that? The feature extraction is pretty vital. In fact, adding this feature extraction model improved perforamnce compared to the previous feature extraction model. I couldn't really just remove the feature extraction model... So, what to do? 

Let's condsider the feature extraction model itself. Running the feature extractor model once didn't take long. After all, mediapipe created the holistic model (which I was using as a feature extractor) to be run in real time. The problem was that I was waiting until 50 frames were ready then running the feature extractor 50 times back to back. **That** is what caused the delay. Well, could I run the feature extractor on each new frame as it is received? Not really, because I needed to wait for the entire clip to be ready before I could crop it, and it was these cropped frames that were given to the feature extractor. hmmm 

### To crop or not to crop?

Ok, so cropping the frames imposed some limitations. Well, is cropping actually necessary? Could the solution be as simple as not cropping? I tried removing the cropping and running the feature ectractor model on the un-cropped frames. The model performed much worse than the cropped videos. It seemed that the best performance was acheived when the video was cropped. This made sense to me because of how the features were represented. The mediapipe holistic model outputs the positional coordinates of the various body parts as percentages of the width and height of the frame. If the frames are all cropped to include only the person, then the magnitude of these percentages matters. For instance, if the frame is cropped to the top of the person's head, the fact that the hand is near the top of the frame is useful information because we know its position relative to the head. However, if the frames are not consistently cropped, then whether the hand is close to the top of the frame or not is meaningless. To a certain extent, these positional relationships can be inferred from the other body landmarks that are generated by the holistic model. However, the holistic model isn't perfect and it often looses track of the hands and other body parts. The position of points relative to the frame on the other hand, is not subject to the same random dropout. So, while it may be possible to produce a good model without cropping the frame, it seemed that with my current model I would get the best results if the frames were cropped. So if the frames need to be cropped, then was I stuck? Did I need to simply accept the delay as an unavoidable fact? Well not necessarily. 

The purpose of cropping the video was to ensure that all relevant points are visible and that the person is consistently in the center (ish) of the frame. 
In this case, the relevant points are the coordinates of the body parts as identified by the holistic model. 
With that in mind, what if I ran the holistic model then cropped the frame to be the smallest rectangle possible that contained all the points? 
This would mean that running the YOLO model to find the person within the frame wouldn't be necessary at all. Ok, that's good, but what about the image vs video cropping? 

### When to crop the frames?

In a previous post, I talked about how cropping each individual frame resulted in weird compressions as each frame was resized a different amount. 
For instance, if I moved my arm out to the side to perform a certain sign, the cropped and resized image would be much more compressed horizontally than the previous and following frames. 

Here is the example from my previous post. 

https://user-images.githubusercontent.com/102377660/194618727-e5112a7f-23bf-4d6a-91db-29eb94edec2a.mov


To solve this issue, I kept track of the bounding box of each frame, but didn't immediately crop the frames. 
Instead, I waited until a clip of 50 frames was ready, then cropped the video using a single bounding box that encompassed all other boxes. (One box to rule them all!) 
This ensured that the frames were resized equally. 

Now, back to the current code. If I ran the holistic model on each new (uncropped) frame of data, the coordinates of the body parts would be given as percentages of the current frame size. 
Then, once the clip is ready, and the video is cropped, the percentage values would all be wrong. To address this new issue, I needed to re-scale the coordinates. I can do that. I hope. 

## The new plan

Ok, so I had a plan. 

First, get rid of the YOLO model. It was no longer needed. Instead, use the holisitc feature extractor for cropping.

For each frame,
1. Set up the holistic model to locate the body lankmarks
2. Convert the landmark coordinates from % of frame dimensions to actual pixel coordinates
3. Generate a bounding box that encompassed all body parts within the frame

Next, once 50 frames were ready,
1. Find the minimums and maximums of the bounding box corners
2. Generate the largest possible bounding box using the minimum and maximum values (a box that encompasses all bounding boxes from all of the frames)
3. Use the new bounding box to crop the video
4. Re-normalize the body landmark positions as % of the width and height of the new (cropped) video 
5. Pass these landmark positions to the classification model as the features. 

Wow, ok, this was turning into a whole big thing. Honestly, I was having a hard time keeping all this straight in my head. 
Visualizing processes always helps me work through problems, so I started working on some illustrations and class diagrams to help me think. But first, I want to present the logic behind the re-normalization method I came up with. 

## Cropping and re-normalization

Take a look at the figure below. The top left pannel (A) shows a 640 pixel by 480 pixel image with a point of interest at coordinates (0.3,0.6). These coordinates are the fraction of the the frame width and height (the origin is in the top let corner). This is the way the holistic model outputs the coordinates of the points. Ok, great. Pannel B shows how this point could also be represented in pixel value by multiplying 0.3 x 640 and 0.6 x 480 which gives the same point with coordinates (192,288), in pixels. This transformation is done for each point of interest every time a new frame is processed. Panel C shows how two points are used to define the bounding box. For this example, the bounding box has points (55,40) and (424,465) in pixels, shown in pannel D.

![cropping explanation figure 1](https://user-images.githubusercontent.com/102377660/194641285-00110c4d-0b44-4422-9069-5178552e32d4.png)

Now take a look at this next image. Panel D is coppied from the previous image. Panel E shows the image after it has been cropped and the point coordinates adjusted to the coordinate system of the new frame. The coordinates of the point previously were (192,288). The x_min and y_min values are subtracted from the point coordinates to give the coordinates of the point in the new (cropped) image. The new coordinates of the point are (192 - 55, 288 - 40) which gives (137, 248). Also notice the width and height of the cropped image. These values were obtained by subtracting the coordinates of x_min,y_min from x_max,y_max, so 424 - 55 = 369, 465 - 40 = 425. Now that the coordinates of the point are known in the new coordinate system, normalize by the image width and height to get the fractional coordinates (137/369, 248/425) which gives (0.37, 0.58), see pannel F. Now when the images are resized to be 350 by 350, the position of the point is still valid, see pannel G. 

![cropping explanation figure 2](https://user-images.githubusercontent.com/102377660/194643027-ebfd61da-79d8-4bc0-a975-c73870391c02.png)

As a way to check that the resized coordinates were correct, during processing of the database images, I randomly saved 10% of the videos and tracked the position of the left index finger. Here are a couple examples. The videos are for the words 'But' and 'Finish'. 



https://user-images.githubusercontent.com/102377660/194874399-e55e35a7-25d3-4560-acbb-6d2f4acc1f7b.mov



https://user-images.githubusercontent.com/102377660/194874454-5aeddb68-028f-4c6a-9693-de277d272af0.mov



As you can see, the tip of the left index finger is being correctly tracked. Also notice that the frame is cropped to include only the points of interest. The position of the eyes are included in the holistic model, but not the top of the head which is why the top of the cropped frame is at the level of the person's eyes. Cool, so the resizing is working as intended. 

## Wrap up

Alright, so I have explained my approach to stream-line and speed up the code. I've done a few small tests and know what I want to do. In the next post I will go through the code and talk about how I implemented everything. 
Thanks for reading!

