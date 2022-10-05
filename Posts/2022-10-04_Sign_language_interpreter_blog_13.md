# Developing a sign language interpreter using machine learning Part 13 - Refactoring to improve program speed

As a refresher, here is a video of the current sign language word identification program. 

https://user-images.githubusercontent.com/102377660/193940632-c7080bd4-08bb-46d8-9c6b-0f3fbec07b88.mov

As you can see, the video freezes for a few seconds before the next guess is printed in the top left corner of the frame. I wanted to eliminate this delay. 
This post will be about the process of refactoring the code and how I needed to generate a new dataset and train a whole new model. 

## What is taking so long? 

I wanted to speed up the code. Well that's great, but first I needed to identify which parts of the code where slowing down the program. I imported the time module and checked various sections of the code. 

In the first draft of this section I started by copying out the main function that runs the sign language interpreter program. I then used the time module to evaluate the methods and follow the rabbit trail until I identified the part of the code that was taking the longest. After some thought, I removed all that. 

Here is what you need to know. For each frame, the YOLO model was used to identify the person in the frame. This was done as the frames were received. The YOLO model was fast enough to run as the frames were received. Therefore, the YOLO model was not the cause of the delay. Once 50 frames of video are ready, then the features are extracted. The features are then passed to the classification model. The classification model runs once per clip of 50 frames. This model could be simplified a bit, but compared to the YOLO and feature extraction models, the classification model was basically instantaneous. No, the YOLO and classification models were not the problem. The part of the code responsible for the several second delay before each new classifier decision was the feature extraction model.

Well darn, what to do about that? The feature extraction is pretty vital and is responsible for the good persormance of the model. The previous feature extractor I used was faster but had much worse performance. Again, what to do? 

Running the feature extractor model once didn't take too long. After all, mediapipe created the models to be run in real time. The problem was that I was waiting until 50 frames were ready then running the feature extractor 50 times. That is what caused the delay. Well, could I run the feature extractor on each new frame as it is received? Well, not really, since I needed to wait for the entire clip to be ready before I could crop it, and it was these cropped frames that were given to the feature extractor. Ok, so cropping the frames imposes some limitations. Well, is cropping actually necessary? could the solution be as simple as not cropping? Honestly, maybe. I tried removing the cropping and running the feature ectractor model on the un-cropped frames. The model performed much worse than the cropped videos. It seemed that the best performance was acheived when the video was cropped. 

