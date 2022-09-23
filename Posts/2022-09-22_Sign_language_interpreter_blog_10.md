# Developing a sign language interpreter using machine learning Part 10 - Scaling up the model

In the previous post I talked about getting the sign language identification model to classify video captured from my webcam. The model did ok, and could identify about half of the words I was signing. 

In this post I'll talk about scaling up the model to include 100 words instead of just 10. 

## Generating the dataset

In the previous posts I talked about the pre-processing steps. First, I created a script to crop the videos and sort them into folder according to their label. I also wrote a script to load, augment and save the videos. Yet another script open the videos, performs feature extraction and saves the result. All of these scripts were written to be very easy to set and forget. If you recall, the feature extraction script takes a long time to run so I would often start the code, do something else and check on it occasionally to see if it had finished. But there was one step of the video processing pipeline that needed to be done manually - checking the videos.

### Manually checking over 1000 video clips 

As with any dataset, the one I was using was not perfect. Some videos were missing or corrupted. Others were cropped so that only part of the person were visible. In other cases, the video istelf was fine but the label was incorrect. For these reasons it was necessary for me to manually check every video before using it. When the number of words was increased to 100, that meant many hundreds of videos needed to be checked. So on one rainy autumn day I put on some sports-ball in the background and methodically went through all of the videos. 

## Re-training the model with a larger dataset

Since I had always intended to increase the number of words, the code I had written for the 10 word dataset barely needed to be changed at all. The main difference was adding '_100' to the end of the output file names... Once the videos were checked and augmented, and the features were extracted, I retrained the classification model. I quickly noticed that the model was struggling to train. To help fix this, I replaced the simple recurrent neural network layers with gated recurrent unit layers and added a few more dense layers. After some tweaking this is the model architecture I used.

```
    # create the initial RNN layer
    x = keras.layers.GRU(512, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    # add a second RNN layer
    x = keras.layers.GRU(512)(x)
    # add several dense layers with dropout
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    # define the output layer
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)
```
Compared to the 10 word model, this one has more dense layers, more nodes per layer, and a higher dropout rate. I increased the dropout when I noticed the model showing signs of overfitting (large difference between training and validation performance). After training this model for a few hunderd epochs it reached a respectable validation accuracy of 93.44%. Here is the graph of the training. 


![GRU_100_training_Sep_22](https://user-images.githubusercontent.com/102377660/191865490-879e1d5f-0181-42e6-83c4-e1f2f49867be.png)

The model performance was pretty similar for the training and validation sets. The training dataset was better, but that is to be expected and the difference was fairly small. 

So I ran the model to see how many of the 100 words I could remember. Oh, and to see if the classifier worked. Here is a video of me signing the words 'Paint', 'Table' and 'Hat'.

https://user-images.githubusercontent.com/102377660/191869536-4d61ea57-f5cf-4f4a-974c-44d04b48260e.mov

After testing, the model was rather disapointing. You can see that although the model eventually guesses the signs correctly it made a lot of mistakes. For many words, it could not guess them at all. Honestly, the classification was bad. It struggled with most signs and could only classify a handful of words correctly. I dug into the model to investigate possible issues. I started running the model and stepped through the program checking the variables and printing the outputs whenever possible. I came across something interesting. 

## Resizing frames
 
When I started developing the word classification system I treated the videos as a stack of 2D images. Each frame was cropped, and had its features extracted before being classified by the sign detection model. What I failed to realize was that by cropping each frame around the person I was distorting the positional relationships of landmarks between frames. Take a look at this video and you'll see what I mean. 

Here is a video of me signing the word 'Deaf'.


https://user-images.githubusercontent.com/102377660/191867788-f2fdc8d0-6b05-4605-9252-4885cb8e5f4c.mov


And now, here is what the model sees after each frame was cropped.


https://user-images.githubusercontent.com/102377660/191867808-fe267dfa-9046-4c48-b891-9c3cdde49a6a.mov


A bit different don't you think? When I move my hand out beside me the YOLO object detection model expands the bounding box to keep my arm in the frame. When the image is resized, the result is that the frame is squished horizontally. This makes perfect sense when you think about it. Frankly it was a bit embarassing that I didn't realize the problem sooner. I won't be making that mistake again. 

Anyway, the videos that I gave the model for training didn't have randomly compressed images which cause edges to jump positions between consecutive frames. This could be why the model was struggling. 

## Fixing the resizing issue  

To solve this problem I had a simple solution - use the same bounding box for the entire video clip.

First I moved the frame resizing out of the get_frame() method. Instead of returning the cropped frame, this method now returns the coordinates of the bounding box. This change was very simple so I am not including the code here. The more significant changes were in the main function. Here, take a look.

```
def main(camera):

    predicted_sign = "Nothing"

    # Create the two queues
    q_frames = CircularQueue(50)
    q_boxes = CircularQueue(50)

    while True:
        # Get the next video frame from the camera
        bounding_box, frame = camera.get_frame()

        # add the frame
        q_frames.enQueue(frame)
        q_boxes.enQueue(bounding_box)

        if q_frames.tail == 49:

            # get the bounding boxes and video from the queues
            clip = q_frames.getQueue()
            clip_boxes = q_boxes.getQueue()

            # process the video and crop to largest bounding box in 'clip_box'
            frame_features, frame_mask, new_clip = camera.process_clip(
                np.asarray(clip), np.asarray(clip_boxes)
            )

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
```
The main function now instantiates 2 circular queues. One is for the video frames, as before, and the other is for the list of bounding box coordinates. If the frames queue has 50 frames, then begin processing the clip. The video clip and bounding boxes are extracted from the queues then passed to the process_clip() method. This method (not shown) simply uses np.max() and np.min() function to identify the outermost points of the bounding boxes. For the x1,y1 point smaller values are the outermost since this is the upper-left most point. For the x2,y2 point the larger values are the outermost points. These points are then used to crop all frames of the video clip. After that, the features are extracted and the clip is classified as done previously. 

Here is an example of the video clip when all the frames are cropped to the outermost points of the bounding boxes. 




https://user-images.githubusercontent.com/102377660/191872902-0b84468a-7e93-434a-9f4d-38cd87be09d2.mov


As you can see, the video looks much better. 


## Re-test the model

After implementing the change the model was slightly better (maybe). Honestly it was hard to tell, since I wasn't using a well defined testing strategy. 





 Well darn. So what now? I thought about throwing in the towel, I even gave the model a 2 day time-out as punishment, but then I came up with another idea. Well, to be more precise, I recalled an idea that the dataset authors mentioned in their paper. Pose detection. Up to this point I had been using a general purpose feature extraction model as a pre-processing step. These features were then fed into the classification model. Since I want to identify the movement of the arms and hands, wouldn't the positional coordinates of the elbows, hands, and fingers be the ultimate input features? This this in mind, I set my sights on a pose detection approach. 

## Wrap up 

Increasing the model vocabulary from 10 to 100 didn't make the model any better (shocker I know). Even after fixing an issue with the frame resizing, the model was still struggling. Well plan A had failled. Plan A being the Convolutional NN approach. So I guess plan A was actually plan C (C for convulution). It was time to move on to plan B, which was Pose estimation, aka plan P. If plan P didn't work I would have to move on to plan C - Despair. But not C as in convolution, because that was actually plan A. If plan P failled, which is actually plan B, then I would need to try a third thing - plan C aka plan D. Goodness, this is needlessly confusing isn't it? Let's hope I don't need to resort to plan C / D.


