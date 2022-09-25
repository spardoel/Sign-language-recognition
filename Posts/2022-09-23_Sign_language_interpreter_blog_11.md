# Developing a sign language interpreter using machine learning Part 11 - Switching to a pose estimation approach 

In the last post I talked about scaling the model up from a 10 word vocabulary to a 100 word vocabulary. Even after fixing a bug in the code the model performance was not great. The model could only reliably a handful of words. 
I decided to change approaches. 

This post will be about my exploration of Mediapipe and their hand tracking API. 

## Pose estimation  

Pose estimation is basically whata it sounds like. When given an image or video of a human, the computer estimates the position of key anatomical landmarks. You may have seen some images of people with colourful skeletons drawn on top. 
Here is an example taken from the Mediapipe page on pose estimation (https://google.github.io/mediapipe/solutions/pose.html).

![image](https://user-images.githubusercontent.com/102377660/192027676-64839a92-4e44-4cc1-99f5-85b31393cbe1.png)

This is accomplished using a variety of machine learning tools and techniques and can be applied to many different useful situations. One example is MyVeloFit. MyVeloFit is a service that uses pose estimation to assess cyclists' position on their bike. 
This is useful when trying to find the most comfortable position on the bike or to determine if the bike is correctly sized for you. Here is a picture taken from their website (https://www.myvelofit.com/about-fit).

![image](https://user-images.githubusercontent.com/102377660/192028250-990c1288-fddc-424b-afb2-77dd4514ccd8.png)

As you can see, their software takes a video of someone riding on an indoor trainer and uses pose estimation to track the movement of their body and limbs. The software then recommends adujstments of various bike components to optimize the fit. 

That brief introduction out of the way, let's look at Mediapipe.

## Mediapipe

Mediapipe is an open source computer vision framework made for pose estimation and object detection. It offers several different models including facial detection, hand tracking, and whole body tracking (called holistic). 
My previous attempt at sign languge word identification was not great. To improve the model I planned to replace the generic feature extractor model with a mediapipe model that identified the coordinates of anatomical landmarks. 
I was hoping that the positional coordinates of the fingers and hands would be better than the generic features when fed into my custom word classifier. 


## Hand tracking

I had decided to use the mediapipe hand tracking model. Here is a link to the website https://google.github.io/mediapipe/solutions/hands.html.
The page gives an overview of the function of the model and a bit of example code. This made it extremely easy to test for myself. After downloading the basic test script I recorded the following video. The code for this test is found in 'pose_estimator_hands.py'


https://user-images.githubusercontent.com/102377660/192034316-aaabfbe7-3f8e-4183-a863-746e90e1cfba.mov


Pretty cool! As you can see, the model hand tracking is good but isn't perfect. when the palm is visible, the tracking is excellent. But the model gets confused when the hands are on top of each other, or when the hand is not fully visible. 


### Testing hand tracking on the dataset

I wanted to see if the hand tracking model could be run on the videos in the dataset. I used the mediapipe example code as a starting point and modified some of my existing code. In just a few minutes, I was able to generate videos from the database with the hand tracking visualization. Take a look.

Here are the words 'Go', 'Book' and 'Computer' after running the mediapipe hand tracking on the augmented videos. 

Here is the word 'Go' 

https://user-images.githubusercontent.com/102377660/192120875-f3cd801c-bb86-47ee-b507-8a162adf534d.mov

Next, 'Book' 

https://user-images.githubusercontent.com/102377660/192120877-8ced1981-a18a-4e22-86c8-568c800ff956.mov

Finally, 'Computer'

https://user-images.githubusercontent.com/102377660/192120882-105f7c58-211e-47c2-afa0-9ec5c024cd6c.mov


Just like when I was running the model on myself, the hand tracking is good but not perfect. It also seemed that the salt and pepper noise may be affecting the tracking. Something to keep in mind for later. 

To generate the above videos, the hand tracking was applied to each frame in the video and the points were drawn onto the frame. Then the frames were saved as a video. This is great, but not actually useful beyond visualization. The next step was to break down the model outputs to access the coordinates of the hands. These positional coordinates would be the inputs for my classification model

## Hand coordinate feature extraction

Even though the hand tracking wasn't perfect, I wanted to give it a try. So I modified the feature extraction code to generate the point coordinates as features. 
You can find the following code in 'preprocess_and_save_hand_coordinate_features.py'

Like the previous versions of the 'preprocess and save...' scripts, this version started by loading the video file paths and labels, then splitting them into training, testing, and validation sets. Then, for each set, each video was loaded, and the features were extracted. The resulting features and masks were saved. To use the hand coordinates as features, only one function needed to be changed. Namely, the function that performed the feature extraction - extract_hand_coordinates(). Here it is. 

```
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


NUM_FEATURES = 126


def extract_hand_coordinates(frames):

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:

        # create the output variable for the features.
        # there are 40 points on each hand and 3 coordinates (x,y,z) for each point

        landmarks_list = np.zeros(shape=(len(frames), NUM_FEATURES), dtype="float32")

        for fr_num, image in enumerate(frames):
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # perform the hand tracking for this frame
            results = hands.process(image)


            landmarks_list_temp = []
            # if there was a hand in the frame
            if results.multi_hand_landmarks:
                # loops through both hands
                for hand_landmarks in results.multi_hand_landmarks:
                    # loops through each point on the hand
                    for mark in hand_landmarks.landmark:
                        # copy the x,y,z, coordinates as separate values
                        landmarks_list_temp.append(mark.x)
                        landmarks_list_temp.append(mark.y)
                        landmarks_list_temp.append(mark.z)

                # copy the features to the output variable. 
                landmarks_list[fr_num, : len(landmarks_list_temp)] = landmarks_list_temp
                
        # Returns the features
        return landmarks_list

```
After importing the packages needed to run mediapipe, the number of variables is defined as a constant. According to the mediapipe documentation, there are 21 points being tracked per hand. For two hands, that is 42 points. For each point there is an X,Y and Z coordinate. This gives a total of 126 features. 

The extract_hand_coordinates() function is where the magic happens. First, the hand detection model is created and called 'hands'. Then the 'landmarks_list' variable is created as an array of zeros. The variable is pre-allocated so that if some points are not available, the remaining points will be zero padded.  
After the output variable is created, the code loops through each frame of the video. For each frame, the frame is prepared by converting the colour to RGB (this was part of the provided demo script, I don't know if it is strictly necessary). Then, the frame is passed to 'hands' for processing. The 'results' variable is a rather complex data structure. I had to do some trial and error to figure out how to access the coordinates. Here is what I came up with. 

Check 'results.multi_hand_landmark' to see if at least one hand was visible. If at least one hadn was visible, loop through each hand. For each hand, loop through the points for that hand. In the code, 'for mark in hand_landmarks.landmark:' loops through the points on a given hand. Then, for each point, append the X,Y and Z coordinates to a temporary list. Once all the available points have been coppied to the temporary list, copy the list to the pre-allocated output variable. Notice that the 'landmarks_list' is filled with the 'landmarks_list_temp' and may have trailing zero padding. It is important to note that if part of a hand is visible, the model will track the points that are visible and estimate the location of the missing points. This means that there are either 21 or 42 points. This is important since it maintains the order of points within the output variable.

Once the features are extracted, they are saved as a pickle file, just like the previous versions. 

### Training model with hand coordinate features

To train the model, I simply needed to change the names of input an output files in the model training script. The rest of the script was practically identicay. I did end up tweaking the model parameters, so I created a separate script to train the hand coordinate model. The following model training used the code in "load_features_and_train_model_hand.py".

After changing the name of the input file, and changing the numberof features to 126, I ran the model training. Initially, the training wasn't great. There were large differences between the training and validation sets. I decreased the model complexity and increased the dropout to 0.5, which helped. Using GRU layers instead of simple RNN layers also helped improve the model. After training for 140 epochs, the model reach a validation accuracy of 80%. Here is the graph of training. 

![GRU_hands_10](https://user-images.githubusercontent.com/102377660/192146997-d8de9056-0e1f-4954-82be-beebc2610bdc.png)


Despite tweaking some parameters the accuracy never really improved beyond 80%. Having generated the hand tracking output on the dataset videos, this was not surprising. The tracking wasn't great on the training videos and often lost track of a hand during the video. With that in mind, I didn't spend too much time trying to optimize the model, since the relatively low accuracy was probably caused by the features themselves and not the model parameters. Instead, I moved on to a real-life test using my webcam. 

### Testing the model

To evaluate the  classification model that used hand tracking as the feature inputs, I set it up to run with my webcam. Again, this was pretty straightforward. I just needed to update a bit of old code to replace the feature extraction step. I won't go into detail about the implementation, but if you are curious the code is available in 'run_sign_language_word_detector_10_hand_coordinates.py'. (I will go into detail when I talk about implementing the holistic feature extraction, so stay tuned for that).

To test the model I signed all 10 words: "before", "book", "candy", "chair", "clothes", "computer", "cousin", "drink", "go" and "who".




https://user-images.githubusercontent.com/102377660/192147556-31e9adc3-3c0c-49a1-b0ef-bafb4e39dba5.mov

As before, the video freezing indicates that the clip is being processed. For now, I wasn't worried about it. 
As you can see the model was ok, but not great. It misclassified 'Before', 'Chair', and 'Go', which means it correctly guessed 7 / 10. Considering the previous feature extraction appraoch classified about half of the words consistently, this is an improvement. But I wasn't satisfied. I knew the hand tracking was imperfect and that the input features had issues. To improve things, I wanted to add more points. Namely, I wanted to also track the arms and the location of the face. For that, I used mediapipe's holistic model. 


## Holistic body position estimation 



Mediapipe offer many different models. Originally, I wanted to use the hand tracking model. This model identified the hand and fingers of one or more hands. However, sign language uses more than just the hands. 
For example, here are the signs for 'Man' and 'Woman'.
Man


https://user-images.githubusercontent.com/102377660/192033925-3354f530-2c66-4761-933d-046ee56e8c30.mov


Woman


https://user-images.githubusercontent.com/102377660/192033933-16ef8790-3e7c-4a9b-9cbe-f9c917187635.mov


Notice how similar the hand movements are. The main difference is the position of the hand relative to the face. The word 'Man' touches the thumb to the forehead then the chest. Whereas the word 'Woman' touches the thumb to the chin, then chest. Other words have similar issues. Only looking at the hands may not provide enough information to differentiate between words.

So, the position of the hands relative to the face or other parts of the body may be important.  Since the frames are cropped to isolate the person within the frame, we could assume that the person is usually centered with the top of their head near the top of the frame. 
Therefore, using the cropped frame as a reference, the position of hands relative to the person can be inferred to some extent. The problem with this is that it leaves additional information on the table. For example, in sign language, the movement of the head and facial expression can be used to convey additional meaning which could be helpful for the model. 

The best example of this is the signs for 'Yes' and 'No'. When signing these words, many of the people in the dataset also either smiled and nodded, or frowned and shook their head to reinforce the meaning of the word they were signing. 
So, it may be useful to include facial or emotion detection as additional inputs to the model as well. 

As I mentioned mediapipe had many models to choose from. Did I want to use the hand tracking model? Or did I want to use the Holistic model which is the combination of hand, pose, and facial estimation models?
Well, I wasn't sure. Yes, the face could have useful information, but not necessarily. Also, if I trained the model with the facial expression as an input, whould the model be able to function if the face is covered?
I didn't want to introduce too many variables at once, so I decided to start small and only use the hand tracking model. I could always expand the model and add other inputs later on. 
