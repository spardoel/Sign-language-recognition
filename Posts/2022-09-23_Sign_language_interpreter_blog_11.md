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

As you can see, their software takes a video of someone riding on an indoor trainer and uses pose estimation to track the movement of their body and limbs. 
That brief introduction out of the way, let's look at Mediapipe.

## Mediapipe

Mediapipe is an open source computer vision framework made for pose estimation and object detection. It offers several different models including facial detection, hand tracking, and whole body tracking (called holistic). 
My previous attempt at sign languge word identification was not great. To improve the model I planned to replace the generic feature extractor model with a mediapipe model that identified the coordinates of anatomical landmarks. 
I was hoping that the positional coordinates of the fingers and hands would be better than the generic features when fed into my custom word classifier. 

Mediapipe offer many different models. Originally, I wanted to use the hand tracking model. This model identified the hand and fingers of one or more hands. However, sign language uses more than just the hands. 
For example, here are the signs for 'Man' and 'Woman'.
Man


https://user-images.githubusercontent.com/102377660/192033925-3354f530-2c66-4761-933d-046ee56e8c30.mov


Woman


https://user-images.githubusercontent.com/102377660/192033933-16ef8790-3e7c-4a9b-9cbe-f9c917187635.mov


Notice how similar the hand movements are. The main difference is the position of the hand relative to the face. The word 'Man' touches the thumb to the forehead then the chest. Whereas the word 'Woman' touches the thumb to the chin, then chest. 
There are also other examples of similar movements in the dataset. So, the position of the hands relative to the face may be important. I still intended to crop the frames to isolate the person within the frame so we could assume that the person is usually centered with the top of their head near the top of the frame. 
Therefore, using the cropped frame as a reference, the position should be relatively constant. The problem with this is that is leaves additional information on the table. 
In sign language the novement of the head and facial expression can be used to convey additional meaning which could be helpful for the model. 

The best example of this is the signs for 'Yes' and 'No'. Then signing these words, many of the people in the dataset also either smiled and nodded, or frowned and shook their head to reinforce the meaning of the word they were signing. 
So it may be useful to include facial or emotion detection as additional inputs to the model. 

As I mentioned mediapipe had many models to choose from. Did I want to use the hand tracking model? Or did I want to use the Holistic model which is the combination of hand, pose, and facial estimation models?
Well, I wasn't sure. Yes, the face could have useful information, but not necessarily. Also, if I trained the model with the facial expression as an input, whould the model be able to function if the face is covered?
I didn't want to introduce too many variables at once, so I decided to start small and only use the hand tracking model. I could always expand the model and add other inputs later on. 

## Hand tracking

I had decided to use the mediapipe hand tracking model. Here is a link to the website https://google.github.io/mediapipe/solutions/hands.html.
The page gives an overview of the function of the model and a bit of example code. This made it extremely easy to test for myself. After downloading the basic test script I recorded the following video.


https://user-images.githubusercontent.com/102377660/192034316-aaabfbe7-3f8e-4183-a863-746e90e1cfba.mov


Pretty cool! As you can see the model hand tracking isn't perfect, and it gets confused when the hands are on top of each other, or when the hand is not fully visible.
I was impressed by the hand tracking but I wasn't 100% convinced that it would be enough for the word detection model.

### Testing hand tracking on the dataset

I wanted to see if the hand tracking model could be run on the videos in the dataset. I used the mediapipe example code as a starting place and modified some of my existing code. In a few minutes I was able to generate videos from the database with the hand tracking visualization. 
