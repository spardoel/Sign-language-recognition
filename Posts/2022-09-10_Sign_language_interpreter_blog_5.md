# Developing a sign language interpreter using machine learning Part 5 - Graduating from the alphabet to words

So far in this project I had trained a model to identify the American sign language alphabet. The model wasn't perfect, and could certainly be improved but it was good enough to act as a stepping stone for me. 
Instead of re-working the alphabet detection model I turned my attention to what was always my main goal for the project - sign language word detection. 

Now, graduating from letters to words may not seem like a big step, but trust me, it is. The main difference in terms of classification is that the dimensionality of the inputs are increasing. 
When classifying letters, a single still image contained all the necessary information and fully represented a letter. This is not the case for words. 
In sign language, words are usually not static poses but rather dynamic movements. For example, here is a video of someone signing the word 'Book'.




https://user-images.githubusercontent.com/102377660/189506312-b5fde66a-45e6-4074-bebf-dd096d4ab472.mov


As you can see, the sign involves both hands opening in front of the person as is they are holding a book. 
This meant, that instead of image classification, to identify words in sign language I needed to do video classification. 

## The hunt for data

I spent some time on Kaggle.com looking for datasets of sign language clips. There were a few useful datasets, but nothing really caught my eye. 
Then, I found the WLASL dataset and accompanying paper. 
Here is a link to the website.
https://dxli94.github.io/WLASL/

The Word-Level American Sign Language (WLASL) project includes an extremely large dataset of sign language words. The project also produced several adacemi publications including two papers published in 2020. The paper that I found most interesting is
Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong, IEEE Winter Conference on Applications of Computer Vision, 2020

The paper explains how the dataset was constructed by pulling thousands of videos from various websites. The paper discussed the lack of large scale datasets and limitations of previous attempts at word identification. The authors go on to describe how their dataset consisting of 21,083 videos was used to develop different models. They describe two approaches. The first being a baseline 3-dimentional convolutional neural network approach, and the second being a pose-based method. In the final sentence of the introduction, the authors mention that their models reached accuracies of 63% when attempting to classify a dataset of 2000 words. The paper goes into more detail regarding their classification appraoch and the previous published attempts using similar datasets. The paper is very interesting and I will likely discuss it more later on, but for now let's put it aside. For me, the initial takeaways were pretty straightforward. 
First, a huge dataset was available for me to use. This was great news. 
Second, the paper (published in 2020) only acheived a classification accuracy of 63% this means there is room for improvement and I could take a crack at an interesting challenge that hasn't already been solved. Who knows, maybe, just maybe, I could do better than 63%. 

I was filled with excitement by the idea of a new dataset, filled with determination in the face of a new challenge and filled with the overconfidence of someone who doesn't yet know what they're in for. 

## Getting the WLASL dataset

Fist of all I needed the datset. The WLASL project has their code available on github, so that was the place to start. The README file did a pretty good job explaining the structure of the data and how to obtain it. They provide a tool that downloads all the videos from various websites and adds the details to a .json file. The following is taken directly from their github page (https://github.com/dxli94/WLASL). 

Data Description
-----------------

* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.

* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.

* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.

* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

* `instance_id`: *int*, id of the instance in the same class/gloss.

* `signer_id`: *int*, id of the signer.

* `source`: *str*, a string identifier for the source site.

* `split`: *str*, indicates sample belongs to which subset.

* `url`: *str*, used for video downloading.

* `variation_id`: *int*, id for dialect (indexed from 0).

* `video_id`: *str*, a unique video identifier.


As you can see, each video has several datafields accompanying it. The main ones I was interested in were 'gloss', which is the name of the words being signed, 'bbox' which is the box generated by YOLOv3 to identify the person in the frame, and 'frame start' and 'frame end' which are the frame numbers that encoumpas the desired sign. 

After reading through the provided documentation, I cloned the repository and started looking through the provided files. 
To obtain the dataset, I needed to run the video_downloaded.py. This took several hours (close to 12 actually) but eventually I had 17,266 video files to work with. During the video fetching, several thousnd videos were unavailable or had broken links. Although unfortunate, I was happy with the data I had to work with so I didn't spent any time trying to recover the lost videos. Instead I started to explore the data. 

## The raw dataset 

After the download had finished, I had a folder with several thousand videos. Since I do not know sign language and the videos are not labeled, the folder was a confusing place for me. 
Here is a picture of the data folder.

![raw dataset](https://user-images.githubusercontent.com/102377660/189526125-1aab2ef8-1f81-4856-bbc6-0fcf1f4b1340.JPG)

As you can see the videos are not labeled. To identify the videos they needed to be matched to the corresponding entry in the json file. 

## Match the videos to the labels

Starting from some example code provided by the WLASL team, I wrote the following code to identify the videos. 

```
import json
import os

import numpy as np
import pandas as pd

# the path to the dataset
DATASET_PATH = "data/"

# the json file with the dataset details
labels_file_path = "WLASL_v0.3.json"

# open the json file and load its content
with open(labels_file_path) as ipf:
    content = json.load(ipf)


# create the empty variables
labels = []  # for the gloss, aka video label
video_file_names = []  # for the video name, i.e. identification number
box_location_x1y1x2y2 = []  # stores the box made by yolov3 to identify person
frame_start = []  # the video frame corresponding to the sign start
frame_end = []  # the video frame corresponding to the sign end (-1 means endof video)

# list all videos in the dataset
available_videos = os.listdir(DATASET_PATH)

# choose the number of labels to use
num_classes = 15

# loop through the first x glosses in the jsaon file
for ent in content[:num_classes]:

    # loop through all samples in the json file for this label
    for inst in ent["instances"]:

        # construct the video name from the json file
        vid_id = inst["video_id"]
        vid_file_name = vid_id + ".mp4"

        # check if the video is in the dataset folder
        if vid_file_name in available_videos:

            # if the video is in the dataset, save the desired details
            labels.append(ent["gloss"])
            video_file_names.append(os.path.join(DATASET_PATH, vid_file_name))
            box_location_x1y1x2y2.append(inst["bbox"])
            frame_start.append(inst["frame_start"])
            frame_end.append(inst["frame_end"])


# combine the lists into a pandas dataframe. First convert to pandas Series and then concatenate.
df = pd.concat(
    [
        pd.Series(video_file_names, name="video_paths"),
        pd.Series(labels, name="video_labels"),
        pd.Series(box_location_x1y1x2y2, name="box_x1y1x2y2"),
        pd.Series(frame_start, name="frame_start"),
        pd.Series(frame_end, name="frame_end"),
    ],
    axis=1,
)

# count the times each label (category) appears
print(df["video_labels"].value_counts())
```
After some initial imports, the code sets the location of the data folder and the json file. Then, the json data and the list of videos in the dataset are loaded. The num_classes variable is used to limit the number of classes to use. In this case it is set to 15 which corresponds to the first 15 glosses (labels) in the json file. Next, loop through the desired number of classes (glosses). For each one list the videos that are a part of that class and check to see if they are available in the local database directory. If the video is there, save the video path, label, crop-box coordinates and start and end frame numbers. Once that is done, convert the lists to a pandas dataframe and display the number of videos in each class. Here is the output. 

```
drink       28
computer    27
before      24
go          23
thin        20
cousin      19
deaf        19
who         17
candy       17
fine        17
help        17
no          16
book        14
chair       14
clothes     10
```
The output is the list of the first 15 glosses (labels) and the number of videos for each. There are between 10 and 28 videos for each label - not too bad, but not great either. More videos per class would be good, but for now it should be enough. 

## Loading and preparing the data

At this point I had some decisions to make. Did I want to develop a custom CNN model from scratch, or try to replicate the methods used in the WLASL paper? 
I briefly considered building a custom, super deep 3-dimensional CNN, but then remembered I only have 16 Gb or ram and am using a 7 year old machine. Training very complex models might use more memory than I have available and might take hours to train... No, I was better off starting small and adding complexity only if needed. 

I decided to use a pre-trained image classification model as a feature extractor and then train a handful of recurrent layers on top of that. For this I used the pre-trained Keras applications models, but more on that later. 

### Train / test partitions

To split the data into subsets I used the pandas dataframe with the video paths and labels. 
```
# use 80% of the data for training. Within that training set, use 20% for validation and 20% for testing.
train_split = 0.8
val_split = 0.25

# use scikit learn's train test split function to generate testing data
intermediate_df, test_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)

# from the remaining data, generate the training and validation sets
train_df, valid_df = train_test_split(intermediate_df, train_size=1 - val_split, shuffle=True, random_state=123)

# Print the number of samples in each set to confirm the intended proportions
print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}, Validation samples: {len(valid_df)}")
```
I set the train / test split to be 80%. Then, from the training data I took 25% for validation. I did this using scikit-learn's train_test_split() function twice, as shown above. 

### Feature extraction

After deciding to use a transfer learning appraoch I had some functions to set up. 

As a quick aside, in this part of the project I was using separate functions instead of classes. This was for a few reasons, for one it was faster for me to write and deploy individual functions. Second, I knew that my approach may need to change drastically if things weren't working so I didn't want to invest a lot of time into planning an OOP architecture just to scrap it a few days later. Third, this was not the final step in the process. I had plans for additional steps after the words level classification, and code I was writing was more of a stepping stone towards my final plan. In that final version I intended to use an OOP approach, but for now I wanted to keep things basic. 

Right, so I had a pandas dataframe with the video paths and labels. To work with the videos I needed a function to load the video, crop the frame around the person to remove excess background, and also trim the video clips to isolate the desired sign. The load_video() function does just that. 

```
IMG_SIZE = 350
MAX_SEQ_LENGTH = 50  # frame rate is 25, so 50 frames is 2 seconds of video

# Define the crop function
def crop_center_square(frame, box_coordinates):
    x1, y1, x2, y2 = box_coordinates # unpack the coordinates
    return frame[y1:y2, x1:x2]

# define the function that loads the video from the provided path then crops and resizes the video.
def load_video(path,max_frames=0,resize=(IMG_SIZE, IMG_SIZE),crop_box=(0, 0, 0, 0),start_end_frames=(1, -1)):
    cap = cv2.VideoCapture(path)

    frames = []
    counter = 1
    # use try / finally to automatically close the VideoCapture
    try:
        while True:
            # while new frames are available, load the next frame
            ret, frame = cap.read()

            if not ret:
                # if no frames are available break the loop
                break
            
            # if the current frame number is larger than the start frame
            if counter >= start_end_frames[0]:
            
                # prepare the frame
                frame = crop_center_square(frame, crop_box)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :]
                
                # add the frame to the list
                frames.append(frame)
                counter += 1 # increment the frame counter

            # if the length of 'frames' is equal to the desired max number of frames break the loop.
            # Also break the loop with the frame counter is larger than the end of the desired clip, but only if the end of the desired clip is larger than-1
            # -1 indicated the end of the video, so only stop the loop early if the sign clip ends before the end of the video
            
            if len(frames) == max_frames or (counter >= start_end_frames[1] and start_end_frames[1] > 1):
                break
    finally:
        cap.release()
    return frames

```

The load_video() function accepts the video path, the maximum number of frames allowed, the desired image output size, and coordinates of the desired crop-box, and the start and end frame numbers. The function first opens the video using openCV VideoCapture(). Next while new frames frames are available a new frame is loaded from the video. If the frame number counter is greater than the 'start frame' then process the frame and save it. Processing includes cropping, resizing and appending to the 'frames' list. Next, check the stopping conditions. The loop with stop early if the maximum number of frames is reached, or if the 'end frame' is reached provided that the 'end_frame' is different than the last frame of the video. A frame end value of -1 indicates that the desired clip goes to the end of the video, so the stopping condition using the frame end value is only relevant if the value is not -1.
