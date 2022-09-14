# Developing a sign language interpreter using machine learning Part 7 - Reorganizing and inspecting the dataset

In the previous post I talked about training the fisrt version of the sign language video classifier. In short, the model was useless. 
But I have a confession to make. I didn't thoroughly inspect the data before starting. Since the videos were not labeled and I don't know sign language I would have had to code something to associate each video with the correct label and output them in a format I could understand. 
I thought that perhaps this was unecessary and I could go straight into model development. Evidently not. So this post will go over the dataset checking and reorganization I did prior to trying to develop another model. 

## Save videos in separate class folders

The code I wrote to re-organize the dataset started the same way the previous processing code did. The video paths and other details are loaded into a dataframe. Then the vidoes are cropped and resized. 
After that, things changed a bit. Here is the main function. 
```
def process_and_save_video(df_row):
    # accept a row from the main dataframe
    video_path = df_row["video_paths"]
    video_crop_boxe = df_row["box_x1y1x2y2"]
    label = df_row["video_labels"]
    start_end_frms = (df_row["frame_start"], df_row["frame_end"])

    # load the video, also crop and resize
    frames = load_video(
        video_path,
        max_frames=MAX_SEQ_LENGTH,
        resize=(IMG_SIZE, IMG_SIZE),
        crop_box=video_crop_boxe,
        start_end_frames=start_end_frms,
    )

    # Get the current working directory
    curent_directory = os.getcwd()
    # Check if the destination folder apready exists
    new_folder_path = os.path.join(curent_directory, "data_folders", label)
    try:
        os.mkdir(new_folder_path)
    except:
        print("destination folder already exists")

    # get the video file name
    video_file = os.path.basename(video_path)  # get the video name
    # join the file name to the new save path
    new_save_path = os.path.join(new_folder_path, video_file)
    print(f"Saving: {new_save_path}")
    
    # Set up the video writter. By default the fps or all videos is 25.
    out = cv2.VideoWriter(
        new_save_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (IMG_SIZE, IMG_SIZE)
    )

    # loop through each video frame and write to file. 
    for frame in frames:
        out.write(frame)
        # print(np.shape(frames[0]))

    # Release the video writer
    out.release()

# run the main function on each row of the dataframe.
df.apply(process_and_save_video, "columns")

```
The process_and_save_video() function loads each video (and crops and resizes it as before), then the video name and the label are used to create a new save path for the video. 
The video is then saved. This frunction is applied to every row in the dataset. In this case the first part of the code that loaded the available vidoes limited the number of classes to include. 

## Inspecting the reorganized dataset

After running the code for the first 10 classes, the new data subset looked like this. 
![dataset re-organization](https://user-images.githubusercontent.com/102377660/189546034-c0e7faf3-fac8-497c-b6ad-e87be964bfc3.JPG)
As you can see there is a folder for each class. Inside each folder there are the videos corresponding to that sign, with their original names.
Here is an example. 
![dataset re-organization computer](https://user-images.githubusercontent.com/102377660/189546096-b1c34fa7-bc50-4e60-911c-abe04d102022.JPG)
The keen eyed among you may notice something odd about these videos. Specifically, look at the first frames of videos 12326 - 12328. Now, I don't know sign language, but to me those videos appear to be completely different movements. 
Here are the videos themselves. 


https://user-images.githubusercontent.com/102377660/189546231-51e04d43-bd9c-437e-bafd-ffd1f459617d.mov


https://user-images.githubusercontent.com/102377660/189546232-c36cb543-48fa-4470-b541-ed67fadb8fab.mov


https://user-images.githubusercontent.com/102377660/189546233-32b6f732-b794-41b6-b25f-63c3393a4589.mov


Now, call me crazy, but those appear to be completely different. Atfer a quick google search I learned that these are all valid variations. 
All of these signs mean 'Computer' and they are used more or less commonly in different places. That was a very interesting discovery. 
It goes a long way toward explaining why the model was so poor. We were telling the model that these were the same word, and yet they are very different movements. 
To give the model the best chance of success, I went through the first 10 classes and removed different versions so that each class had a single sign. 
According to my brief search, video 12326 showing a C-shaped hand brushing past the opposite forearm is the more common version. So all the videos showing this sign were saved and the others were deleted. 

### Removing possibly confusing signs
As the section title suggests, I also removed versions of signs that were technically correct but potentially confusing for the model. 
Here is an example. All of these signs are 'Drink'.


https://user-images.githubusercontent.com/102377660/189546639-cb0bb329-a254-4ca2-b93d-5b038b480c34.mov


https://user-images.githubusercontent.com/102377660/189546640-5845b996-2988-48a6-b203-719259ce1b8b.mov


https://user-images.githubusercontent.com/102377660/189546641-623124b7-7679-4d4e-96b4-ec59cd1a5b99.mov


https://user-images.githubusercontent.com/102377660/189546642-702f7c17-f392-4568-8e94-fbca73165a7f.mov


As you might notice, video 17716 shows the person performing the sign as if they are delicately drinking tea. To you or I, the extra meaning is obvious, but to the model, the splayed fingers of the right hand and the position of the left hand (mimicking a saucer) could be confusing.
To help the model, I removed this video and other with similar variation. 

## Dataset augmentation

Now, if you recall the spectacularily terrible training of the video classification model, the validation loss was not decreasing as it should. 
The increasing validation loss is a bad sign and could indicate that more training data is required. 
Since I had just deleted all the ambiguous videos thus further reducing the size of the dataset, I turned to data augmentation to help compensate.

Data augmentation refers to the process of making copies of existing data sample but altering them in some way to artificially produce more 'new' data samples.
For the data augmentaion I used the Vidaug library from github (https://github.com/okankop/vidaug). The repository home page does a good job illustrating the different augmentation appraoches included in the library. 

My my data augmentation I wrote some new code. First, some imports 
```
import numpy as np
import cv2
import os
import random

from PIL import Image

from vidaug import augmentors as va
```
Notice that the video augmentation library is imported as va.
Next I created a few functions. First, the function that will create the data agumentor object based on the input parameter. Here is the code. 

```
def get_augmentor(augmentation):

    if augmentation == "HorizontalFlip":
        seq = va.Sequential([va.HorizontalFlip()])
    elif augmentation == "Rotate":
        seq = va.Sequential([va.RandomRotate(degrees=10)])
    elif augmentation == "Translate":
        seq = va.Sequential([va.RandomTranslate(x=60, y=60)])
    elif augmentation == "Add":
        add_amount = random.randint(10, 60)
        seq = va.Sequential([va.Add(add_amount)])
    elif augmentation == "Subtract":
        add_amount = random.randint(-60, -10)
        seq = va.Sequential([va.Add(add_amount)])
    elif augmentation == "Salt":
        salt_amount = random.randint(75, 120)
        seq = va.Sequential([va.Salt(salt_amount)])
    elif augmentation == "Pepper":
        salt_amount = random.randint(75, 120)
        seq = va.Sequential([va.Pepper(salt_amount)])

    return seq
```
I wrote this function specifically to be self explanatory. The input parameter is the name of the desired transformation. The function just checks the input string and returns an augmentation object that performs that augmentation. There is some nuance regarding the numerical values used as additional parameters. For example, the random rotation transformation requires a maximum angle of rotation. In this case I provided 10 degrees. When applied to the video, the augmentation will randompy select an amount between 0 and 10 and rotate the video accordingly. The same is true for the random translation augmentation wich generates a random value between 0 and the provided value. For the other transformations that do not perform randomization, I added that myself. For example, the Add and Subtract augmentations make the video lighter or darker by a set amount. Before passing the value to the augmentor, I randomized the amount within a specific range. 
Let's move on to the other functions. 
Here is the video loader function. 
```
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            PIL_image = Image.fromarray(frame)
            # must use the PIL Image format. Otherwise the image rotation doesn't work and corrupts the video.
            # (the flip works with numpy arrays though)

            frames.append(PIL_image)

    finally:
        cap.release()
    return frames
```
First the video path is used to create an OpenCV video capture object. Then each frame is converted to a PIL image format and appended to a list of frames. 
The video augmentation library works best with PIL formats. I tried to use numpy array format but the resulting video was garbled when performaing certain tranformations. 
Moving on to the function that generates the new save path for the transformed videos. 
```
def create_save_path(input_path, folder, suffix=""):
    # get current directory
    curent_directory = os.getcwd()

    # Check if the destination folder apready exists
    new_folder_path = os.path.join(curent_directory, "augmented", folder)
    try:
        os.mkdir(new_folder_path)
    except:
        print("destination folder already exists")

    # get the name of the video file
    video_file = os.path.basename(input_path)  # get the video name

    # Add the transformation type to the name
    new_save_path = os.path.join(
        new_folder_path, video_file[:-4] + "_" + suffix + ".mp4"
    )

    return new_save_path
```
The function accepts the video path, the folder name, and the suffix to append to the end of the video name string. 
After creating a new save folder if needed, the video name is taken from the path using os.path.basename(). Then, the new file save path is created by joining the path, the video name (with the '.mp4' removed), and underscore, the suffix, and finally the .mp4 file type. This name is then returned. 

The next function is used to save the augmented video. 
```
def write_video_to_file(save_path, frames):

    print(f"Saving: {save_path}")
    # by default the fps or all videos is 25
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (350, 350),  # assuming a square image
    )

    for frame in frames:
        out.write(np.asarray(frame))

    out.release()
```
The function accepts the save path and the video frames. Then, as previously done, a videowriter object is created. Here, the frame rate and image size are manually set, for convinience. If this script were ever integrated into a larger project, I would want these values to be automatically infered from the loaded video. Ok, moving on. 
Next I defined a list containing the desired tranformations. 
```
transformations = [
    "HorizontalFlip",
    "Rotate",
    "Translate",
    "Add",
    "Subtract",
    "Salt",
    "Pepper",
]

```
Then, the main function. 
```
for class_folder in dataset_path:
    class_path = os.path.join(DATASET_LOCATION, class_folder)

    videos_in_class = os.listdir(class_path)

    for video_file in videos_in_class:
        video_path = os.path.join(class_path, video_file)

        video = load_video(video_path)

        # for each type of augmentation

        for augment in transformations:

            seq = get_augmentor(augment)

            augmented_video = seq(video)

            save_path = create_save_path(video_path, class_folder, suffix=augment)

            write_video_to_file(save_path, augmented_video)

```
For each folder in the dataset directory, the list of available videos is generated using os.listdir. Each video is loaded in turn, and for each video, each transformation is applied. After the transformation, the video is saved. 
Here is an example of the output videos. 
First the original video. 

https://user-images.githubusercontent.com/102377660/189548009-af9a8ac9-9352-462f-9d80-958d4a8f108a.mov

Then the horizontal flip

https://user-images.githubusercontent.com/102377660/189548014-d8df58fa-c2e2-493e-a49b-39681cc2b380.mov

The translation

https://user-images.githubusercontent.com/102377660/189548802-3a695db2-7f6b-4b08-b112-310e4d69378d.mov

The salt and pepper.

https://user-images.githubusercontent.com/102377660/189548820-e41ba378-035a-4dd2-b4e1-fc04bcd7914d.mov

https://user-images.githubusercontent.com/102377660/189548824-6809a86b-d11a-4d71-b80f-131c53ce67ce.mov

The Add and subtract

https://user-images.githubusercontent.com/102377660/189548838-b083f644-789f-463b-aec3-ea8ff047e763.mov

https://user-images.githubusercontent.com/102377660/189548843-09b6fda7-13ef-4860-ab68-37c07c870dc5.mov


And, finally, the rotation

https://user-images.githubusercontent.com/102377660/189548854-a5cb64ec-fc33-4a97-bbeb-0db39a8357b0.mov


## Wrap up
With the newly generated videos I had more data to work with. It was time to take another crack at classification. 
But first a few minor changes needed to be made in order to accept the new video inputs... 
