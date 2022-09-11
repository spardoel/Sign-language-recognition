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
For the data augmentaion I used the Vidaug library from github (https://github.com/okankop/vidaug). The repository home page does a good hob illustrating the different augmentation appraoches included in the library. 






