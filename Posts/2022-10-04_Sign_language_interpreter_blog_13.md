# Developing a sign language interpreter using machine learning Part 13 - Refactoring to improve program speed

Asa a refresher, here is a video of the current lign language word identification program. 

https://user-images.githubusercontent.com/102377660/193940632-c7080bd4-08bb-46d8-9c6b-0f3fbec07b88.mov

As you can see, the video freezes for a few seconds before the next guess is printed in the top lefthand corner of the frame. I wanted to eliminate this delay. 
This post will be about the process of refactoring the code and how I needed to generate a new dataset and train a whole new model. 

## What is taking so long? 

I wanted to speed up the code. Well that's great, but first I needed to identify which parts of the code where slowing down the program. I imported the time module and checked various sections of the code. 

Here is a simplified bit of pseudo code ilustrating the function of the main() function when the porgram is run. 

```
while True:

  # Get the next video frame from the camera
  bounding_box, frame = camera.get_frame()
  
  if clip is ready            
  
    # process the video and crop to largest bounding box 
    frame_features, frame_mask, new_clip = camera.process_clip()
    
    # predict the label of the video
    predicted_sign = loaded_model.predict([frame_features, frame_mask])[0]
    
    print(predicted_sign)

```
While the code is running, new frames are taken from the webcam. When a clip is ready, the clip is processed. The delay in the program comes during the execution of process_clip(). 

Here is a simplified outline of the process_clip() method. 

```
    def process_clip(self, frames, bounding_boxes):

        # find the largest bounding box
        x1_min, y1_min = np.min(bounding_boxes, axis=0)[0:2]
        x2_max, y2_max = np.max(bounding_boxes, axis=0)[2:4]

        # crop all frames to max and resize
        clip_cropped = frames[:, y1_min:y2_max, x1_min:x2_max, :]
        
        for fr in clip_cropped:
            clip_resized.append(cv2.resize(fr, (350, 350)))

        clip = np.asarray(clip_resized)
        
        frame_features = extract_holistic_coordinates(clip)

        return (frame_features)
```
Ok, so the process_clip() method extracts the maximum bounding box. Recall that the YOLO model was used to identify the person within each frame and the bounding box was recorded. This first part of the process_clip() method finds the largest bounding box and crops the entire video clip using this box. 
