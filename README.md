# Sign-language-recognition

Examples of the final program.


https://user-images.githubusercontent.com/102377660/195370504-88c770b1-07b9-48df-b0d5-594ffe99224c.mov




https://user-images.githubusercontent.com/102377660/195370511-7603f1b5-cc8b-4cd1-aadd-f17949a7b803.mov


## Project description

American sign language is commonly used by the deaf community in North America. The language is entierly visual and involves making complex gestures with the hands. 

My goal for this project was to create a sign language interpretation program that could recognize American sign language letters and words.
I wanted to use various deep learning models and methods such as convolutional and recurrent networks. I also wanted to practice with libraries such as TensorFlow, Keras and OpenCV.

The project is separated into multiple phases. 

### Phase 1 - ASL alphabet recognition

![image](https://user-images.githubusercontent.com/102377660/188241142-5a4b53ac-6798-4414-ba48-04d25f66d2d6.png)

For this first phase I had 2 main goals. 

1. Train a model that can classify still images of ASL letters. 
2. Run the model in real time using a live video from my webcam.

Blog posts 1 - 4 are related to Phase 1. 


1. Introduction
2. Building a basic CNN model
3. Testing the model
4. Improving the model


The following code files contain the code for this phase:

```
"Code files/sign_alphabet_train_model.py"
"Code files/run_sign_language_alphabet_detector.py"
```

Here is a video of the final result of Phase 1.


https://user-images.githubusercontent.com/102377660/190267410-8db5a828-2a98-4d23-995d-783c073a5c82.mp4


### Phase 2 - Word level recognition

In sign language, words are more complex than the alphabet. Here is a video of someone signing the word 'book'.

https://user-images.githubusercontent.com/102377660/190267506-7b234af2-7e89-47b8-b268-ebf50c747379.mov


Phase 2 of the project had basically the same goals but using videos instead of still images.

1. Train a model that can classify video clips of ASL words. 
2. Run the model in real time using video from my webcam. 


Blog posts 5 - 15

5. Graduating from the alphabet to words
6. Training a model for video identification
7. Reorganizing and inspecting the dataset
6. Training a model for video identification
9. Setting up the real-time video classification
10. Scaling up the model
11. Switching to a pose estimation approach
12. Increasing the holistic feature model vocabulary
13. Refactoring to improve program speed
14. Implementing the new holistic cropping approach
15. Testing the final model

Several variations and methods were used in Phase 2. A generic CNN based feature extractor was tested as well as the YOLOv5 model for object detection. In the end, both of these appraoches were abandoned in favour of the mediapipe Holistic model. The mediapipe model tracks body landmarks. Then, a custom cropping function trims each frame. Below is an example of the holistic model landmark tracking. The coordinates of these landmarks were used as features and passed to a custom classification model.  

https://user-images.githubusercontent.com/102377660/194672424-9799b4f0-4087-4aa8-9bde-ec5f5627e9f7.mov

The custom classification model used GRU layers followed by several dense layers. The classification model was trained to identify 100 different words. Below is an example of the program in action. In the below example, the program recorded 2 seconds of video then classified that 2 second clip. In later versions of the code (such as the examples at the top of the page) this 2 second interval was reduced to 1 second. 


https://user-images.githubusercontent.com/102377660/194610812-763d6af1-a9f1-400b-8f4d-af2b6e1b04a2.mov


