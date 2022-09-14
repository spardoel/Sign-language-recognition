# Sign-language-recognition

American sign language is commonly used by the deaf community in North America. The language is entierly visual and involves making complex gestures with the hands. 

My goal for this project was to create a sign language interpretation program that could recognize American sign language letters and words.
I wanted to use various deep learning models and methods such as convolutional and recurrent networks. I also wanted to practice with libraries such as TensorFlow, Keras and OpenCV.

The project is separated into multiple phases. 

## Phase 1 - ASL alphabet recognition

![image](https://user-images.githubusercontent.com/102377660/188241142-5a4b53ac-6798-4414-ba48-04d25f66d2d6.png)

For this first phase I had 2 main goals. 

1. Train a model that can classify still images of ASL letters. 
2. Run the model in real time using a live video from my webcam.

Blog posts 1 - 4 are related to Phase 1. 
The following code files contain the code for this phase:

```
"Code files/sign_alphabet_train_model.py"
"Code files/run_sign_language_alphabet_detector.py"
```

Here is a video of the final result of Phase 1.


https://user-images.githubusercontent.com/102377660/190267410-8db5a828-2a98-4d23-995d-783c073a5c82.mp4


## Phase 2 - Word level recognition

Here is a video of someone signing the word 'book'.

https://user-images.githubusercontent.com/102377660/190267506-7b234af2-7e89-47b8-b268-ebf50c747379.mov


This phase had basically the same goals but using videos instead of still images.

1. Train a model that can classify video clips of ASL words. 
2. Run the model in real time using video from my webcam. 


Blog posts 5 - ?
The following code files contain the code for this phase:

```
"Code files/preprocess_and_save_video_features.py"
"Code files/load_features_and_train_model.py"
```

