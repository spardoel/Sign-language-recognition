# Developing a sign language interpreter using machine learning Part 12 - Increasing the holistic feature model vocabulary

In the previous post I talked about switching from a generic feature extractor model to a human pose estimator. I used Mediapipe's Holistic model to track the position of the head, shoulders, arms, hands, and fingers. 
These positional coordinates were then used as input features for a recurrent neural network classifier. The model did really well! As a refresher, (and because I want to show off the model), here is the vedio of me testing the 10 word holistic feature model. 


https://user-images.githubusercontent.com/102377660/192385777-171b6fb8-8693-4c32-bd3e-73fd65e5c7d1.mov

In the video, I signed the words, "before", "book", "candy", "chair", "clothes", "computer", "cousin", "drink", "go", "who", and the model correctly guessed all of them. 

This was great, but not particularily useful. Perhaps this model could be used in some very specific setting to identify a few hand signs, but my goal was to make a more general sign language identification model. I needed more words. 

Luckily, I had the data ready to go. Before switching to a pose estimation approach, the generic feature extraction model was trained on a 100 word dataset. That model had not performed very well, but maybe this time around things will be better. 

## Training the holistic model with 100 words

Training the model on the larger dataset was pretty trivial. The dataset was already prepared, the feature extraction script was written in the previous post, and the model training was pretty straight forward. 
For the most part, I just needed to train the model multiple times and tweak the parameters. 
