# Developing a sign language interpreter using machine learning Part 12 - Increasing the holistic feature model vocabulary

In the previous post I talked about switching from a generic feature extractor model to a human pose estimator. I used Mediapipe's Holistic model to track the position of the head, shoulders, arms, hands, and fingers. 
These positional coordinates were then used as input features for a recurrent neural network classifier. The model did really well! As a refresher, (and because I want to show off the model), here is the vedio of me testing the 10 word holistic feature model. 


https://user-images.githubusercontent.com/102377660/192385777-171b6fb8-8693-4c32-bd3e-73fd65e5c7d1.mov

In the video, I signed the words, "before", "book", "candy", "chair", "clothes", "computer", "cousin", "drink", "go", "who", and the model correctly guessed all of them. 

This was great, but not particularily useful. Perhaps this model could be used in some very specific setting to identify a few hand signs, but my goal was to make a more general sign language identification model. I needed more words. 

Luckily, I had the data ready to go. Before switching to a pose estimation approach, the generic feature extraction model was trained on a 100 word dataset. That model had not performed very well, but the dataset could be reused for a new model. Maybe this time around, things will be better. 

## Training the holistic model with 100 words

Training the model on the larger dataset was pretty trivial. The dataset was already prepared, the feature extraction script was written in the previous post, and the model training was pretty straight forward. 
For the most part, I just needed to train the model multiple times and tweak the parameters. 
I ended up with the following model architecture.
```
    # create the initial RNN layer
    x = keras.layers.GRU(256, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    # add a Additional RNN layers
    x = keras.layers.GRU(512, return_sequences=True)(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.GRU(512, return_sequences=True)(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.GRU(512)(x)

    # add several dense layers with dropout
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.55)(x)
    # define the output layer
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)
```
The model starts with four GRU layers with dropout, followed by three additional dense layers also with dropout. Here is the graph of the training results. 

![GRU_100_holistic_4GRU_layers](https://user-images.githubusercontent.com/102377660/192537988-c9e95f06-dac1-411e-ad38-29db15a98ce5.png)

As you can see the training is pretty smooth. The validation accuracy reached 91.74%. 

## Testing on real video

The validation accuracy was good, but I needed to test the model on my live webcam data. I implemented the model and started signing. The model seemed to do well, but it was hard to get a feeling for how well since there were so many words. To make things a bit more quantitative, I went through each of the words and recorded the number of guesses the model needed before correctly identifying the word. The number of guesses maxed out at 10. If after 10 guesses the model still hadn't correctly classified the word, then this was considered a failure. I kept track of the results in an excel spreadsheet, then plotted the results using matplotlib, with the following code. 
```
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("100_word_test_results.csv")

plt.bar(x=results["word"], height=results["attempts"])
plt.xticks(rotation=90, ha="left")
plt.title("Number of attempts needed to identify each word")
plt.ylabel("Number of attempts", fontweight="bold")
plt.xlabel("Word being signed", fontweight="bold")

plt.show()
```
Here is a simple bar graph of the results. 

![100 word results bar graph](https://user-images.githubusercontent.com/102377660/192540729-f68ad6a9-e773-43de-b5dd-6d7788f076ec.png)


The X axis is the list of words that I signed, and the Y axis is the number of guesses the classifier needed before correctly identifying the word. As you can see, the majority of the words were identified on the first try. In fact, 77 of the 100 words were identified on the first try, and 87 of the 100 words were identified with 3 guesses or fewer. Pretty good I think! But what about the words that the model could not identify? Let's take a closer look. 

### Investigating the difficult to classify words

To narrow down the number of words to examine, I filtered the results according to the number of attempts. I used the following code to plot the words that took more than 5 attempts to identify. 

```
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("100_word_test_results.csv")

plt.bar(
    x=results["word"][results["attempts"] > 5],
    height=results["attempts"][results["attempts"] > 5],
)
plt.xticks(rotation=90, ha="left")
plt.title("Number of attempts needed to identify each word")
plt.ylabel("Number of attempts", fontweight="bold")
plt.xlabel("Word being signed", fontweight="bold")

plt.show()
```
Which generated the following bar graph.

![problematic words results bar graph](https://user-images.githubusercontent.com/102377660/192541690-3a9d0630-d63c-425b-aacb-84bdce42755c.png)

Alright, this graph shows that the words 'Cool', 'Corn', 'Letter', 'Pizza', 'Play', 'Short', and 'What' all took 10 or more attempts to classify. These are the words that the model could not recognize. 

I started my investigation by checking my notes. During the test, if the model was consistently mistaking a word for a diferent one, I took note of the incorrect guesses. According to my notes, when I signed 'Cool' the model always guessed 'Apple', and when I signed 'Corn', the model always guessed 'Hearing'. There were a few other examples, but let's start with these two. 

Ok, so 'Cool' is confused with 'Apple', and 'Corn' is confused with 'Hearing'. Let's take a look at the sample clips for these words in the dataset. 

Here is an example of 'Cool'

https://user-images.githubusercontent.com/102377660/192545203-9b2049a9-0eee-4040-ab5a-0f223341872a.mov

Versus 'Apple' 

https://user-images.githubusercontent.com/102377660/192545226-3316c3cc-34f4-4b41-a8c2-73cc0ac16c97.mov

Now, here is an example of 'Corn'

https://user-images.githubusercontent.com/102377660/192545249-982778b2-efbd-4df9-9a4e-9be3ed8a628e.mov

Versus 'Hearing' 

https://user-images.githubusercontent.com/102377660/192545271-3e3233fc-a415-4558-9746-aa06650b4896.mov

Hmmm. I think I understand the problem. The signs are extremely similar. The signs for 'Cool' and 'Apple' are nearly identical. The signs for 'Corn' and 'Hearing' both involve the index finger extended out in front of the mouth. The difference is with the motion of the hand. For the word 'Hearing' the hand moves in a circular motion in front of the mouth, whereas for 'Corn' the hand moves sideways. Given the similarity between the signs, I am not surprised that the model misclassified these words. But what to do about it? 

Way back in one of the previous posts, I talked about different variations for words. These variations were different signs that represented the same word. When I was preparing the dataset, I chose the version of the word that was the most commonly used in real life, or that had the best representation in the dataset (the most videos). With that in mind, perhaps there are different variations of the words that could be used which would be more distinct.

### Code optimization 

Other than changing the dataset to reduce sign similarity, I also wanted to improve the model speed. As you can see in the video at the beginning of this post, there is a pause of a few seconds when the model is being run. I wanted to eliminate this pause. I'll save that for the next post. 

## Wrap up
In this post I presented the results of the word classification model. Overall, the model was pretty good! The 10 word model could identify all of the words and the 100 word model identified 77% of the words on the first try, and 87% of the words in 3 or fewer tries. 

I could stop here, but I wasn't satisfied just yet. The model was too slow. After collecting 2 seconds of video, the code needed a few seconds to process before guessing the word that I signed. The model is run on a live webcam feed that is shown to the user. With the delay caused by classification, the video pauses for several seconds before every classification. I wanted to eliminate this pause. More on that in the next post. Thanks for reading! 
