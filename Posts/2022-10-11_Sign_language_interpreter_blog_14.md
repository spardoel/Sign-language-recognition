# Developing a sign language interpreter using machine learning Part 15 - Testing the final model

Alright, the model was trained, the code has been explained, it was time to test things. 

## Testing methodology

Like my previous test, I signed each of the 100 words and counted how many guesses the model needed before correctly identifying the sign. More than 10 guesses counts as a failure for that word. 
The model detected 78 words on the first try and 94 of the words in 3 tries or fewer. Pretty good! 

As a quick note, the last test revealed that some words were very similar to one another, such as 'cool' and 'apple'. In this new model, the sign for 'cool' was replaced with an alternate sign that was more distinct. 
Similarly, the previous model completely failed to identify the word 'pizza', an alternate sign was used for this word as well. 

Here is a plot of the results. 

![100_word_test_holistic_OOP](https://user-images.githubusercontent.com/102377660/195115824-d86929d8-c10c-4b1c-b3ec-7bdedf661292.png)

And here is a better plot of the results. This one shows the words that the model struggled to classify. 

![problematic words results bar graph OOP](https://user-images.githubusercontent.com/102377660/195115806-94226035-6371-4d91-805f-aa149b1c471e.png)

From this plot, the words 'clothes', 'graduate', 'help', 'language', 'like' and 'paper' were difficult to classify. 
For the most part, these words were not surprising. Words 'paper' and 'school' are very similar and easy to confuse, 'like' is similar to 'white', 'clothes' could be easily confused with 'jacket' etc. 
Here is what I mean. 

Here is the sign for School. 


https://user-images.githubusercontent.com/102377660/195191618-d5e1b533-51a8-4404-b999-a3a8e62d4d1e.mov


And here is the sign for paper. 


https://user-images.githubusercontent.com/102377660/195191628-bc6963e6-24b0-4700-a54a-ce48c6720e92.mov


See what I mean? The signs are very similar. 

Overall I think the model did pretty well! But enough of the methodical testing, let's try some sentences!

## Sentence tests 

First of all, I want to acknowledge that sign language has its own gramar and sentence structure.
I do not know the proper sign language sentence structure.
The 'sentences' I am signing below are just a few words strung together to test the code.

Here is an example. 


https://user-images.githubusercontent.com/102377660/195191971-30f332e9-4b51-4b68-9bbf-4d60516ebd10.mov


In that video I signed 'Mother want son study but son decide play basketball'. Not too bad! 
The model wasn't perfect though, it struggled with the word 'Want' and recording that video took several tries. Here is another example sentence.


https://user-images.githubusercontent.com/102377660/195192169-028c5d6e-67d7-407e-9562-5c1aaba0c1d1.mov


This time I signed 'Doctor tell tall man eat blue medicine before bed'. As you can see in the video, the model had a hard time classifying medicine. 

Another test sentence. 


https://user-images.githubusercontent.com/102377660/195192384-0689535f-8309-444e-aa2d-f764fe6588f0.mov


In this one I signed, 'Woman cheat bowling, no pizza later'. 

Overall I was pleased by the classification. However, the 2 second classification interval was too much. The code was set up to record 2 second of video then guess the word. As you can see in the videos, I was usually done signing early and had to wait for the remainder of the 2 seconds to be recorded and the model to classify the word. To speed things up I decreased the classification interval. Here is an example of what I mean. 



https://user-images.githubusercontent.com/102377660/195193617-db1d1aa0-c4ad-479a-b250-c52e35d51b1e.mov


In this video I signed, 'Man meet son give jacket before school'. As you can see, the interval between classifications is much shorter. I think this looks much more natural. The excessively long pauses between classifications was awkward. 

To do this I simply neded to record the desired number of video frames, then pad the feature vector with zeros and set the masks variable to match the number of actual video frames. 

For demonstration purposes, here is a test video with the cropping frame and points of interest visible. 


https://user-images.githubusercontent.com/102377660/195195275-11f29131-4709-484d-a0e5-f2cc1243aaed.mov


## Conclusion

Wow, what a journey. At the beginning of this project the goal was simply to classify static images of the sign language alphabet. Now, the program can identify sign language words in real time! 

Unfortunately, I wasn't able to beat the performance of the research team who published the dataset. But then again, I didn't really try that hard... The dataset would have needed a lot of manual editing and as a wise man once said 'ain't nobody got time for that'. But seriously, the 100 word vocabulary I used gives a good overview of my approach and it demonstrates the program well. Certainly the code would be improved and sped up more, and I may come back and tinker with it from time to time, but for now I am happy to leave it as is.

I have enjoyed this project and I hope you enjoyed reading it. 



