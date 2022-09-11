# Developing a sign language interpreter using machine learning Part 5 - Graduating from the alphabet to words

So far in this project I had trained a model to identify the American sign language alphabet. The model wasn't perfect, and could certainly be improved but it was good enough to act as a stepping stone for me. 
Instead of re-working the alphabet detection model I turned my attention to what was always my main goal for the project - sign language word detection. 

Now, graduating from letters to words may not seem like a big step, but trust me, it is. The main difference in terms of classification is that the dimensionality of the inputs are increasing. 
When classifying letters, a single still image contained all the necessary information and fully represented a letter. This is not the case for words. 
In sign language, words are usually not static poses but rather dynamic movements. For example, here is a video of someone signing the word 'Book'.

https://user-images.githubusercontent.com/102377660/189506125-d551b4c3-ae12-46f7-93f3-f4cf3774d86d.mp4

As you can see, the sign involves both hands opening in front of the person as is they are holding a book. 
This meant, that instead of image classification, to identify words in sign language I needed to do video classification. 

## The hunt for data

I spent some time on Kaggle.com looking for datasets of sign language clips. There were a few useful datasets, but nothing really caught my eye. 
Then, I found the WLASL dataset and accompanying paper. 
