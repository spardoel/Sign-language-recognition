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

And here is the sign for paper. 

See what I mean? The signs are very similar. 

Overall I think the model did pretty well! But enough of the methodical testing, let's try some sentences!

## Sentence tests 

First of all, I want to acknowledge that sign language has its own gramar and sentence structure.
I do not know the proper sign language sentence structure.
The 'sentences' I am signing below are just a few words strung together to test the code.


