# Developing a sign language interpreter using machine learning Part 13 - Refactoring to improve program speed

Asa a refresher, here is a video of the current lign language word identification program. 

https://user-images.githubusercontent.com/102377660/193940632-c7080bd4-08bb-46d8-9c6b-0f3fbec07b88.mov

As you can see, the video freezes for a few seconds before the next guess is printed in the top lefthand corner of the frame. I wanted to eliminate this delay. 
This post will be about the process of refactoring the code and how I needed to generate a new dataset and train a whole new model. 

## What is taking so long? 

I wanted to speed up the code. Well that's great, but first I needed to identify which parts of the code where slowing down the program. I imported the time module and checked various sections of the code. 
For this post, I will be trying to ilustrate the code wherever possible. 
