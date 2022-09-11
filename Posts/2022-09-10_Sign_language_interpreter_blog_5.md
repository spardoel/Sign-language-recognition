# Developing a sign language interpreter using machine learning Part 5 - Graduating from the alphabet to words

So far in this project I had trained a model to identify the American sign language alphabet. The model wasn't perfect, and could certainly be improved but it was good enough to act as a stepping stone for me. 
Instead of re-working the alphabet detection model I turned my attention to what was always my main goal for the project - sign language word detection. 

Now, graduating from letters to words may not seem like a big step, but trust me, it is. The main difference in terms of classification is that the dimensionality of the inputs are increasing. 
When classifying letters, a single still image contained all the necessary information and fully represented a letter. This is not the case for words. 
In sign language, words are usually not static poses but rather dynamic movements. For example, here is a video of someone signing the word 'Book'.




https://user-images.githubusercontent.com/102377660/189506312-b5fde66a-45e6-4074-bebf-dd096d4ab472.mov


As you can see, the sign involves both hands opening in front of the person as is they are holding a book. 
This meant, that instead of image classification, to identify words in sign language I needed to do video classification. 

## The hunt for data

I spent some time on Kaggle.com looking for datasets of sign language clips. There were a few useful datasets, but nothing really caught my eye. 
Then, I found the WLASL dataset and accompanying paper. 
Here is a link to the website.
https://dxli94.github.io/WLASL/

The Word-Level American Sign Language (WLASL) project includes an extremely large dataset of sign language words. The project also produced several adacemi publications including two papers published in 2020. The paper that I found most interesting is
Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong, IEEE Winter Conference on Applications of Computer Vision, 2020

The paper explains how the dataset was constructed by pulling thousands of videos from various websites. The paper discussed the lack of large scale datasets and limitations of previous attempts at word identification. The authors go on to describe how their dataset consisting of 21,083 videos was used to develop different models. They describe two approaches. The first being a baseline 3-dimentional convolutional neural network approach, and the second being a pose-based method. In the final sentence of the introduction, the authors mention that their models reached accuracies of 63% when attempting to classify a dataset of 2000 words. The paper goes into more detail regarding their classification appraoch and the previous published attempts using similar datasets. The paper is very interesting and I will likely discuss it more later on, but for now let's put it aside. For me, the initial takeaways were pretty straightforward. 
First, a huge dataset was available for me to use. This was great news. 
Second, the paper (published in 2020) only acheived a classification accuracy of 63% this means there is room for improvement and I could take a crack at an interesting challenge that hasn't already been solved. Who knows, maybe, just maybe, I could do better than 63%. 

I was filled with excitement by the idea of a new dataset, filled with determination in the face of a new challenge and filled with the overconfidence of someone who doesn't yet know what they're in for. 

## Getting the WLASL dataset

Fist of all I needed the datset. The WLASL project has their code available on github, so that was the place to start. The README file did a pretty good job explaining the structure of the data and how to obtain it. They provide a tool that downloads all the videos from various websites and adds the details to a .json file. The following is taken directly from their github page (https://github.com/dxli94/WLASL). 

Data Description
-----------------

* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.

* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.

* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.

* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

* `instance_id`: *int*, id of the instance in the same class/gloss.

* `signer_id`: *int*, id of the signer.

* `source`: *str*, a string identifier for the source site.

* `split`: *str*, indicates sample belongs to which subset.

* `url`: *str*, used for video downloading.

* `variation_id`: *int*, id for dialect (indexed from 0).

* `video_id`: *str*, a unique video identifier.


As you can see, each video has several datafields accompanying it. The main ones I was interested in were 'gloss', which is the name of the words being signed, 'bbox' which is the box generated by YOLOv3 to identify the person in the frame, and 'frame start' and 'frame end' which are the frame numbers that encoumpas the desired sign. 

After reading through the provided documentation, I cloned the repository and started looking through the provided files. 
To obtain the dataset, I needed to run the video_downloaded.py. This took several hours (close to 12 actually) but eventually I had 17,266 video files to work with. During the video fetching, several thousnd videos were unavailable or had broken links. Although unfortunate, I was happy with the data I had to work with so I didn't spent any time trying to recover the lost videos. Instead I started to explore the data. 

## The raw dataset 

After the download had finished, I had a folder with several thousand videos. Since I do not know sign language and the videos are not labeled, the folder was a confusing place for me. 


