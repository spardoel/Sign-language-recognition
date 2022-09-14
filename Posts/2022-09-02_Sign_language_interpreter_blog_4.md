# Developing a sign language interpreter using machine learning Part 4 - Improving the model

When I tested the model in the next post it fell flat on its metaphorical face. 
To try to improve it I made the following changes:

1. Increased number of training samples to 1000 per class
2. Added 2 more convolutional layers to the network
3. That's it. Only those 2 changes.

I wanted to take things slow and only make minor changes between test runs. That way if something worked well (or not) I would know what precisely changed since the last iteration.

Here is a quick look at the new model architecture.
```
model = Sequential()
# 1 - Convolution
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(HEIGHT, WIDTH, CHANNELS)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 2nd Convolution layer
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 3nd Convolution layer
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 4nd Convolution layer
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening
model.add(Flatten())
# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
# Fully connected layer 2nd layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
# Final dense layer with softmax activation
model.add(Dense(num_classes, activation="softmax"))
```
I let the model train for 25 epochs which took about 20 minutes. But I hadn't tried to optimize the training speed or check for bottlenecks so the model probably could have trained faster... 
Here is a quick look at the output during training. 

```
Epoch 1/25
2022-09-02 17:34:01.991039: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8401
270/270 [==============================] - ETA: 0s - loss: 1.8806 - accuracy: 0.4729   
Epoch 1: saving model to model_weights.h5
270/270 [==============================] - 498s 2s/step - loss: 1.8806 - accuracy: 0.4729 - val_loss: 0.9041 - val_accuracy: 0.7101 - lr: 5.0000e-04
Epoch 2/25
270/270 [==============================] - ETA: 0s - loss: 0.6596 - accuracy: 0.8058  
Epoch 2: saving model to model_weights.h5
270/270 [==============================] - 21s 79ms/step - loss: 0.6596 - accuracy: 0.8058 - val_loss: 0.5034 - val_accuracy: 0.8303 - lr: 5.0000e-04
Epoch 3/25
270/270 [==============================] - ETA: 0s - loss: 0.3501 - accuracy: 0.8966 
Epoch 3: saving model to model_weights.h5
270/270 [==============================] - 21s 78ms/step - loss: 0.3501 - accuracy: 0.8966 - val_loss: 0.2639 - val_accuracy: 0.9201 - lr: 5.0000e-04
Epoch 4/25
270/270 [==============================] - ETA: 0s - loss: 0.2227 - accuracy: 0.9336 
Epoch 4: saving model to model_weights.h5
270/270 [==============================] - 21s 78ms/step - loss: 0.2227 - accuracy: 0.9336 - val_loss: 0.1475 - val_accuracy: 0.9528 - lr: 5.0000e-04
Epoch 5/25
270/270 [==============================] - ETA: 0s - loss: 0.1599 - accuracy: 0.9524 
Epoch 5: saving model to model_weights.h5
270/270 [==============================] - 21s 77ms/step - loss: 0.1599 - accuracy: 0.9524 - val_loss: 0.0933 - val_accuracy: 0.9719 - lr: 5.0000e-04
Epoch 6/25
270/270 [==============================] - ETA: 0s - loss: 0.1177 - accuracy: 0.9650 
Epoch 6: saving model to model_weights.h5
270/270 [==============================] - 20s 73ms/step - loss: 0.1177 - accuracy: 0.9650 - val_loss: 0.0533 - val_accuracy: 0.9868 - lr: 5.0000e-04
Epoch 7/25
270/270 [==============================] - ETA: 0s - loss: 0.0919 - accuracy: 0.9729 
Epoch 7: saving model to model_weights.h5
270/270 [==============================] - 20s 73ms/step - loss: 0.0919 - accuracy: 0.9729 - val_loss: 0.0740 - val_accuracy: 0.9744 - lr: 5.0000e-04
Epoch 8/25
270/270 [==============================] - ETA: 0s - loss: 0.0820 - accuracy: 0.9753 
Epoch 8: saving model to model_weights.h5
270/270 [==============================] - 23s 85ms/step - loss: 0.0820 - accuracy: 0.9753 - val_loss: 0.0818 - val_accuracy: 0.9742 - lr: 5.0000e-04
Epoch 9/25
270/270 [==============================] - ETA: 0s - loss: 0.0534 - accuracy: 0.9859 
Epoch 9: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0534 - accuracy: 0.9859 - val_loss: 0.0311 - val_accuracy: 0.9918 - lr: 5.0000e-05
Epoch 10/25
270/270 [==============================] - ETA: 0s - loss: 0.0456 - accuracy: 0.9869 
Epoch 10: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0456 - accuracy: 0.9869 - val_loss: 0.0304 - val_accuracy: 0.9914 - lr: 5.0000e-05
Epoch 11/25
270/270 [==============================] - ETA: 0s - loss: 0.0448 - accuracy: 0.9879 
Epoch 11: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0448 - accuracy: 0.9879 - val_loss: 0.0336 - val_accuracy: 0.9900 - lr: 5.0000e-05
Epoch 12/25
270/270 [==============================] - ETA: 0s - loss: 0.0445 - accuracy: 0.9869 
Epoch 12: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0445 - accuracy: 0.9869 - val_loss: 0.0278 - val_accuracy: 0.9926 - lr: 5.0000e-05
Epoch 13/25
270/270 [==============================] - ETA: 0s - loss: 0.0380 - accuracy: 0.9902 
Epoch 13: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0380 - accuracy: 0.9902 - val_loss: 0.0279 - val_accuracy: 0.9915 - lr: 5.0000e-05
Epoch 14/25
270/270 [==============================] - ETA: 0s - loss: 0.0373 - accuracy: 0.9899 
Epoch 14: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0373 - accuracy: 0.9899 - val_loss: 0.0264 - val_accuracy: 0.9925 - lr: 5.0000e-05
Epoch 15/25
270/270 [==============================] - ETA: 0s - loss: 0.0352 - accuracy: 0.9912 
Epoch 15: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0352 - accuracy: 0.9912 - val_loss: 0.0273 - val_accuracy: 0.9925 - lr: 5.0000e-05
Epoch 16/25
270/270 [==============================] - ETA: 0s - loss: 0.0332 - accuracy: 0.9914 
Epoch 16: saving model to model_weights.h5
270/270 [==============================] - 22s 81ms/step - loss: 0.0332 - accuracy: 0.9914 - val_loss: 0.0238 - val_accuracy: 0.9932 - lr: 5.0000e-05
Epoch 17/25
270/270 [==============================] - ETA: 0s - loss: 0.0328 - accuracy: 0.9916 
Epoch 17: saving model to model_weights.h5
270/270 [==============================] - 22s 82ms/step - loss: 0.0328 - accuracy: 0.9916 - val_loss: 0.0242 - val_accuracy: 0.9925 - lr: 5.0000e-05
Epoch 18/25
270/270 [==============================] - ETA: 0s - loss: 0.0306 - accuracy: 0.9922 
Epoch 18: saving model to model_weights.h5
270/270 [==============================] - 23s 86ms/step - loss: 0.0306 - accuracy: 0.9922 - val_loss: 0.0240 - val_accuracy: 0.9932 - lr: 5.0000e-05
Epoch 19/25
270/270 [==============================] - ETA: 0s - loss: 0.0283 - accuracy: 0.9931 
Epoch 19: saving model to model_weights.h5
270/270 [==============================] - 20s 75ms/step - loss: 0.0283 - accuracy: 0.9931 - val_loss: 0.0221 - val_accuracy: 0.9935 - lr: 1.0000e-05
Epoch 20/25
269/270 [============================>.] - ETA: 0s - loss: 0.0291 - accuracy: 0.9924 
Epoch 20: saving model to model_weights.h5
270/270 [==============================] - 20s 73ms/step - loss: 0.0291 - accuracy: 0.9924 - val_loss: 0.0225 - val_accuracy: 0.9935 - lr: 1.0000e-05
Epoch 21/25
270/270 [==============================] - ETA: 0s - loss: 0.0284 - accuracy: 0.9925 
Epoch 21: saving model to model_weights.h5
270/270 [==============================] - 19s 70ms/step - loss: 0.0284 - accuracy: 0.9925 - val_loss: 0.0217 - val_accuracy: 0.9935 - lr: 1.0000e-05
Epoch 22/25
270/270 [==============================] - ETA: 0s - loss: 0.0275 - accuracy: 0.9934 
Epoch 22: saving model to model_weights.h5
270/270 [==============================] - 19s 70ms/step - loss: 0.0275 - accuracy: 0.9934 - val_loss: 0.0225 - val_accuracy: 0.9936 - lr: 1.0000e-05
Epoch 23/25
270/270 [==============================] - ETA: 0s - loss: 0.0271 - accuracy: 0.9931 
Epoch 23: saving model to model_weights.h5
270/270 [==============================] - 19s 70ms/step - loss: 0.0271 - accuracy: 0.9931 - val_loss: 0.0214 - val_accuracy: 0.9939 - lr: 1.0000e-05
Epoch 24/25
270/270 [==============================] - ETA: 0s - loss: 0.0271 - accuracy: 0.9928 
Epoch 24: saving model to model_weights.h5
270/270 [==============================] - 19s 71ms/step - loss: 0.0271 - accuracy: 0.9928 - val_loss: 0.0217 - val_accuracy: 0.9933 - lr: 1.0000e-05
Epoch 25/25
270/270 [==============================] - ETA: 0s - loss: 0.0283 - accuracy: 0.9925 
Epoch 25: saving model to model_weights.h5
270/270 [==============================] - 19s 71ms/step - loss: 0.0283 - accuracy: 0.9925 - val_loss: 0.0213 - val_accuracy: 0.9935 - lr: 1.0000e-05
```
This time, the model finished training with a validation accuracy of 99%. Let's see if that translated to better performance in the real-time test. 

## Real-time sign identification test second attempt

I ran the real-time test again with the new model. Here is a video of me signing A, B, C, D, E, F.
For reference here is a picture of the ASL alphabet signs. (image from https://www.ai-media.tv/ai-media-blog/sign-language-alphabets-from-around-the-world/)
![image](https://user-images.githubusercontent.com/102377660/188241142-5a4b53ac-6798-4414-ba48-04d25f66d2d6.png)


https://user-images.githubusercontent.com/102377660/188240071-381178cd-0077-40ee-ad22-485ca38995f9.mp4

Awesome! Pretty good. Nearly perfect I'd say. 

...

Ok. You got me, I stopped signing at F because the model kind of goes off the rails after that...
Fine, if you insist, I suppose I can show you the model struggling... 
Here is a video of me attempting to sign a few more letters (F, G, H, I, J, K, L). 


https://user-images.githubusercontent.com/102377660/188240918-c5facbcf-6e09-4916-8936-ea8c2fb09caa.mp4


As you can see the model struggled a bit with G and H, had no idea aboud I and J, and correctly identified K. Then, shockingly, the model failled to identify L, instead thinking it was an A... I say shockingly, because to me L seems like one of the easiest to identify. Just goes to show you that what is obvious to you and me can be hard for computers.

## Wrap up
In the second attempt the model did much better! It identified many of the letters correctly. The model was unable to identify some of the letters, but there is a definite improvement over the previous model. While I was tweaking the model and trying to improve performance I discovered that the modeled seemed totally incapable of identifying M, N, and T. After some investigation, I think I know why. It turns out that there are some problems with the dataset. Mainly, the signs used for M, N and T are not the standard ASL signs I had found. So it was no surprise the model couldn't recognize the signs I was showing, it had never seen them before!

This marks the end of phase 1 of this project. I had created a program that can classify the sign language alphabet from a live video. Sure, the model wasn't great and had never seen the letters N,M or T before, but how important are those letter really? I was satisfied enough to move on to phase 2 of the project. I'll explain more in the next post. 



