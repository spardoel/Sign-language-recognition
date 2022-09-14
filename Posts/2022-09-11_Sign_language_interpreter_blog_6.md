# Developing a sign language interpreter using machine learning Part 6 - Training a model for video identification

In the previous post I briefly introduced the dataset and the data preparation methods. In this post, I will go over loading the features, and training a model for classification. 
The code for this post can be found in "Code files/loadFeaturesAndTrainModel.py"

## Load the features and masks

To start, I loaded the pickle file with the features and masks that was created in the previous post.
```
# Load the pickle file with the datasets
with open("preprocessed_videos.pkl", "rb") as f:
    (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        class_vocab,
    ) = pickle.load(f)


print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
```

Running this gave the following output. 

```
Frame features in train set: (168, 50, 1280)
Frame masks in train set: (168, 50)
```
Ok, so there are 168 videos in the training set, each video has 50 frames and 1280 features. Great. 

## Define the model

The model creation was placed inside a function called get_sequence_model(). Here it is.

```
# set the constants
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 1280

# Creation of the sequence model.
def get_sequence_model():
    # define input shapes using the number of features and max video length constants
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # create the initial RNN layer
    x = keras.layers.SimpleRNN(256, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    # add a second RNN layer
    x = keras.layers.SimpleRNN(128)(x)
    # add several dense layers with dropout
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)

    # define the output layer
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    # Create the model
    rnn_model = keras.Model([frame_features_input, mask_input], output)

    # Complie the model
    opt = Adam(learning_rate=0.0005)
    rnn_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=opt,
        metrics=["accuracy"],
    )
    # return the compiled model
    return rnn_model

```
First the constants for the number of features and the max video length are defined. Theoretically these could be inferred from the loaded data, but here they are just defined manually. 
The model is pretty straightforward. Two simple RNN layers are used followed by several dense layers with droput between each. This configuration was determined mainly based on some trial and error. The dropouts were included to help prevent overfitting.
The Adam optimizer was used with an initial learning rate of 0.0005, and categorical cross entropy was used as the loss metric. 

## Setting up the model training

To create and train the model, a new function was created. 

```
BATCH_SIZE = 8
EPOCHS = 50

# The main function to train the model
def run_experiment():
    # define some keras callbacks
    
    # define a learning rate schedule to reduce learning rate by 10 for every 2 epochs without an improvement in validation loss
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, mode="auto"
    )
    # define the early stopping callback
    early_stopping_cb = EarlyStopping(
        monitor="val_accuracy", patience=10
    )

    # get the model
    seq_model = get_sequence_model()

    # train the model
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        batch_size=BATCH_SIZE,
        validation_data=([val_data[0], val_data[1]], val_labels),
        epochs=EPOCHS,
        callbacks=[early_stopping_cb, reduce_lr],
    )

    return history, seq_model
    

# Train the model
history, sequence_model = run_experiment()
```
After setting the number of training epochs and the batch size as constants, the run_experiment() function is created. 
The function starts by defining the learning rate schedule callback and the early stopping callback. The learning rate scheduler is specifically intended to reduce the learning rate when the model training hits a plateau. The inputs for this callback determine what value to monitor, how much to decrease the learning rate by, the minimum learning rate and how many epochs must be completed without improvement before the learning rate is changed. Similarly, the early stopping callback monitors the validation accuracy and will stop the model training if there was no improvement in the last 10 epochs. Then the model is created and trained. The model training parameters are simply the training data, the batch size, the validation dataset, the maximum number of epochs and the callback functions. Finally, the run_experiment() function is called to start the model training. The function returns the training history and the trained model. The history is used for visualization of the training.

## Setting up the model evaluation and visualization

After the model finished training, I wanted to be able to visualize the training process and check a few test samples. I realize that checking 2 or 3 samples from the test set is not an adequate test. But at this point I just wanted to make sure everything was working and that I was able to run the trained model on new videos. I would need to be able to do that for the next phase I was planning, but let's not get ahead of ourselves. For now, the model visualization was coded and a few samples were tested.  
For that the following code was added. 
```
def check_random_test_sample():

    # randomly select an sample from the test set
    test_vid_num = random.randint(0, len(test_labels) - 1)
    
    # get the true label of the sample
    test_video_label = test_labels[test_vid_num]

    # Print the true label
    print(f"Test video label: {class_vocab[test_video_label[0]]}")
    
    # pass the sample video and video number (index) to the prediction function
    sequence_prediction(test_data, test_vid_num)

def sequence_prediction(test_data_video, test_vid_number):

    # get the frames and masks from the input video
    frame_features = test_data_video[0][test_vid_number][np.newaxis, :]
    frame_mask = test_data_video[1][test_vid_number][np.newaxis, :]
    
    #frame_mask2 = frame_mask[np.newaxis, :]
    #frame_features2 = frame_features[np.newaxis, :, :]
    
    # Run the model and get the probability that the video belongs to each class
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    # display the probabilities
    print("Model prediction probabilities:")
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")


check_random_test_sample()
check_random_test_sample()
check_random_test_sample()
```
To evaluate the model the check_random_test_sample() and  sequence_prediction() functions were created. These select a random sample from the test set and classify the video using the trained model. 
After this, the training history is plotted uing matplotlib as shown below. 
```

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 5.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

```

## Running the code

Alright, with those functions out of the way it was time to run the code and train the model! 
For this first run, the batch size was set to 8 and the model was limited to 50 epochs. Here was the output. 
```
Epoch 1/50
21/21 [==============================] - ETA: 0s - loss: 2.7254 - accuracy: 0.0833      
Epoch 1: val_loss improved from inf to 2.71591, saving model to video_classifier.h5
21/21 [==============================] - 8s 218ms/step - loss: 2.7254 - accuracy: 0.0833 - val_loss: 2.7159 - val_accuracy: 0.1228 - lr: 5.0000e-04
Epoch 2/50
21/21 [==============================] - ETA: 0s - loss: 2.7093 - accuracy: 0.0893    
Epoch 2: val_loss improved from 2.71591 to 2.70548, saving model to video_classifier.h5
21/21 [==============================] - 4s 171ms/step - loss: 2.7093 - accuracy: 0.0893 - val_loss: 2.7055 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 3/50
21/21 [==============================] - ETA: 0s - loss: 2.7103 - accuracy: 0.0893
Epoch 3: val_loss did not improve from 2.70548
21/21 [==============================] - 3s 164ms/step - loss: 2.7103 - accuracy: 0.0893 - val_loss: 2.7365 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 4/50
21/21 [==============================] - ETA: 0s - loss: 2.6765 - accuracy: 0.1190
Epoch 4: val_loss did not improve from 2.70548
21/21 [==============================] - 4s 172ms/step - loss: 2.6765 - accuracy: 0.1190 - val_loss: 2.7159 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 5/50
21/21 [==============================] - ETA: 0s - loss: 2.7178 - accuracy: 0.0952
Epoch 5: val_loss did not improve from 2.70548
21/21 [==============================] - 4s 170ms/step - loss: 2.7178 - accuracy: 0.0952 - val_loss: 2.7111 - val_accuracy: 0.0526 - lr: 5.0000e-04
Epoch 6/50
21/21 [==============================] - ETA: 0s - loss: 2.6951 - accuracy: 0.0833    
Epoch 6: val_loss did not improve from 2.70548
21/21 [==============================] - 3s 162ms/step - loss: 2.6951 - accuracy: 0.0833 - val_loss: 2.7141 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 7/50
21/21 [==============================] - ETA: 0s - loss: 2.6912 - accuracy: 0.1071
Epoch 7: val_loss did not improve from 2.70548
21/21 [==============================] - 4s 174ms/step - loss: 2.6912 - accuracy: 0.1071 - val_loss: 2.7136 - val_accuracy: 0.0526 - lr: 5.0000e-04
Epoch 8/50
21/21 [==============================] - ETA: 0s - loss: 2.6902 - accuracy: 0.0833
Epoch 8: val_loss did not improve from 2.70548
21/21 [==============================] - 3s 167ms/step - loss: 2.6902 - accuracy: 0.0833 - val_loss: 2.7239 - val_accuracy: 0.0175 - lr: 5.0000e-04
Epoch 9/50
21/21 [==============================] - ETA: 0s - loss: 2.6924 - accuracy: 0.0774    
Epoch 9: val_loss improved from 2.70548 to 2.69942, saving model to video_classifier.h5
21/21 [==============================] - 4s 183ms/step - loss: 2.6924 - accuracy: 0.0774 - val_loss: 2.6994 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 10/50
21/21 [==============================] - ETA: 0s - loss: 2.6943 - accuracy: 0.0833
Epoch 10: val_loss did not improve from 2.69942
21/21 [==============================] - 3s 161ms/step - loss: 2.6943 - accuracy: 0.0833 - val_loss: 2.7016 - val_accuracy: 0.0175 - lr: 5.0000e-04
Epoch 11/50
21/21 [==============================] - ETA: 0s - loss: 2.6980 - accuracy: 0.1131
Epoch 11: val_loss did not improve from 2.69942
21/21 [==============================] - 4s 171ms/step - loss: 2.6980 - accuracy: 0.1131 - val_loss: 2.7147 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 12/50
21/21 [==============================] - ETA: 0s - loss: 2.6983 - accuracy: 0.1012
Epoch 12: val_loss improved from 2.69942 to 2.69883, saving model to video_classifier.h5
21/21 [==============================] - 4s 185ms/step - loss: 2.6983 - accuracy: 0.1012 - val_loss: 2.6988 - val_accuracy: 0.1404 - lr: 5.0000e-04
Epoch 13/50
21/21 [==============================] - ETA: 0s - loss: 2.6925 - accuracy: 0.1190
Epoch 13: val_loss did not improve from 2.69883
21/21 [==============================] - 3s 166ms/step - loss: 2.6925 - accuracy: 0.1190 - val_loss: 2.7067 - val_accuracy: 0.0175 - lr: 5.0000e-04
Epoch 14/50
21/21 [==============================] - ETA: 0s - loss: 2.6874 - accuracy: 0.0952    
Epoch 14: val_loss improved from 2.69883 to 2.68318, saving model to video_classifier.h5
21/21 [==============================] - 4s 182ms/step - loss: 2.6874 - accuracy: 0.0952 - val_loss: 2.6832 - val_accuracy: 0.1404 - lr: 5.0000e-04
Epoch 15/50
21/21 [==============================] - ETA: 0s - loss: 2.6906 - accuracy: 0.0893    
Epoch 15: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 167ms/step - loss: 2.6906 - accuracy: 0.0893 - val_loss: 2.7164 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 16/50
21/21 [==============================] - ETA: 0s - loss: 2.6682 - accuracy: 0.1190    
Epoch 16: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 168ms/step - loss: 2.6682 - accuracy: 0.1190 - val_loss: 2.7264 - val_accuracy: 0.0175 - lr: 5.0000e-04
Epoch 17/50
21/21 [==============================] - ETA: 0s - loss: 2.6580 - accuracy: 0.0714    
Epoch 17: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 166ms/step - loss: 2.6580 - accuracy: 0.0714 - val_loss: 2.7249 - val_accuracy: 0.0526 - lr: 5.0000e-04
Epoch 18/50
21/21 [==============================] - ETA: 0s - loss: 2.6653 - accuracy: 0.1131    
Epoch 18: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 177ms/step - loss: 2.6653 - accuracy: 0.1131 - val_loss: 2.7209 - val_accuracy: 0.0000e+00 - lr: 5.0000e-04
Epoch 19/50
21/21 [==============================] - ETA: 0s - loss: 2.6739 - accuracy: 0.1488
Epoch 19: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 172ms/step - loss: 2.6739 - accuracy: 0.1488 - val_loss: 2.6966 - val_accuracy: 0.0351 - lr: 5.0000e-04
Epoch 20/50
21/21 [==============================] - ETA: 0s - loss: 2.6633 - accuracy: 0.0714
Epoch 20: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 164ms/step - loss: 2.6633 - accuracy: 0.0714 - val_loss: 2.7119 - val_accuracy: 0.0175 - lr: 5.0000e-04
Epoch 21/50
21/21 [==============================] - ETA: 0s - loss: 2.6549 - accuracy: 0.1607    
Epoch 21: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 178ms/step - loss: 2.6549 - accuracy: 0.1607 - val_loss: 2.7206 - val_accuracy: 0.0000e+00 - lr: 5.0000e-04
Epoch 22/50
21/21 [==============================] - ETA: 0s - loss: 2.6600 - accuracy: 0.0952
Epoch 22: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 176ms/step - loss: 2.6600 - accuracy: 0.0952 - val_loss: 2.7276 - val_accuracy: 0.0877 - lr: 5.0000e-04
Epoch 23/50
21/21 [==============================] - ETA: 0s - loss: 2.5809 - accuracy: 0.1607
Epoch 23: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 180ms/step - loss: 2.5809 - accuracy: 0.1607 - val_loss: 2.7765 - val_accuracy: 0.0526 - lr: 5.0000e-04
Epoch 24/50
21/21 [==============================] - ETA: 0s - loss: 2.6208 - accuracy: 0.1190
Epoch 24: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 165ms/step - loss: 2.6208 - accuracy: 0.1190 - val_loss: 2.7294 - val_accuracy: 0.1404 - lr: 5.0000e-04
Epoch 25/50
21/21 [==============================] - ETA: 0s - loss: 2.5450 - accuracy: 0.1310    
Epoch 25: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 162ms/step - loss: 2.5450 - accuracy: 0.1310 - val_loss: 2.7750 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 26/50
21/21 [==============================] - ETA: 0s - loss: 2.4961 - accuracy: 0.2024
Epoch 26: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 162ms/step - loss: 2.4961 - accuracy: 0.2024 - val_loss: 2.8366 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 27/50
21/21 [==============================] - ETA: 0s - loss: 2.4841 - accuracy: 0.1548
Epoch 27: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 163ms/step - loss: 2.4841 - accuracy: 0.1548 - val_loss: 2.8726 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 28/50
21/21 [==============================] - ETA: 0s - loss: 2.5147 - accuracy: 0.1667
Epoch 28: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 165ms/step - loss: 2.5147 - accuracy: 0.1667 - val_loss: 2.8626 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 29/50
21/21 [==============================] - ETA: 0s - loss: 2.4795 - accuracy: 0.1964
Epoch 29: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 166ms/step - loss: 2.4795 - accuracy: 0.1964 - val_loss: 2.7936 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 30/50
21/21 [==============================] - ETA: 0s - loss: 2.3946 - accuracy: 0.1905
Epoch 30: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 166ms/step - loss: 2.3946 - accuracy: 0.1905 - val_loss: 2.8889 - val_accuracy: 0.0351 - lr: 2.5000e-04
Epoch 31/50
21/21 [==============================] - ETA: 0s - loss: 2.4019 - accuracy: 0.1429    
Epoch 31: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 173ms/step - loss: 2.4019 - accuracy: 0.1429 - val_loss: 2.9247 - val_accuracy: 0.0526 - lr: 2.5000e-04
Epoch 32/50
21/21 [==============================] - ETA: 0s - loss: 2.4029 - accuracy: 0.1548    
Epoch 32: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 167ms/step - loss: 2.4029 - accuracy: 0.1548 - val_loss: 3.1059 - val_accuracy: 0.0351 - lr: 2.5000e-04
Epoch 33/50
21/21 [==============================] - ETA: 0s - loss: 2.3374 - accuracy: 0.2083
Epoch 33: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 165ms/step - loss: 2.3374 - accuracy: 0.2083 - val_loss: 2.8907 - val_accuracy: 0.0175 - lr: 2.5000e-04
Epoch 34/50
21/21 [==============================] - ETA: 0s - loss: 2.3775 - accuracy: 0.1607
Epoch 34: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 165ms/step - loss: 2.3775 - accuracy: 0.1607 - val_loss: 2.8840 - val_accuracy: 0.0351 - lr: 2.5000e-04
Epoch 35/50
21/21 [==============================] - ETA: 0s - loss: 2.3222 - accuracy: 0.1667
Epoch 35: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 169ms/step - loss: 2.3222 - accuracy: 0.1667 - val_loss: 2.8807 - val_accuracy: 0.0351 - lr: 1.2500e-04
Epoch 36/50
21/21 [==============================] - ETA: 0s - loss: 2.2835 - accuracy: 0.2083
Epoch 36: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 165ms/step - loss: 2.2835 - accuracy: 0.2083 - val_loss: 2.8221 - val_accuracy: 0.0702 - lr: 1.2500e-04
Epoch 37/50
21/21 [==============================] - ETA: 0s - loss: 2.2628 - accuracy: 0.2024
Epoch 37: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 157ms/step - loss: 2.2628 - accuracy: 0.2024 - val_loss: 2.9456 - val_accuracy: 0.0351 - lr: 1.2500e-04
Epoch 38/50
21/21 [==============================] - ETA: 0s - loss: 2.2648 - accuracy: 0.2024
Epoch 38: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 169ms/step - loss: 2.2648 - accuracy: 0.2024 - val_loss: 2.8594 - val_accuracy: 0.0702 - lr: 1.2500e-04
Epoch 39/50
21/21 [==============================] - ETA: 0s - loss: 2.1777 - accuracy: 0.2500
Epoch 39: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 162ms/step - loss: 2.1777 - accuracy: 0.2500 - val_loss: 2.9465 - val_accuracy: 0.0351 - lr: 1.2500e-04
Epoch 40/50
21/21 [==============================] - ETA: 0s - loss: 2.1221 - accuracy: 0.2798
Epoch 40: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 160ms/step - loss: 2.1221 - accuracy: 0.2798 - val_loss: 3.0058 - val_accuracy: 0.0351 - lr: 1.2500e-04
Epoch 41/50
21/21 [==============================] - ETA: 0s - loss: 2.1411 - accuracy: 0.2798
Epoch 41: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 166ms/step - loss: 2.1411 - accuracy: 0.2798 - val_loss: 2.8510 - val_accuracy: 0.0526 - lr: 1.2500e-04
Epoch 42/50
21/21 [==============================] - ETA: 0s - loss: 2.1640 - accuracy: 0.2560
Epoch 42: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 164ms/step - loss: 2.1640 - accuracy: 0.2560 - val_loss: 2.9067 - val_accuracy: 0.0526 - lr: 1.2500e-04
Epoch 43/50
21/21 [==============================] - ETA: 0s - loss: 2.1270 - accuracy: 0.2619
Epoch 43: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 166ms/step - loss: 2.1270 - accuracy: 0.2619 - val_loss: 3.0560 - val_accuracy: 0.0526 - lr: 1.2500e-04
Epoch 44/50
21/21 [==============================] - ETA: 0s - loss: 2.0739 - accuracy: 0.2679
Epoch 44: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 166ms/step - loss: 2.0739 - accuracy: 0.2679 - val_loss: 3.0757 - val_accuracy: 0.0351 - lr: 1.2500e-04
Epoch 45/50
21/21 [==============================] - ETA: 0s - loss: 2.0748 - accuracy: 0.2738
Epoch 45: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 167ms/step - loss: 2.0748 - accuracy: 0.2738 - val_loss: 3.0543 - val_accuracy: 0.0351 - lr: 6.2500e-05
Epoch 46/50
21/21 [==============================] - ETA: 0s - loss: 2.0467 - accuracy: 0.2500
Epoch 46: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 161ms/step - loss: 2.0467 - accuracy: 0.2500 - val_loss: 3.0762 - val_accuracy: 0.0702 - lr: 6.2500e-05
Epoch 47/50
21/21 [==============================] - ETA: 0s - loss: 2.0203 - accuracy: 0.2738
Epoch 47: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 164ms/step - loss: 2.0203 - accuracy: 0.2738 - val_loss: 3.0251 - val_accuracy: 0.0877 - lr: 6.2500e-05
Epoch 48/50
21/21 [==============================] - ETA: 0s - loss: 1.9743 - accuracy: 0.2619
Epoch 48: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 163ms/step - loss: 1.9743 - accuracy: 0.2619 - val_loss: 3.1330 - val_accuracy: 0.0526 - lr: 6.2500e-05
Epoch 49/50
21/21 [==============================] - ETA: 0s - loss: 1.9554 - accuracy: 0.3214
Epoch 49: val_loss did not improve from 2.68318
21/21 [==============================] - 4s 183ms/step - loss: 1.9554 - accuracy: 0.3214 - val_loss: 2.9541 - val_accuracy: 0.1053 - lr: 6.2500e-05
Epoch 50/50
21/21 [==============================] - ETA: 0s - loss: 1.9977 - accuracy: 0.3095
Epoch 50: val_loss did not improve from 2.68318
21/21 [==============================] - 3s 164ms/step - loss: 1.9977 - accuracy: 0.3095 - val_loss: 3.0252 - val_accuracy: 0.1228 - lr: 6.2500e-05
Test video label: fine
1/1 [==============================] - 0s 474ms/step
Model prediction probabilities:
  go: 32.55%
  before: 16.19%
  computer: 13.44%
  thin:  9.37%
  no:  6.13%
  drink:  5.88%
  who:  2.92%
  deaf:  2.86%
  help:  2.76%
  candy:  2.57%
  fine:  1.98%
  book:  1.19%
  cousin:  0.77%
  clothes:  0.75%
  chair:  0.61%
Test video label: clothes
1/1 [==============================] - 0s 50ms/step
Model prediction probabilities:
  go: 36.16%
  before: 16.15%
  computer: 13.54%
  thin:  9.06%
  no:  5.67%
  drink:  5.52%
  candy:  2.56%
  who:  2.45%
  help:  2.44%
  deaf:  2.40%
  fine:  1.65%
  book:  0.90%
  clothes:  0.57%
  cousin:  0.54%
  chair:  0.40%
Test video label: before
1/1 [==============================] - 0s 63ms/step
Model prediction probabilities:
  go: 26.48%
  before: 21.14%
  computer: 19.89%
  drink:  9.25%
  thin:  7.20%
  no:  5.49%
  candy:  2.24%
  who:  2.05%
  help:  1.78%
  deaf:  1.60%
  fine:  1.44%
  clothes:  0.42%
  book:  0.42%
  cousin:  0.30%
  chair:  0.30%
```
And here is the graph of the training progress. 

![video training 1 Sept 11 2022](https://user-images.githubusercontent.com/102377660/189537495-c44c0d42-4e35-48f3-bb9c-7209e1c41ef3.JPG)

Hmmm not very good. The training and validation accuracy are both pretty low, there was a large difference between the results of the training and validation data, and the validation loss was even increasing when it should be decreasing. Looking at the 3 random test samples, the model didn't guess any of them correctly. Well crap. 

## Wrap up 
So... the model pretty much sucked. I spent some time testing different model configurations and training parameters. In some situations, the model trained better, but the validation accuracy was still very poor and the validation loss always increased. 
Something clearly wasn't working. My next step was to re-organize the dataset into sub folders and take a look at the input data more closely. As they say, garbage in garbage out. I was certainly getting a garbage output, so I needed to check the inputs and if needed, try to make them less trashy. 
