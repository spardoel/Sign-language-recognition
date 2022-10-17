import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
import pickle

import random

from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Load the pickle file with the datasets
with open("preprocessed_videos_augmented10_hand_coordinates.pkl", "rb") as f:
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
print(class_vocab)

# set the constants
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 126

# Creation of the sequence model.
def get_sequence_model():
    # define input shapes using the number of features and max video length constants
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # create the initial RNN layer
    x = keras.layers.GRU(128, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    # add a second RNN layer
    x = keras.layers.GRU(256)(x)
    # add several dense layers with dropout
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.55)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    # define the output layer
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    # Create the model
    rnn_model = keras.Model([frame_features_input, mask_input], output)

    # Complie the model
    opt = Adam(learning_rate=0.0003)
    rnn_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=opt,
        metrics=["accuracy"],
    )
    # return the compiled model
    return rnn_model


BATCH_SIZE = 16
EPOCHS = 200

# The main function to train the model
def run_experiment():
    # define some keras callbacks

    # define a learning rate schedule to reduce learning rate by 10 for every 2 epochs without an improvement in validation loss
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, mode="auto"
    )
    # define the early stopping callback
    early_stopping_cb = EarlyStopping(monitor="val_accuracy", patience=20)

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

    # Run the model and get the probability that the video belongs to each class
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    # display the probabilities
    print("Model prediction probabilities:")
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")


# Train the model
history, sequence_model = run_experiment()

check_random_test_sample()
check_random_test_sample()
check_random_test_sample()


weights_output_file = "model_weights_aug10_hand_features.h5"
json_model_file = "model_aug10_hand_features.json"

# Save model
model_json = sequence_model.to_json()
sequence_model.save_weights(weights_output_file)
with open(json_model_file, "w") as json_file:
    json_file.write(model_json)

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
