import os
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split


DATASET_PATH = "../../American/"


num_classes = len(os.listdir(DATASET_PATH))
print(f"There are {num_classes} classes")
classes = os.listdir(DATASET_PATH)
print(classes)

# Create pandas dataframes containing the location and class of each file
image_paths = []
image_labels = []
NUM_IMAGES_PER_CLASS = 1000  # The number of images to use from each class

# Check each folder in the dataset. Each one corresponds to a class (category)
for category in os.listdir(DATASET_PATH):
    # for each category folder create folder path
    category_path = os.path.join(DATASET_PATH, category)

    # randomly select (without replacement) a subset of the images in the current class folder
    sample_images = sample(os.listdir(category_path), NUM_IMAGES_PER_CLASS)

    # for each image in the category subset
    for image in sample_images:
        # Add the image name to the path and append the image path and label to their respective lists
        image_paths.append(os.path.join(category_path, image))
        image_labels.append(category)


# combine the lists into a pandas dataframe. First convert to pandas Series and then concatenate.
df = pd.concat(
    [
        pd.Series(image_paths, name="image_paths"),
        pd.Series(image_labels, name="image_labels"),
    ],
    axis=1,
)


# count the times each label (category) appears
# print(df["image_labels"].value_counts())
##----------------


# use 80% of the data for training. Within that training set, use 20% for validation and 20% for testing.
train_split = 0.8
val_split = 0.25

# use scikit learn's train test split function to generate testing data
intermediate_df, test_df = train_test_split(
    df, train_size=train_split, shuffle=True, random_state=123
)
# from the remaining data, generate the training and validation sets
train_df, valid_df = train_test_split(
    intermediate_df, train_size=1 - val_split, shuffle=True, random_state=123
)

# Print the number of samples in each set to confirm the intended proportions
print(
    f"Training samples: {len(train_df)}, Test samples: {len(test_df)}, Validation samples: {len(valid_df)}"
)


HEIGHT = 50  # height of each image in pixels
WIDTH = 50  # width of each image in pixels
CHANNELS = 1  # number of channels per pixel, 3 for rgb (because pictures are in colour)
BATCH_SIZE = 80  # number of images per batch


# create the data generators
gen_train = ImageDataGenerator()
gen_test = ImageDataGenerator()
gen_val = ImageDataGenerator()

# Configure the generators
train_generator = gen_train.flow_from_dataframe(
    train_df,
    x_col="image_paths",
    y_col="image_labels",
    target_size=(HEIGHT, WIDTH),
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True,
    batch_size=BATCH_SIZE,
)
test_generator = gen_test.flow_from_dataframe(
    test_df,
    x_col="image_paths",
    y_col="image_labels",
    target_size=(HEIGHT, WIDTH),
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False,
    batch_size=BATCH_SIZE,
)
validation_generator = gen_val.flow_from_dataframe(
    valid_df,
    x_col="image_paths",
    y_col="image_labels",
    target_size=(HEIGHT, WIDTH),
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True,
    batch_size=BATCH_SIZE,
)


##----------------

# Initialize and define the CNN
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

opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
# print the model architecture summary
model.summary()


epochs = 25  # number of times the model will go through the training data
steps_per_epoch = (
    train_generator.n // train_generator.batch_size
)  # number of batches per epoch
validation_steps = validation_generator.n // validation_generator.batch_size

## define some keras callbacks
# define a learning rate schedule to reduce learning rate by 10 for every 2 epochs without an improvement in validation loss
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto"
)
# saves the model weights after every epoch
checkpoint = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_accuracy",
    save_weights_only=True,
    mode="max",
    verbose=1,
)
# stops the training early if the validation loss has not improved (decreased) in the last 3 epochs
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)


history = model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[checkpoint, reduce_lr, early_stop],
)


# Plot the training
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
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()


# Save model
model_json = model.to_json()
model.save_weights("model_weights_alphabet.h5")
with open("model_alphabet.json", "w") as json_file:
    json_file.write(model_json)
