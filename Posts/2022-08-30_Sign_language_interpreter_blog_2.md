# Developing a sign language interpreter using machine learning Part 2 - Building a basic CNN model

In this post I will go through my first attempts at building a convolutional nerural network for the classification of still images of the alphabet in American sign language. 
Full disclosure, I referenced this link (https://www.kaggle.com/code/gpiosenka/asl-f1-score-98) and utilized some of the same methodology.
That said, let's jump in. 

## Assessing the dataset
The whole dataset has somewhere around 140 thousand images. For these first tests, I will use a subset of the dataset to speed up training and thus, preserve my sanity a while longer. 
But I am getting ahead of myself... First, it is good to know how many classes (categories) are in the dataset and how many samples (images) are in each. 

To start I provided the path to the dataset folder. The folder is called American because I am using a dataset of American sign language. 
Then, I used a for loop to print the name of each class folder and the number files it contains.
```
DATASET_PATH = "../../American/"

# Print the number of images in each catagory in the dataset
for label in os.listdir(DATASET_PATH):
    print(str(len(os.listdir(DATASET_PATH + label))) + " " + label + " images")
    
num_classes = len(os.listdir(DATASET_PATH))
print(f"There are {num_classes} classes")
```
After running the code, here is the output.
```
3070 0 images
1570 1 images
1570 2 images
1570 3 images
3070 4 images
3070 5 images
3070 6 images
3070 7 images
3070 8 images
3070 9 images
6070 a images
6070 b images
6070 c images
6070 d images
3070 e images
6070 f images
6070 g images
6070 h images
6807 i images
6124 j images
5488 k images
6494 l images
2924 m images
3968 n images
7294 o images
2566 p images
3590 q images
3538 r images
2374 s images
3076 t images
3244 u images
3926 v images
2086 w images
2330 x images
2454 y images
2218 z images
There are 36 classes
```
As you can see there are a lot of classes (categories); 36 to be exact. 26 for the alphabet and 10 for numbers 0-9.
Each class has a few thousand images, but the number of images is not equal between classes. This can lead to issues. 
Giving a model more samples of a certain class can cause bias since the model can 'have more practice' on some classes. This issue is known as class imbalance.
More on that in a minute. 

### Load the image paths and labels

To be able to manipulate the images and to easily create sub-samples, the image paths and labels were loaded into a pandas dataframe. 
During this process, 100 images from each class are kept. This value will be increased in the future, but to keep things fast, the classes are kept small.
Here is the code. 
```
# Create pandas dataframes containing the location and class of each file
image_paths = []
image_labels = []
NUM_IMAGES_PER_CLASS = 100  # The number of images to use from each class

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
print(df["image_labels"].value_counts())

```
This code first declares 2 empty lists that will be used later, and also declares (as a constant) the number of samples to keep from each class (100 for now). 
Then, the code gets the list of folders in the dataset's main folder and loop through them. For each class, the name of the class folder is joined with the dataset path.
Next, the code gets the list of images in the class folder and randomly selects a subset. 
Looping through this subset, the code then creates the path for each image and appends the image path and image label to the image_paths, and image_labels lists.
When this code was run, here was the output.
```
0    100
1    100
k    100
l    100
m    100
n    100
o    100
p    100
q    100
r    100
s    100
t    100
u    100
v    100
w    100
x    100
y    100
j    100
i    100
h    100
8    100
2    100
3    100
4    100
5    100
6    100
7    100
9    100
g    100
a    100
b    100
c    100
d    100
e    100
f    100
z    100
Name: image_labels, dtype: int64
100
```
Great. It looks like each class has exactly 100 images. 
Since we declared the number of samples to keep as a constant, we can easily come back and change this value later to select a larger subset of data. 

## Train / test split
When building a machine learning model, it is good practice to split the available data into training and testing datasets. 
The training data is used to train the model, and the trained model is evaluated using the test data. 
These datasets should be kept separate so that when the model is tested it is being evaluated using samples it has never seen before. 
Importantly, the test data should only be used once, at the very end of the model development process. The whole point of using a separate test set is so to have a representative dataset that your model has never seen before. 
But if you repeatedly train a model, test it on the test set then tweak the model to improve the performance on the test set, well then the test set ceases to be separate and is being used to influence the training of the model. 
This can lead to over fitting since you are tuning the model so that it performs well on the test data. 

Now, you may be asking yourself, how can I tune my model if I can't test it? Well, the answer to that is by using a validation dataset. The validation is basically the 'test' data that is used during model development. 
As the model is trained and adjusted the model performance is monitored using the validation data.

Although the exact proportions can vary, common ratios are 60% training, 20% testing, and 20% validation. These ratios are equivalent to removing 30% of the data for testing, then spliting the remaining data using a 75/25 ratio (as in the code below).
To split the data, the scikit-learn's train_test_split() function is used. 
Here is the code.
```
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

```
The code first defines the ratios to use when splitting the data. Then the data is split into a test set and an intermediate dataset that is then split into the training and validation sets. 
Here is the result of running this section of code.

```
Training samples: 2160, Test samples: 720, Validation samples: 720
```
Alright. So the data has been split into the different sets we will need. 
Well actually, the 'data' has not been split. Rather, dataframe holding the location of the images has been split into different sections. 
To train the model we will need to load the actual images for the model to use, that is where image generators come in. 

## Image Generators
Tensorflow Image Generators are an important part of this project. The image generators load and prepare images that are then passed to the model during training. 
A variety of pre-processing steps can be added to generators making them simple and powerful tools for image classification models. 

First, I declare the image size. The size is defined by the image width and height in pixels and the number of channels per pixel. The images in the dataset are colour, meaning they have 3 channels (rgb). 
However, in this case, the colour of the image doesn't matter. The model will be used to examine the position of the fingers and hands, the colours of the hand is irrelevant. For this reason the images are converted to grayscale in the ImageGenerators. This reduces the number of model parameters which can help speed up training and reduce overfitting. 

The number of images per batch is also defined. During training the model will go through all of the data in the training set multiple times. An epoch refers to the model seeing all of the training data once. 
So if there are 2160 images in our training set, the model will see all 2160 images each epoch. Now, loading that many images into memory at once may not be possible, so batches of images can be used. 
Batching presents the images to the model in smaller groups. After seeing each batch the model weights are updated before starting the next batch. For this reason, the batch size can have an effect on model generalization. 
The ideal batch size is a complicated topic. The size of the batch can affect convergence speed and model generalization. For now, I set the batch size to 80. Keep in mind this is a tunable parameter that can be changed and experimented with during model development. 

```
HEIGHT = 50  # height of each image in pixels
WIDTH = 50  # width of each image in pixels
CHANNELS = 1  # number of channels per pixel. Normally 3 for rgb, but 1 here because the images are converted to grayscale
BATCH_SIZE = 80  # number of images per batch
```
With the image size and batch size constants defined, the imagegenerators were created. After being initantiated, the specific parameters of the generators were set. Check out the following code.
```
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
```
As you can see, several parameters are set for each image generator. First, the dataframe is passed along with the column names corresponding to the x and y data. By convention, the x data is the data being classified (images in this case) and the y data is the category label.
Next, the image size is set. The classification mode is set to 'categorical', the color mode to rgb (because the images are in colour), the shuffle parameter dictates whether the order of the samples is mixed. Finally, the batch size defined earlier is passed in as well. 
After creating the image generators, now it is time to define the neural network architecture. 

## CNN architecture 
To start the CNN model is pretty basic, it has 2 convolutional layers followed by 2 fully connected layers and a softmax output layer. For all layers, a the Relu activation function is used. Batch normalization and dropout layers are also included to help prevent overfitting.
Each convolutional layer uses a Number of filters, filter size of 3 by 3 pixels and padds the output to match the size of the input images. The pooling layers use a 2 by 2 and take the maximum value. 
After the convolutional layers, there are 2 fully connected layers. Finally there is a fully connected layer with 35 nodes (matching the number of classes) and a softmax activation function.
Here is the model architecture.
```
# Initialize and define the CNN
model = Sequential()

# 1 - Convolution layer
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

# Flattening
model.add(Flatten())

# 1st Fully connected layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

# 2nd Fully connected layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))

# Final dense layer with softmax activation
model.add(Dense(num_classes, activation="softmax"))
```
Next, the a few more model parameters are set and the model is compiled. 
```
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# print the model architecture summary
model.summary()
```
The Adam optimizer is used with an initial learning rate of 0.0005, categorical cross entropy is used as the loss function and the metric used for model evaluation is accuracy.

## Define some callbacks
With the model created and compiled, I next defined callbacks to be used during training process. 
### Learning rate schedule
The learning rate affects how aggresively the model parameters are adjusted during back propagation. Larger learning rates can help the model converge quickly but may not produce good performance since the convergence isn't necessarily optimal. 
Smaller learning rates converge more slowly but can make smaller adjustment to really fine tune the convergence. With this in mind, the learning rate will get progressively smaller as the model trains. The idea here is that the larger learning rate at the beginning will get us in the ball park, and from there the smaller learning rate will zero in on the optimal solution.
To do this, the Keras Reduce on Plateau callback is used which will reduce the learning rate when the validation loss stops decreasing. Here is the code.
```
# define a learning rate schedule to reduce learning rate by 10 for every 2 epochs without an improvement in validation loss
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto")
```

### Periodically save model weights
To keep track of the model progress (and to still have a salvageable model if something goes wrong), the model weights are saved after every epoch using the Model Checkpoint callback.
```
# saves the model weights after every epoch
checkpoint = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_accuracy",
    save_weights_only=True,
    mode="max",
    verbose=1,
)

```
### Early stopping
When training a model, the number of epochs to train is set ahead of time. This means, the number of times the model runs through the dataset needs to be chosen in advance. 
This can be somewhat of a guess since it is hard to know how many epochs a model will need to train before the performance plateaus. To solve this issue, early stopping criteria can be set. 
The criteria will be checked after each epoch and will stop training the model if certain conditions are reached. Here is the early stopping callback.
```
# stops the training early if the validation loss has not improved (decreased) in the last 3 epochs
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
```
The callback monitors the validation loss, and if it has not decreased in last 3 epochs, then the training will stop. 

## Model training
Close to actually running the model now. 
You know, since they say that writting is mostly editing, and coding is mostly debugging. I think it could be said that training machine learning models is mostly setup. Just saying. 
Anyway, on to the training step. To train the model,  model.fit() is used. The training data image generator is passed in along with the preset number of epochs, the validation data generator and the callback functions. 
The number of steps per epoch and the number of validation steps are calculated based on the total number of samples and the batch size.
```
epochs = 10  # number of times the model will go through the training data
history = model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[checkpoint, reduce_lr, early_stop],
)

```
When the code is run, here is the output.
```

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 50, 50, 64)        640

 batch_normalization (BatchN  (None, 50, 50, 64)       256
 ormalization)

 activation (Activation)     (None, 50, 50, 64)        0

 max_pooling2d (MaxPooling2D  (None, 25, 25, 64)       0
 )

 dropout (Dropout)           (None, 25, 25, 64)        0

 conv2d_1 (Conv2D)           (None, 25, 25, 128)       73856

 batch_normalization_1 (Batc  (None, 25, 25, 128)      512
 hNormalization)

 activation_1 (Activation)   (None, 25, 25, 128)       0

 max_pooling2d_1 (MaxPooling  (None, 12, 12, 128)      0
 2D)

 dropout_1 (Dropout)         (None, 12, 12, 128)       0

 flatten (Flatten)           (None, 18432)             0

 dense (Dense)               (None, 256)               4718848

 batch_normalization_2 (Batc  (None, 256)              1024
 hNormalization)

 activation_2 (Activation)   (None, 256)               0

 dropout_2 (Dropout)         (None, 256)               0

 dense_1 (Dense)             (None, 256)               65792

 batch_normalization_3 (Batc  (None, 256)              1024
 hNormalization)

 activation_3 (Activation)   (None, 256)               0

 dropout_3 (Dropout)         (None, 256)               0

 dense_2 (Dense)             (None, 36)                9252

=================================================================
Total params: 4,871,204
Trainable params: 4,869,796
Non-trainable params: 1,408
_________________________________________________________________
Epoch 1/10
2022-09-01 18:56:58.736576: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8401
27/27 [==============================] - ETA: 0s - loss: 2.7904 - accuracy: 0.2718  
Epoch 1: saving model to model_weights.h5
27/27 [==============================] - 42s 1s/step - loss: 2.7904 - accuracy: 0.2718 - val_loss: 6.8455 - val_accuracy: 0.0500 - lr: 5.0000e-04
Epoch 2/10
27/27 [==============================] - ETA: 0s - loss: 1.7040 - accuracy: 0.5486
Epoch 2: saving model to model_weights.h5
27/27 [==============================] - 2s 79ms/step - loss: 1.7040 - accuracy: 0.5486 - val_loss: 2.8581 - val_accuracy: 0.1694 - lr: 5.0000e-04
Epoch 3/10
27/27 [==============================] - ETA: 0s - loss: 1.3192 - accuracy: 0.6556
Epoch 3: saving model to model_weights.h5
27/27 [==============================] - 2s 82ms/step - loss: 1.3192 - accuracy: 0.6556 - val_loss: 1.7346 - val_accuracy: 0.5208 - lr: 5.0000e-04
Epoch 4/10
27/27 [==============================] - ETA: 0s - loss: 1.0363 - accuracy: 0.7431
Epoch 4: saving model to model_weights.h5
27/27 [==============================] - 2s 79ms/step - loss: 1.0363 - accuracy: 0.7431 - val_loss: 1.3153 - val_accuracy: 0.6653 - lr: 5.0000e-04
Epoch 5/10
27/27 [==============================] - ETA: 0s - loss: 0.8677 - accuracy: 0.7792
Epoch 5: saving model to model_weights.h5
27/27 [==============================] - 2s 78ms/step - loss: 0.8677 - accuracy: 0.7792 - val_loss: 1.1249 - val_accuracy: 0.7250 - lr: 5.0000e-04
Epoch 6/10
27/27 [==============================] - ETA: 0s - loss: 0.7068 - accuracy: 0.8440
Epoch 6: saving model to model_weights.h5
27/27 [==============================] - 2s 81ms/step - loss: 0.7068 - accuracy: 0.8440 - val_loss: 0.9683 - val_accuracy: 0.7736 - lr: 5.0000e-04
Epoch 7/10
26/27 [===========================>..] - ETA: 0s - loss: 0.5857 - accuracy: 0.8591
Epoch 7: saving model to model_weights.h5
27/27 [==============================] - 2s 82ms/step - loss: 0.5848 - accuracy: 0.8579 - val_loss: 0.9038 - val_accuracy: 0.7694 - lr: 5.0000e-04
Epoch 8/10
27/27 [==============================] - ETA: 0s - loss: 0.4732 - accuracy: 0.8921
Epoch 8: saving model to model_weights.h5
27/27 [==============================] - 2s 77ms/step - loss: 0.4732 - accuracy: 0.8921 - val_loss: 0.7994 - val_accuracy: 0.7903 - lr: 5.0000e-04
Epoch 9/10
27/27 [==============================] - ETA: 0s - loss: 0.3932 - accuracy: 0.9102
Epoch 9: saving model to model_weights.h5
27/27 [==============================] - 2s 79ms/step - loss: 0.3932 - accuracy: 0.9102 - val_loss: 0.7403 - val_accuracy: 0.8069 - lr: 5.0000e-04
Epoch 10/10
26/27 [===========================>..] - ETA: 0s - loss: 0.3465 - accuracy: 0.9240
Epoch 10: saving model to model_weights.h5
27/27 [==============================] - 2s 85ms/step - loss: 0.3479 - accuracy: 0.9231 - val_loss: 0.6690 - val_accuracy: 0.8347 - lr: 5.0000e-04
```
You'll notice the model summary lists each layer in the model and the number of trainable parameters. Our relatively simple model has nearly 5 million parameters. With that in mind it is easy to see why training model can sometimes take hours, or even days.
AFter the model summary the model was trained over 10 epochs. We set the maximum to be 10, which means the model didn't stop training early. This could indicate that the model might have improved if we let it run longer - something to keep in mind for later.
You may notice that the first epoch took 42 seconds to train whereas the rest took about 2 s each. The first layer takes longer because some things need to be set up on the backend.
You might also notice that our accuracy is 92%. Not too bad for a first try! 


