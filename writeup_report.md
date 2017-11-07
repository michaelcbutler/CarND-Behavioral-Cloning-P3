# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (unmodified)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording the successful navigation of track one

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I considered a sequence of models beginning with a simple regression network consisting of only a flatten layer followed by a fully-connected layer:
```py
model.add(Flatten())
model.add(Dense(1))
```
The second model architecture is based on LeNet. It consists of two 5x5 convolution layers using ReLu activation with each followed by a max pooling layer, then a flatten layer followed by three fully-connected layers.
```py
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
```
The final model is based on Nvidia's PilotNet architecture as detailed in the paper "Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car" by Bojarski et al. This architecture consists of five convolution layers using ReLu activation, followed by a flatten layer and four fully-connected layers.
```py
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

The baseline model uses linear regression, but the LeNet and PilotNet models use ReLu activation to introduce nonlinearity.

All three models use two pre-processing steps to normalize the data and crop the visual noise at the top and bottom of the input images.
```py
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

All three models used mean-squared error for the loss function and the adam optimizer. All three models also randomly shuffle the input data before splitting off 20 per cent for validation.

#### 2. Attempts to reduce overfitting in the model

The data set was agumented to reduce overfitting. Using the images from the offset (right and left) cameras with adjusted steering inputs and flipping all three images (center, right, left) with the steering angle negated added greater data diversity for the training process. The data augmentation occurs in the generator function at lines 50-88.

Observing the training loss versus validation loss trend suggested limiting the number of EPOCHS to reduce overfitting. The final version used five EPOCHS.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

All models used the adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Using the simulator, I generated a data set of center lane driving for two laps of track one. However, I found the sample data set provided to be adequate for my training needs once augmented. The augmentation process is detailed in Section 2 above.

#### 5. Use of generator function

For my first attempt, I was able to train the PilotNet model with the fully augmented data set, but just barely. I had to close all other programs to have enough memory available. The subsequent use of the generator function removed this memory constraint.