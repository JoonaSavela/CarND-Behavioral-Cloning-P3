# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create, train and save the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 containing one autonomous lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia's convolutional neural network as a starting point for my model.

The model consists of five convolutional layers, the first three layers with 4x4 filter sizes, 2x2 strides, and depths between 24 and 48 (model.py lines 73-75), and the fourth and the fifth layer with a filter of 2x2 and 1x1 (respectively), 1x1 strides, and depths of 64 (model.py lines 76-77). Flattening is applied to the output of the last convolutional layer, after which the model consists of four fully connected layers with outputs of 100, 50, 10, and 1.

The model includes RELU activation functions in all convolutional and fully connected layers (except the output layer) to introduce nonlinearity (model.py lines 73-77, 80, 82, 84). The data is cropped, normalized, and mean centered in the model using a cropping layer (line 71) and a Keras lambda layer (line 72). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79, 81, 83, and 85). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18 and 63-64). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and some smooth curve driving. The most valuable data in order to get the car to drive successfully seemed to be the center lane driving and recovery driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to construct a good model, gather good data, and then to test what could be done to improve the performance of the model.

My first step was to use a convolution neural network model similar to the Nvidia's neural network for self-driving cars. I thought this model might be appropriate because it's being used to drive real cars on real roads.

I experimented with different techniques in order to improve the performance of the model, such as resizing the image, altering the number of epochs, and changing the dimensions of the network. In the end, I decided not to resize the images, to get as much information as possible in exchange for longer training time. I also decided to use 4 epochs, since any more than that often seemed to overfit the model based on the test driving on the track. Also, the filter sizes of the convolution layers were made smaller because of the resizing process, but they were working fine with the regular sized images so I left them as they were.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. However, this did little to help me predict how the model would perform on the track. Only when the validation loss would oscillate could I have a clue that the car might not perform well, but even that was not very reliable. The overfitting of the model could be seen much better from test driving on the track. An overfit model would drive out of the track always on the same spots, that is, when the road is curving to the left, the car starts to steer to the right, whereas an underfit model would simply not curve enough to the left.

To combat the overfitting, I modified the model by applying dropout layers, and by training the model with 4 epochs (as mentioned earlier). Also, if the model failed on some parts of the road where it earlier had no problems with, and otherwise did well on the track, I would simply retrain the model, since randomness has quite a significant role in training the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track because it did not curve enough, implying underfitting. To improve the driving behavior in these cases, I cautiously gathered more driving data from those parts of the road, especially recovery driving data. It was essential not to gather too much of that data, otherwise the model could have overfit to it and could have not been able to drive well on the rest of the track.

At the end of these processes, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-86) consisted of a convolution neural network with the following layers and layer sizes: 

| Layer  | Description  |
|----------|--------------------|
| Input  | (160, 320, 3) image |
| Cropping2D  | cropping: 70 pixels from the top, 25 pixels from the bottom; output (65, 320, 3)  |
| Lambda | normalization and mean centering, output (65, 320, 3)  |
| Convolution 4x4  | 2x2 stride, RELU activation, output (31, 159, 24) |
| Convolution 4x4  | 2x2 stride, RELU activation, output (14, 78, 36)  |
| Convolution 4x4  | 2x2 stride, RELU activation, output (6, 38, 48)  |
| Convolution 2x2  | 1x1 stride, RELU activation, output (5, 37, 64)  |
| Convolution 1x1  | 1x1 stride, RELU activation, output (5, 37, 64)  |
| Flatten  |   |
| Dropout  | drop rate 0.5  |
| Fully connected  | output 100  |
| Dropout  | drop rate 0.5  |
| Fully connected  | output 50  |
| Dropout  | drop rate 0.5  |
| Fully connected  | output 10  |
| Dropout  | drop rate 0.5  |
| Fully connected (output)  | output 1  |

Total number of (trainable) parameters in the model is 1,248,915.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
