# **Behavioral Cloning** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### How to run the project:
0. Download and run the [simulator-linux](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip) and choose the TRAIN MODE, save the traing data. The simulator is very great!
1. ```sh  
    source activate carnd-term1
2.  ```sh  
    python model.py
3. ```sh 
    python drive.py model.h5
4. run the simulator-linux and choose the AUTONOMOUS MODE
5. Enjoy it or ...  destory it ?  Maybe enjoy.

### Something important from classroom 
1. A rule of thumb to choose training data and validation data quantity
A rule of thumb could be to use 80% of your data for training and 20% for validation or 70% and 30%. Be sure to randomly shuffle the data before splitting into training and validation sets.
2. underfitting
If model predictions are poor on both the training and validation set (for example, mean squared error is high on both), then this is evidence of underfitting. Possible solutions could be to:
* a.increase the number of epochs
* b.add more convolutions to the network.
3. overfitting
When the model predicts well on the training set but poorly on the validation set (for example, low mean squared error for training set, high mean squared error for validation set), this is evidence of overfitting. If the model is overfitting, a few ideas could be to:
* use dropout or pooling layers
* use fewer convolution or fewer fully connected layers
* collect more data or further augment the data set

#### Data Preprocessing
* normalizing the data(model.py line 163)
* mean centering the data(model.py line 163)
#### Data augmentation 
* Flipping Images And Steering Measurements (model.py line 116)
   because the model is learning to steer to the left, we have to flipping images.
benefits: 1. we have more date to use for trainging the network; 2. the date is more comprehensive


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My final model come from nvida's [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), you can get in in 'doc/', too.
The model architecture:
![alt text][image1]

#### 2. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 189). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 201).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

### The steps I finish it:
1. First, I use the nvida’s network and only use the middle camera's images. But the car only turn left.
2. Fliping the middle camera's images and train again. Great.
3. And then, I revise the images size which include road or only trees, it's greater.
4. Finally I find a little overfitting, I decrease the numbers of epochs and use more images.
All the data: 48216
Length of X_train:  38572
Length of X_valid:  9644
The final loss is: loss: 0.0152 - val_loss: 0.0132
![alt text][image2]
And the most important is that it not overfitting.And the car can successfully drives around track one without leaving the road
5. I modify the speed to 22 ( drive.py line 47), it can successfully drives around track one without leaving the road, too.


[//]: # (Image References)
[image1]: ./pics/Nvidanetwork.png "network"
[image2]: ./pics/loss.png "finalLoss"
