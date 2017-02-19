#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of three covnets with 5x5 filter sizes and depths between 24 and 48, two covnets with 3x3 filter sizes and depths of 64. Lasty, there are three fully connected layers with depths of 100, 50, and 10 (model.py lines 74-86).

The model includes RELU activation functions to introduce nonlinearity (code lines 77-81), and the data is normalized in the model using a Keras lambda layer (code line 75). 

####2. Attempts to reduce overfitting in the model 

Roughly half of the training data was captured by driving around the tracks in the opposite direction. All images were augmented by applying a random percent of brightness change. All images were duplicated and flipped to avoid the model overfitting to one steering direction. The model was trained and validated on different data sets. The model was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

This model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovery maneuvers from both the left and the right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The strategy for deriving a model architecture was to find a model that has proven success, rather than to reinvent the wheel. Many Udacity students have reported success with [Nvidia's CNN designed for self-driving cars](https://arxiv.org/pdf/1604.07316v1.pdf).

The model immediately performed well, but was showing signs of overfitting. I experimented with adding dropout to a variety of combination of layers, with little if any improvement. The car would zig-zag on parts of the track, drive very smoothly on other parts, and drive straight off the track on otherts. This indicated that the model was still overfitting to parts of the data and not generalizing.

To combat the overfitting, I drove both tracks in both directions. This included center driving and recovery maneuvers. When the car showed signs of trouble in certain parts of the track, I would go back and drive that part of the track with center driving and recovery maneuvers, in both directions. However, as soon as the car improved on one part of the track, it would performed worse on other parts. This told me that the model was still not generalizing well enough.

The final step was to apply random brightness levels to all of the images. This provided enough variance in the data to enable the model to generalize, rather than to memorize. I saw an immediate and drastic improvement in the car's ability to navigate the track. It performed well in all areas of the track and recoverered well. It was immediately obvious that the model had learned and was generalizing well.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

* Input (160x320x3)
* Lambda layer (normalize image data to betweeen -0.5 and 0.5)
* Cropping2D (new image shape is 65x260x3)
* Conv Layer (5x5x3 filter, stride 2 to 31x128x24)
* Relu
* Conv Layer (5x5x24 filter, stride 2 to 14x62x36)
* Relu
* Conv Layer (5x5x36 filter, stride 2 to 5x29x48)
* Relu
* Conv Layer (3x3x48 filter, stide 1 to 3x27x64)
* Relu
* Conv Layer (3x3x64 filter, stide 1 to 1x25x64)
* Relu
* Flatten (1536 nodes)
* Fully connected layer (with added dropout) (1600->100)
* Fully connected layer (with added dropout) (100->50)
* Fully connected layer (with added dropout) (50->10)
* Output (10->1)


####3. Creation of the Training Set & Training Process

I recorded several laps driving in the center of the road to teach the vehicle good driving behavior. I did this on both tracks, driving in both directions. Here is an example image of center lane driving:

![alt text](./images/center_driving_example.png "Center Driving")



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from off-center positions. These images show what a recovery looks like starting from the left side :

![alt text](./images/recovery_1.png "Recovery Driving")
![alt text](./images/recovery_2.png "Recovery Driving")
![alt text](./images/recovery_3.png "Recovery Driving")



To augment the dataset, I flipped images and angles thinking that this would normalize the distrubtion of string angles. For example, here is an image that has then been flipped:

![alt text](./images/normal_image.png "Normal")
![alt text](./images/flipped_image.png "Flipped")



Here is the distribution of steering angles after adding offset angles for left and right cameras and adding flipped images/angles:

![alt text](./images/angle_distribution.png "Steering Angles")



Applying brightness changes did wonders in preventing overfitting. Here are examples of the range of the random brightness adjustments applied to images:

![alt text](./images/brightness_1.png "brightness 0.4")
![alt text](./images/brightness_2.png "brightness 0.5")
![alt text](./images/brightness_3.png "brightness 0.6")
![alt text](./images/brightness_4.png "brightness 0.7")
![alt text](./images/brightness_5.png "brightness 0.8")
![alt text](./images/brightness_6.png "brightness 0.9")
![alt text](./images/brightness_7.png "brightness 1.0")
![alt text](./images/brightness_8.png "brightness 1.1")
![alt text](./images/brightness_9.png "brightness 1.2")



After collecting and augmenting the data, I had a total of 906800 images. I then preprocessed the data using a keras lambda layer to normalize the image data and a cropping layer to crop 70 pixels from the top of the image, 25 pixels from the bottom, and 30 pixels from each side. The purpose of cropping is to greatly reduce the amount of irrelavent data, freeing up capacity in the neural network to learn from the most relavant data. Here are examples of the final images fed into the neural network:

![alt text](./images/final_1.png "Final 1")
![alt text](./images/final_2.png "Final 2")
![alt text](./images/final_3.png "Final 3")
![alt text](./images/final_4.png "Final 4")
![alt text](./images/final_5.png "Final 5")
![alt text](./images/final_6.png "Final 6")
![alt text](./images/final_7.png "Final 7")
![alt text](./images/final_8.png "Final 8")
![alt text](./images/final_9.png "Final 9")
![alt text](./images/final_10.png "Final 10")



The dataset was shuffled and split into training and valdiation sets. The validation set consisten of a random 20% of the data. Both sets were shuffled and the neural network was trained with 13824 images per epoch for 15 epochs. Training and validation losses were 0.0329 and 0.0291, respectively. 

![alt text](./images/Screen Shot 2017-02-17 at 1.38.01 PM.png "trained") 

Below is the car driving on both tracks 1 and 2:

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/olzW_mLLwZc/0.jpg)](https://youtu.be/olzW_mLLwZc)



The car now drives very smoothly on both tracks, with no major problem areas on the tracks. I think I could have done it with a lot less images, if only I had realized the power of data augmentation sooner. The two things that had the most visible effect during testing were adding random brightness and cropping more of the image from all sides. This supplied much more variety and much less irrelavent data, making my neural network much more powerful in its learning. If I were to do it again, I would implement a few more methods of image augmentation right away and drive less.

