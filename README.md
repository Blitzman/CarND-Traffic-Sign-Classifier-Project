#**Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[datasetperclass]: ./images/datasetperclass.png "datasetperclass"
[explorationtrain]: ./images/train1.png "explorationtrain"
[explorationtest]: ./images/test1.png "explorationtest"
[explorationvalid]: ./images/valid1.png "explorationvalid"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Dataset Summary

The code for this step is contained in the third code cell of the IPython notebook. I used *shape*, *len*, and *set* functions to calculate summary statistics of the traffic signs dataset. A CSV reader with a dictionary was used to read and store class IDs and sign names. The results are as follows:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes in the dataset is 43 (for all splits: train, valid, and test) and they are the following ones:

| Class ID							| Sign Name							|
|:---------------------:|:---------------------------------------------:| 
| 0						| Speed limit (20km/h)							|
| 1						| Speed limit (30km/h)							|
| 2						| Speed limit (50km/h)							|
| 3						| Speed limit (60km/h)							|
| 4						| Speed limit (70km/h)							|
| 5						| Speed limit (80km/h)							|
| 6						| End of speed limit (80km/h)					|
| 7						| Speed limit (100km/h)							|
| 8						| Speed limit (120km/h)							|
| 9						| No passing									|
| 10					| No passing for vehicles over 3.5 metric tons |
| 11					| Right-of-way at the next intersection |
| 12					| Priority road |
| 13					| Yield |
| 14					| Stop |
| 15					| No vehicles |
| 16					| Vehicles over 3.5 metric tons prohibited |
| 17					| No entry |
| 18					| General caution |
| 19					| Dangerous curve to the left |
| 20					| Dangerous curve to the right |
| 21					| Double curve |
| 22					| Bumpy road |
| 23					| Slippery road |
| 24					| Road narrows on the right |
| 25					| Road work |
| 26					| Traffic signals |
| 27					| Pedestrians |
| 28					| Children crossing |
| 29					| Bicycles crossing |
| 30					| Beware of ice/snow |
| 31					| Wild animals crossing |
| 32					| End of all speed and passing limits |
| 33					| Turn right ahead |
| 34					| Turn left ahead |
| 35					| Ahead only |
| 36					| Go straight or right |
| 37					| Go straight or left |
| 38					| Keep right |
| 39					| Keep left |
| 40					| Roundabout mandatory |
| 41					| End of no passing |
| 42					| End of no passing by vehicles over 3.5 metric tons |

### Dataset Exploratory Visualization

The code for this step is contained in the third code cell of the IPython notebook. We performed an exploratory visualization of each split of the dataset, randomly selecting four samples of each one of them together with their classes. We can observe that the dataset is not a particularly easy one since there are various conditions such as illumination, occlusion, and slight rotations that might be difficult to learn and classify properly. However, it is worth noticing that the three splits look quite similar so the training set looks representative enough of what we will find in the test and validation ones.

#### Training Images Visualization
![alt text][explorationtrain]

#### Testing Images Visualization
![alt text][explorationtest]

#### Validation Images Visualization
![alt text][explorationvalid]

#### Split per-class Summary
![alt text][datasetperclass]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### Model Architecture

The code for my final model is located in the sixth cell of the notebook. The final model is a modified version of LeNet-5 which consists of the following layers:

| Layer									|     Description																| 
|:---------------------:|:---------------------------------------------:| 
| Input									| 32x32x3 RGB image															| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6		|
| ReLU									| -																							|
| Max pooling						| 2x2 stride,  outputs 14x14x6									|
| Convolution 5x5				| 1x1 stride, VALID padding, outputs 10x10x16		|
| Flatten								| 5x5x16 input, 400 output											|
| Fully connected				| 1024 neurons																	|
| ReLU									| -																							|
| Dropout								| -																							|
| Fully connected				| 1024 neurons																	|
| ReLU									| -																							|
| Dropout								| -																							|
| Fully connected				| 43 neurons output (classes)										|
| Softmax								| -										        									|

All weights were initialized using a truncated normal distribution with mean 0 and standard deviation 0.001. 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
