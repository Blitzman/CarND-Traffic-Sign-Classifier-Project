# Traffic Sign Recognition

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
[preprocessing]: ./images/preprocessing.png "preprocessing"
[learningcurves]: ./images/learningcurves.png "learningcurves"
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

The code for this step is contained in the third code cell of the IPython notebook. We performed an exploratory visualization of each split of the dataset, randomly selecting four samples of each one of them together with their classes. We can observe that the dataset is not a particularly easy one since there are various conditions such as illumination, occlusion, and slight rotations that might be difficult to learn and classify properly. However, it is worth noticing that the three splits look quite similar so the training set looks representative enough of what we will find in the test and validation ones. We also explored the dataset split per-class distribution, but it will be discussed later in this report.

#### Training Images Visualization
![alt text][explorationtrain]

#### Testing Images Visualization
![alt text][explorationtest]

#### Validation Images Visualization
![alt text][explorationvalid]

### Design and Test a Model Architecture

#### Data Preprocessing

The code for this step is contained in the fifth code cell of the notebook. This cell contains the following functions that were implemented for consideration in the preprocessing pipeline:

* Feature Normalization
* Feature Standardization
* Feature Rescaling
* RGB to Grayscale Conversion
* Histogram Equalization

After various experiments we found out that the combination of histogram equalization and feature rescaling to the range -0.5-0.5 offered the best performance. Feature normalization and standardization did not seem to work well on this problem in particular and RGB to grayscale conversion, despite the fact that it simplifies the problem for smaller networks, throws out important color information that helps getting rid of certain ambiguities. The following figure shows the results of our preprocessing pipeline on a training image.

![Preprocessing Steps][preprocessing]

#### Data Splits

We used the data splits provided by the dataset itself. Those splits were analyzed to determine the per-class occurrences. As we previously stated, the training set is composed by 34799 examples, the validation split contains 4410 images, and the testing one contains 12630 traffic signs. According to the visualization, we can observe that the training set is significantly imbalanced. The validation and testing splits are also imbalanced for some classes but the difference is not that big. Nevertheless, it appears to be a good balance between the splits so that no class is quite populated in one split but too sparse or not that present in others.

![Dataset Splits Per-class Distribution][datasetperclass]

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

#### Model Training

The code for training the model is located in cells 7, 8, and 9 in the notebook.

Cell #7 defines the placeholders for the needed data (images, labels, and dropout keep probability). It also defines a one-hot encoding vector for the labels, as well as the hyperparameters: epochs (32), batch size (128), learning rate (1e-3), and regularization parameter (1e-6). Those hyperparameters were chosen empirically after a thorough experimentation with various setting. In addition, this cell defines the logits and weights using the aforementioned network architecture, the cross entropy function, the loss operation (which includes L2 regularization by adding a cost with a self-defined function in the very same cell), the optimizer (we used ADAM), and at last the training operation.

Cell #8 defines a evaluation function to compute both the loss and the accuracy given a set of images and its labels. That function makes use of a previously defined accuracy operation which is just a mean reduction of the correct predictions, that were in turn computed using a comparison between the argmax of the logits and the argmax of the one-hot encoding for the labels.

Cell #9 defines the training process itself. The training session will go over the whole dataset using minibatches for a certain number of epochs. For each epoch both training and validation losses and accuracies are computed and stored (for further analysis). We also keep a running best validation accuracy so that we save the model weights each time we hit a new best performer.

#### Finding a Solution

After training the model with the code we explained above, we plotted a couple of learning curves to analyze the evolution of both training and validation accuracy and loss. The function we used to plot those curves is located in the ninth cell of the notebook.

![Learning Curves][learningcurves]

As we can observe, we achieved a good learning behavior with an appropriate learning rate (loss decreases slowly but steadily). Training accuracy is almost stable after epoch 15 while validation accuracy still increases slowly until epoch 30. At that point, training accuracy is already 1.000 and we get a maximum validation accuracy of 0.967. That was the best set of weights that we achieved for our network with this training procedure and the previously stated hyperparameter setup. We saved that model and evaluated the accuracy on the test set, achieving a significantly good 0.945. That fact proves that our model is not overfitting that much to the training data and it is capable of generalizing to new examples.

It is important to remark that this architecture/procedure/model was the final outcome of an iterative approach which we can describe by answering to the following questions:

* What was the first architecture that was tried and why was it chosen?

At first, we tried a plain LeNet-5 architecture, given its many success stories and the fact that we already implemented it in the nanodegree lessons. This network was straightforward to define and easy to setup to get a first version of the pipeline up and running.

* What were some problems with the initial architecture?

The main problem was that the network's accuracy on the validation set was getting stuck quite early and really low (less than 0.9). Inspecting the evolution of training and validation accuracies it was an obvious overfitting problem. On the other hand, we also observed that the loss decreased extremely fast during the first epochs and then it stabilized.
 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

We carried out some architectural changes to address overfitting. The main change consisted of introducing dropout layers after the already existing fully connected ones. We used a standard keep probability of 0.5. In addition, to avoid underfitting due to that change, we also increased the number of neurons in first two fully connected layers to 1024. 

* Which parameters were tuned? How were they adjusted and why?

We decreased the learning rate from 1e-2 to 1e-3 to get a steadier gradient descent. This led to better convergence in general.

We also increased the original batch size from 32 to 128. We ran different experiments with varying batch sizes (32, 64, 128, and 256) and we found a slight increase in accuracy as we increased the batch size. However, in order not to make training too slow we decided to keep an intermediate size of 128.

We also added L2 regularization with a regularization parameter of 1e-6. We found no significant difference with other close values (1e-5 and 1e-7).

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
