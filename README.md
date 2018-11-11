# Road detection on video streams
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Term 3, Project 2: Semantic Segmentation
### Keywords: fully convolutional network, VGG16, FCN-8

[//]: # (Image References)
[result]: ./runs/results.gif "Result" 

---

### Overview
In this projects, the pixels of images are labeled in two classes (road, not road) using a Fully Convolutional Network (FCN).

### Dataset
This project uses the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) (downloadable from [here](http://www.cvlibs.net/download.php?file=data_road.zip)).
The data set contains 289 training and 290 test images. For the training images, groun truth labels are available. The images are categorized in three groups: urban unmarked (uu), urban marked, urban multiple marked lanes (umm).

### Model
The model is based on [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). 
For this end, a pretrained VGG16 model is extended. VGG16 last layer gives one value: the image class. To go to pixelwise classification,
the last fully-connected layer of VGG16 is replaced by a 1x1 convolution and is then upsampled to the original resolution (transposed convolution).
Additionally, two skip layers are included. These add pixelwise the information from previous pooling layers, which contain more information about the general picture e.g. a big scene rather than a little puzzle piece from the convolutional network. They can therefore help to reconstruct overall shape.
The weights of each additional layer were initialized with a truncated normal distribution (global parameter STD) and are regularized in order to use all weights (global parameter REG). The regularization will particularly help to get large connected pataches of road detected rather than single pixel.

#### Remarks from Udacity
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.

### Training
The model was trained testing different sets of hyperparameters (epochs, batch size, dropout, learning rate, weight regularization and initialization).
The final chosen parameters were

| Parameter     | Value     | Comment  |
| ------------- |:-------------:| :-----:|
| Epochs        | 30            | the loss did not change anymore (NN has converged) |
| Batch size    | 3             | Small batch sizes generalize better but cause longer training time. Here due to the small amount of training images, a number between 2-4 still results in resonable training times under 1 hour.|
| Learning rate | 1e-4          |   |
| Dropout       | 0.7           |  |
| Regularization | 1e-3        | comparable small due to small batch size, no jumping of the loss was observed (no overfitting) |
| Std deviation | 1e-2          | |

During training the combined loss from cross entropy and regularization is minimised using the Adam optimizer. The training took about 30 min inside the udacity workspace.

### Result

#### Terminal output:
```
EPOCH 1/30 : Total loss = 48.78, batch loss = 0.170
EPOCH 2/30 : Total loss = 15.09, batch loss = 0.072
EPOCH 3/30 : Total loss = 11.86, batch loss = 0.125
EPOCH 4/30 : Total loss = 9.95, batch loss = 0.115
EPOCH 5/30 : Total loss = 8.39, batch loss = 0.134
EPOCH 6/30 : Total loss = 7.36, batch loss = 0.040
EPOCH 7/30 : Total loss = 6.66, batch loss = 0.055
EPOCH 8/30 : Total loss = 6.27, batch loss = 0.078
EPOCH 9/30 : Total loss = 5.68, batch loss = 0.061
EPOCH 10/30 : Total loss = 5.18, batch loss = 0.058
EPOCH 11/30 : Total loss = 7.48, batch loss = 0.030
EPOCH 12/30 : Total loss = 5.20, batch loss = 0.040
EPOCH 13/30 : Total loss = 4.61, batch loss = 0.103
EPOCH 14/30 : Total loss = 4.23, batch loss = 0.045
EPOCH 15/30 : Total loss = 5.02, batch loss = 0.076
EPOCH 16/30 : Total loss = 5.94, batch loss = 0.024
EPOCH 17/30 : Total loss = 4.18, batch loss = 0.050
EPOCH 18/30 : Total loss = 3.91, batch loss = 0.024
EPOCH 19/30 : Total loss = 3.80, batch loss = 0.023
EPOCH 20/30 : Total loss = 3.62, batch loss = 0.061
EPOCH 21/30 : Total loss = 3.41, batch loss = 0.016
EPOCH 22/30 : Total loss = 3.38, batch loss = 0.041
EPOCH 23/30 : Total loss = 4.35, batch loss = 0.049
EPOCH 24/30 : Total loss = 3.93, batch loss = 0.018
EPOCH 25/30 : Total loss = 3.40, batch loss = 0.037
EPOCH 26/30 : Total loss = 3.17, batch loss = 0.013
EPOCH 27/30 : Total loss = 3.08, batch loss = 0.063
EPOCH 28/30 : Total loss = 3.00, batch loss = 0.037
EPOCH 29/30 : Total loss = 2.98, batch loss = 0.019
EPOCH 30/30 : Total loss = 4.01, batch loss = 0.074
Training Finished. Saving test images to: ./runs/1541959209.7537029
```

 The total loss (summed over all batches) decreased non-monotonicaly. This indicates that the learning rate could have been set still lower. However due to the long training time and limited ressoruces, the training was not repeated.
 
#### Classification of test images
 
 ![alt text](runs/result.gif)
 

Different road material e.g. cobble stone and illumination gradients make problems for the model. Augmenting the data set to have more copies of the these kind of images can improve the accuracy.