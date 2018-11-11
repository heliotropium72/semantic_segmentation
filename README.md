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

### Training
The model was trained testing different sets of hyperparameters (epochs, batch size, dropout, learning rate, weight regularization and initialization).
The final chosen parameters were

| Parameter     | Value     | Comment  |
| ------------- |:-------------:| :-----:|
| Epochs        | 30            | the loss did not change anymore (NN has converged) |
| Batch size    | 2             | small values causes longer training time but due to the small amount of training images, this is not a problem. |
| Learning rate | 1e-4          |   |
| Dropout       | 0.7           |  |
| Regularization | 1e-3        | comparable small due to small batch size, no jumping of the loss was observed (no overfitting) |
| Std deviation | 1e-2          | |

During training the combined loss from cross entropy and regularization is minimised using the Adam optimizer. The training took about 30 min inside the udacity workspace.

### Result

#### Terminal output:
```
EPOCH 1/5 : Cross entropy loss = 53.413Cross entropy loss (last batch) = 0.182
EPOCH 2/5 : Cross entropy loss = 19.291Cross entropy loss (last batch) = 0.097
EPOCH 3/5 : Cross entropy loss = 15.982Cross entropy loss (last batch) = 0.074
EPOCH 4/5 : Cross entropy loss = 12.516Cross entropy loss (last batch) = 0.065
EPOCH 5/5 : Cross entropy loss = 10.978Cross entropy loss (last batch) = 0.077
```

 The total cross entropy loss (summed over all batches) decreased monotonicaly.
 
#### Classification of test images
 
 ![alt text](runs/result3.gif)
 

 
 ### Tips from Udacity
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
