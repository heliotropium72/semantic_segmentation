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

### Training
The model was trained testing different sets of hyperparameters (epochs, batch size, dropout, learning rate, regularization).

### Result

| Parameter     | First run     | Second run  |
| ------------- |:-------------:| :-----:|
| Folder        | [here](run/1541429909.976964) | [here](run/1541432389.5504768)|
| Epochs        | 30            | 20 |
| Batch size    | 15            | 15 |
| Learning rate | 0.001         |  0.0008 |
| Dropout       | 0.5           | 0.7 |
| Regularizatioin | 1e-4        | 1e-3 |
| CEL batch     | 0.164         | 0.176 |
| CEL total     | -             | 3.65 |

Result in run/1541429909.976964
Epoch 30: single batch CEL=0.164, generally decreasing

First run
![alt text](runs/result2.gif)

In the first run, only the cross entropy loss of the last batch of images was reported to the console. It did not decrease constantly but with an overall trend. That can be expected for a random sample.

 Second run
 ![alt text](runs/result3.gif)
 
 The total cross entropy loss (summed over all batches) decreased monotonicly.
 
 ### Tips from Udacity
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.