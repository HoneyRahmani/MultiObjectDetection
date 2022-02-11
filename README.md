
## Table of Content
> * [Multi Object Detection - Pytorch](#MultiObjectDetection-Pytorch)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)

# Multi Object Detection - Pytorch
## About the Project

In this project, the COCO dataset is used to train YOLO-v3 model that is a regression/classification-based method.

![recipe](https://user-images.githubusercontent.com/75105778/153649787-46a34ba4-83b7-4a1f-9e9f-87babf9a3d95.jpg)


## About Database

Find dataset in https:/ / github. com/pjreddie/darknet 

The images folder contains two subfolders called train2014 and val2014 with
82783 and 40504 images, respectively.
The labels folder is included two subfolders called train2014 and val2014 with
82081 and 40137 text files, respectively. These text files contain the bounding box coordinates of the objects in the images with [ID, xc, yc, w, h] format.

ID: the object ID

xc: the centroid coordinates

yc: the centroid coordinates

w: width of the bounding box

h: height of the bounding box

List of images that will be used to train the model is in the trainvalno5k.txt file and List of images that will be used for validation the model is in the 5k.txt file

## Built with
* Pytorch
* Model is YOLO-v3.
* The model output comprises the following elements:
[x, y, w, h] of bounding boxes
An objectness score
Class predictions for 80 object categories 
Thus, it is defined a function that compute loss for six elements. It is composed with:
The mean squared error is used to calculate loss of x, y, w, h.
The binary cross-entropy is used to calculate loss of the objectness score.
The binary cross-entropy is used to calculate loss of class predictions.
* Intersection over Union (IOU) performance metric.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch
