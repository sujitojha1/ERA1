# ERA1 Session 8 Assignment

## Problem Statement
Assignment is:  
  
1. Change the dataset to CIFAR10  
2. Make this network:  
    1. C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10  
    2. Keep the parameter count less than 50000  
    3. Try and add one layer to another  
    4. Max Epochs is 20  
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):  
    1. Network with Group Normalization  
    2. Network with Layer Normalization  
    3. Network with Batch Normalization  
4. Share these details  
    1. Training accuracy for 3 models  
    2. Test accuracy for 3 models  
    3. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images.  
5. write an explanatory README file that explains:
    1. what is your code all about,
    2. your findings for normalization techniques,
    3. add all your graphs
    4. your collection-of-misclassified-images 
6. Upload your complete assignment on GitHub and share the link on LMS

## Introduction

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch and the Albumentation library to achieve an accuracy of 85% on the CIFAR10 dataset. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./EVA8_S6_CIFAR10.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.

## Model Architecture

The model for this project is based on the C1C2C3C40 architecture with several modifications. Instead of max pooling, the network consists of 3 convolutional layers with 3x3 filters and a stride of 2. The final layer utilizes global average pooling (GAP). One layer uses depthwise separable convolution, while another layer uses dilated convolution. The architecture leverages mobileNetV2, which combines expand, depthwise, and pointwise convolution with residual connections.
Data Augmentation

## Data augmentation 
Augmentation is performed using the Albumentations library. Three techniques are applied in the training data loader: horizontal flipping, shiftScaleRotate, and coarseDropout. No dropout was included in the model as these data augmentation methods provide similar regularization effects.

Sample images,  
![augmentation](./images/dataloader_preview.png)

## Results

The model was trained for 25 epochs and achieved an accuracy of 84.64% on the test set. The total number of parameters in the model was under 200k. The training logs, as well as the output of torchsummary, are included in this notebook.

Trainling accuracy: 82.84 %
Test accuracy: 84.64 %

## Classwise Accuracy

![classwise_accuracy](./images/classwise_accuracy.png)

## Misclassified Images

Few Samples of misclassified images,  
![misclassified](./images/misclassified_images.png)
