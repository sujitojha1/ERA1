# 🌐 ERA1 Session 13 Assignment 🌐

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Introduction](#-introduction)
3. [Model Architecture](#-model-architecture)
4. [Data Augmentation](#-data-augmentation)
5. [PyTorch Lightning Implementation](#-pytorch-lightning-implementation)
6. [Results](#-results)
7. [Misclassified Images](#-misclassified-images)
8. [Gradio App](#-gradio-app)

## 🎯 Problem Statement

1. Move the code to PytorchLightning  
2. Train the model to reach such that all of these are true:  
    1. Class accuracy is more than 75%   
    2. No Obj accuracy of more than 95%  
    3. Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)  
    4. Ideally trailed till 40 epochs  
3. Add these training features:  
    1. Add multi-resolution training - the code shared trains only on one resolution 416  
    2. Add Implement Mosaic Augmentation only 75% of the times  
    3. Train on float16  
    4. GradCam must be implemented.  
4. Things that are allowed due to HW constraints:
    1. Change of batch size
    2. Change of resolution
    3. Change of OCP parameters
5. Once done:  
    1. Move the app to HuggingFace Spaces  
    2. Allow custom upload of images  
    3. Share some samples from the existing dataset  
    4. Show the GradCAM output for the image that the user uploads as well as for the samples. 
    5. Mention things like:  
        1. classes that your model support  
        2. link to the actual model  
6. Assignment:
    1. Share HuggingFace App Link  
    2. Share LightningCode Link on Github  
    3. Share notebook link (with logs) on GitHub  
7. Recommendations:  
    1. Make sure to link your [GDrive](https://towardsdatascience.com/different-ways-to-connect-google-drive-to-a-google-colab-notebook-pt-1-de03433d2f7a).  
    2. Once you're using OCP, your model cannot stop training. It would be better to pre-calculate the schedule and store it on a text file, and then read this file and take note of where your training got interrupted, so you can start from that point.  
    3. NUM_WORKERS = 0 is for me, not for you!  
    4. Training will take 3-4 days, so make sure that you finish the assignment by max by Tuesday, and put the model on training.  
    5. Remember that you can use Kaggle or Gradient Notebooks as well!  
    6. Remember OCP calculated for a particular batch, needs the model to run on that batch size only.  
    7. Feel free to pick a better point on the OCP graph compared to what's being proposed.  
    8. MAP calculations take a lot of time, and perform only on the last epoch.  
    9. Reduce the calls to plot_couple_of_examples as that might take a lot of webpage memory  
    10. The current code overwrites the checkpoints!  
    11. Write the saved model back to google drive, and add schedule value to its name!  
    12. You're free to use any platform other than Colab.  
    13. Fast for one day to celebrate "Global GPU Day" and use the money saved (500?)  to train your model on a paid infrastructure!  
    14. Add Implement Mosaic Augmentation only 75% of the times << Heavily recommended to keep the ratio between 0.5-0.75. Change the dataloader for test. A fully mosaic logic will train it on small and medium objects only, and we don't want that.  
    15. Instead of from dataset import YOLODataset, do from dataset_org import YOLODataset. We want to see the performance of test on single image and not on mosaic. This will get a slightly better idea on what the results would be when you upload the model.  

## 📚 Introduction

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch, PyTorch Lightning and the Albumentation library to achieve an accuracy of 85% on the CIFAR10 dataset. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./ERA1_S12_CIFAR10_Pytorch_lightning.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.

## 🏗 Model Architecture

The custom ResNet model for CIFAR10 features a preprocessing layer, three primary layers, and a fully connected (FC) layer. The layers incorporate Convolutional layers, MaxPooling, Batch Normalization, ReLU activations, and Residual Blocks to handle feature extraction and to mitigate the issue of vanishing gradients. The model ends with a SoftMax function for class probability scores, leveraging the depth of the model and residual connections for efficient classification on the CIFAR10 dataset.


## 🎨 Data augmentation 
The model uses data augmentation techniques to improve robustness and prevent overfitting by increasing data diversity. This includes RandomCrop (32, 32), applied after a 4-pixel padding, to enhance positional robustness by randomly cropping images. FlipLR is used for introducing orientation robustness by mirroring images along the vertical axis. Lastly, CutOut (8, 8) randomly masks parts of the image, promoting the model's ability to learn from various regions, thereby improving its robustness to occlusions.

## ⚡ PyTorch Lightning Implementation

PyTorch Lightning provides a high-level interface to the PyTorch framework, simplifying many complex tasks and enabling more structured and cleaner code. It abstracts away most of the boilerplate code required for training, validation, and testing, allowing researchers and developers to focus on the actual model logic.

We wrapped our model using PyTorch Lightning module
- Data loading using PyTorch Bolt and Transformation using Albumentation
- Training, Validation and Test Steps
- One Cycle Learning Rate with Adam optimizer


## 📈 Results

The model was trained for 24 epochs and achieved an accuracy of 89.64% on the test set. 

![loss_accuracy](./images/loss_accuracy_plots.png)

## ❌ Misclassified Images with GradCAM

Few Samples of misclassified images,  
![misclassified](./images/miss_classified_images.png)

## 🎧 Gradio App

For this project, a Gradio interface has been set up to let users interact with the trained CIFAR10 model. Users can upload images, adjust GradCAM parameters, and view the model's predictions along with GradCAM visualizations and misclassified images.

[Link](https://huggingface.co/spaces/sujitojha/CIFAR10_CustomResnet_GradCAM)

![Gradio App](./images/gradio_app.png)
