# 🌐 ERA1 Session 16 Assignment 🌐

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [PyTorch Lightning Implementation](#pytorch-lightning-implementation)
- [Results](#results)
- [Gradio App](#gradio-app)

## 🎯 Problem Statement

1. Pick the "en-fr" dataset from opus_books
2. Remove all English sentences with more than 150 "tokens"
3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8
5. Enjoy! 

## 📚 Introduction

Object detection is a pivotal aspect of computer vision, enabling machines to recognize and pinpoint multiple objects within an image or video feed. Over the years, a myriad of approaches have been introduced to tackle this challenge. Among these, the YOLO (You Only Look Once) series has garnered significant attention for its capability to detect objects in real-time without sacrificing accuracy. YOLOv3, a variant in this lineage, brings further enhancements to this approach, boasting improved precision and faster processing times.

Central to our exploration is the PASCAL VOC dataset. Renowned within the computer vision community, PASCAL VOC provides a rich collection of images spread across diverse categories such as 'aeroplane', 'bird', 'car', and more. Utilizing YOLOv3 to perform object detection on this dataset not only offers a deep understanding of the model's capabilities but also highlights the intricacies and challenges associated with detecting a wide array of objects in varying scenarios.

## 🏗 Model Architecture - YOLOv3

YOLOv3, standing for "You Only Look Once version 3," is an evolution in the YOLO series that offers real-time object detection with remarkable accuracy. Unlike its predecessors, YOLOv3 makes detection at three different scales and uses three sizes of anchor boxes for each detection scale, allowing for better detection of objects of various sizes.

The architecture of YOLOv3 is based on Darknet-53, a 53-layer network trained on the ImageNet dataset. This foundational network is followed by a series of convolutional layers tailored for object detection. 

![yolo architecture](./images/YoloV3_architecture.jpeg)


## 🎨 Data Augmentation
This involves enhancing the diversity of training data using techniques like Mosaic Augmentation, ensuring the model is better generalized and robust against unseen data.

![mosaic augmentation](./images/mosaic_augmentation.png)

## ⚡ PyTorch Lightning Implementation

We wrapped the YOLOv3 model using PyTorch Lightning, streamlining the training with a custom loss function, One Cycle Learning Rate policy, and float16 mixed-precision training for enhanced performance and efficiency.


## 📈 Results

- Class accuracy is: 87.600876%
- No obj accuracy is: 98.140427%
- Obj accuracy is: 80.607513%

[notebook](./ERA1_S13_YOLOv3_Pytorch_lightning.ipynb)


## 🎧 Gradio App
A user-friendly interface constructed using Gradio. This app enables users to upload custom images and view model predictions.

[Link to Gradio App](https://huggingface.co/spaces/sujitojha/Object_Detection_YOLOv3)




