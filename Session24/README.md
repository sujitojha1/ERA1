# 🌐 ERA1 Session 24 Assignment: Reinforcement Learning

## 📌 Table of Contents
- Problem Statement
- Introduction
- Model Architecture
- PyTorch Lightning Implementation
- Training
- Results
- HuggingFace App
- Gradio Interface

## 🎯 Problem Statement
### Assignment 1:
- Share a screenshot of you running the whole code on your computer. To make sure it is your computer, put your own photo as the wallpaper (important)  
- You would need to find code for some files to run it successfully. You're of course going to find the files online and paste the code. 
- Once you are done, understand the code you copied and write a pseudo-code to explain what is happening. You need to explain:  
    - __init__  
    - getQValue  
    - computeValueFromQValue
    - computeActionFromQValues
    - getAction
    - update
- Write your pseudo-code on your S24 Assignment Submission page. For example, question you'll find is "Write a the pseudo-code (if you paste direct code, will be awarded 0) for __init__"


### Assignment 2: 

1. Create a new map of some other city for the code shared above  
2. Add a DNN with 1 more FC layer.  
Your map must have 3 targets A1>A2>A3 and your car/robot/object must target these alternatively. 
Train your best model upload a video on YouTube and share the URL
Answer these questions in S24-Assignment-Solution:
What happens when "boundary-signal" is weak when compared to the last reward?
What happens when Temperature is reduced? 
What is the effect of reducing (gamma) 
?
Heavy marks for creativity, map quality, targets, and other things. If you use the same maps or have just replicated shared code, you will get 0 for this assignment and a -50% advance deduction for the next assignment. 

## 📚 Introduction
In this session, we dive into the world of transformers by training a smaller version of the GPT model, known as "nanoGPT", from scratch. This exercise aims to deepen understanding of transformer architectures and their training process.

## 🏗 Model Architecture - nanoGPT
- Number of layers: 6
- Number of attention heads: 6
- Embedding dimension: 384
- Dropout rate: 0.2

## ⚡ Training Notebook
Training Notebook: [link](./ERA1_Session21_nanoGPT.ipynb)

## 📈 Results
![](./images/training_log.png)

## 🎧 HuggingFace App
The trained model is uploaded to HuggingFace Apps, enabling easy sharing and access. Link to the model: (insert link here)

## 🎨 Gradio Interface
A user-friendly interface is created using Gradio, allowing users to interact with the trained model, generating text based on custom input.

Link to Gradio Interface: (insert link here)

## 📺 Reference Video
[Training GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=40s)
