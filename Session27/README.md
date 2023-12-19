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
3. Your map must have 3 targets A1>A2>A3 and your car/robot/object must target these alternatively.  
4. Train your best model upload a video on YouTube and share the URL  
5. Answer these questions in S24-Assignment-Solution:  
    1. What happens when "boundary-signal" is weak when compared to the last reward?  
    2. What happens when Temperature is reduced?  
    3. What is the effect of reducing (gamma) ?  
6. Heavy marks for creativity, map quality, targets, and other things. If you use the same maps or have just replicated shared code, you will get 0 for this assignment and a -50% advance deduction for the next assignment.  

## 📺 Video

Youtube Link: https://www.youtube.com/watch?v=rBd4Obkx4OE

## ⚡ Answer to questions  
1. What happens when "boundary-signal" is weak when compared to the last reward?  
    Answer: When the boundary-signal is weak relative to the last reward, the car tends to get stuck at the boundary. It struggles to return to the road or to reach its intended goal.

2. Temperature - Car was movement what fluctuating very rapidly

3. Gamma - Car was moving straight and getting fixed into loops.
