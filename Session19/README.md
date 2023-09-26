# 🌐 ERA1 Session 19 Assignment 🌐

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Training Optimization](#data-augmentation)
- [Results](#results)


## 🎯 Problem Statement

First part of your assignment is to train your own UNet from scratch, you can use the dataset and strategy provided in this [link](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406). However, you need to train it 4 times:

- MP+Tr+BCE
- MP+Tr+Dice Loss
- StrConv+Tr+BCE
- StrConv+Ups+Dice Loss

and report your results.

 

Design a variation of a VAE that:
- takes in two inputs:
    - an MNIST image, and
    - its label (one hot encoded vector sent through an embedding layer)
- Training as you would train a VAE
- Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
- Now do this for CIFAR10 and share 25 images (1 stacked image)!
- Questions asked in the assignment are:
    - Share the MNIST notebook link ON GITHUB [100]
    - Share the CIFAR notebook link ON GITHUB [200]
    - Upload the 25 MNIST outputs PROPERLY labeled [250]
    - Upload the 25 CIFAR outputs PROPERLY labeled. [450]

## 📝 Introduction

Machine translation, the task of translating text from one language to another, has seen significant advancements with the introduction of deep learning models, particularly the transformer architecture. The transformer's unique structure, characterized by self-attention mechanisms, allows for parallel processing of sequences and offers a departure from the recurrent models traditionally used in sequence-to-sequence tasks.

For this project, we focus on translating English to French. English and French, being closely related Indo-European languages, share a lot of linguistic features, yet present enough challenges in terms of grammar, vocabulary, and idiomatic usage to make the translation task non-trivial.

The transformer model we employ consists of an encoder and a decoder. The encoder processes the input English sentence and compresses this information into a context vector. The decoder then utilizes this context to produce the translated French sentence. Both encoder and decoder comprise several layers of multi-head attention and feed-forward networks, making the transformer a powerful model for our task.

## 📐 Model Architecture

The transformer model consists of N stacked Encoder-Decoder blocks. Each block features multi-head attention mechanisms, Our transformer comprises 6 (N) Encoder-Decoder blocks. Tokens are embedded into 512-dimensional vectors (d_model). Each block utilizes multi-head attention with 8 (h) heads, followed by feed-forward networks of size 128 (d_ff). The model incorporates positional encodings for sequence context and projects the decoder's output to the target vocabulary for translation.

*** Model Dimensions: ***. 
- Embedding Dimension (d_model): This determines the size of the embedding vectors. A common choice is 512.  
- Feed-Forward Dimension (d_ff): Determines the size of the internal layers in the feed-forward networks present in both the encoder and decoder blocks.  
- Number of Attention Heads (h): Influences how many different attention patterns the model can learn.  

## 🔧 Training Optimization

To enhance the model's efficiency and performance:

- **Parameter Sharing**: The weights between the source and target embeddings are shared. This reduces the number of parameters and aligns the vector spaces, benefiting especially in tasks with closely related languages like English and French.

- **AMP (Automatic Mixed Precision)**: It speeds up training by leveraging both FP16 and FP32 data types, ensuring minimal loss in model accuracy.

- **Dynamic Padding**: Sequences in each batch are padded dynamically to the length of the longest sequence in that batch, reducing computational overhead.

- **One Cycle Policy (OCP)**: A learning rate scheduling technique that enables faster convergence and potentially better model outcomes.

These optimization techniques together ensure that the model trains faster, requires less memory, and achieves better performance.


## 📈 Results

The training loss achieved is 1.456. Detailed plots, training logs, and other related results can be viewed [here](./era1-session16-transformer-optimization.ipynb). 

![Alt text](image.png)

