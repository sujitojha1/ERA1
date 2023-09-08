# 🌐 ERA1 Session 16 Assignment 🌐

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Training Optimization](#data-augmentation)
- [Results](#results)


## 🎯 Problem Statement

1. Pick the "en-fr" dataset from opus_books
2. Remove all English sentences with more than 150 "tokens"
3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8
5. Enjoy! 

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

The training loss achieved is 1.522. Detailed plots, training logs, and other related results can be viewed [here](./era1-session16-transformer-optimization-ipynb.ipynb). 

![Alt text](image.png)

