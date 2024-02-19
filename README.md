# Optical Music Recognition

This repository is a PyTorch implementation of several optical music recognition techniques. The goal is to take an image of a music score as input and produce a MIDI file as output. Currently, the main implemented architecture is a CRNN. A Transformer architecture will follow soon.

## Preview 

## Project Description

### Motivation for Application
As a musician for many years, I often had to play according to musical notation - without having an idea what the notation may sound like. Converting music scores to a playable sound representation. This project is an endeavour to automate this process to make the learning process more enjoyable.

### Choice of Tech Stack
I selected PyTorch for this project due to its dynamic computation graph for flexible model adaptation, comprehensive ecosystem for accelerated development, and robust GPU support for efficient training and inference, making it optimal for advanced deep learning needs. Additionally, Pytorch has a lightweight integration for iOS mobile devices which may be beneficial when deploying the project in the future. 

### Repository Structure
`/data/`: Contains datasets and scripts for data preprocessing
- `/data/primus/`: The Camera Primus Dataset
- `/data/monophonic_nn.py`: Neural networks for monophonic models, i.e. single instrument scores (no two-handed piano)

`/networks/`: Contains neural networks for OMR tasks

## Implemented Networks
### CRNN
The first implemented neural network is a CRNN that I reimplemented from the Camera Primus Paper. It uses the a Convolutional Recurrent Neural Network (CRNN) architecture which is characterized by a set of convolutional layers followed by several BiLSTMs and linear layers. Before each activation, batch normalization is performed to make sure that gradients are in an active regime. 

The implementation of this stage is almost completed. However, the training in the original paper had 64,000 epochs which is infeasible in terms of available compute power at this point.

### TrOMR
The second architecture is a transformer architecture reimplemented from the TrOMR Paper. It uses Transfer Learning with a pretrained Vision Transformer to predict sequences of music symbols.
