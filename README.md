# Optical Music Recognition

This repository is a PyTorch implementation of several optical music recognition techniques. The goal is to take an image of a music score as input and produce a MIDI file as output. The project is still in its early stages, so the current focus is on the optical music recognition (OMR) pipeline.

## Preview 

## Project Description

### Motivation for Application

### Choice of Tech Stack

### Repository Structure

## Implemented Networks
### CRNN
The first neural network that I implemented is a CRNN that I reimplemented from the Camera Primus Paper. It uses the a Convolutional Recurrent Neural Network (CRNN) architecture which is characterized by a set of convolutional layers followed by several BiLSTMs and linear layers. Before each activation, batch normalization is performed to make sure that gradients are in an active regime. 

