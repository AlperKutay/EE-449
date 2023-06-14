# EE-449
# Project Name

Homework 1 - Training Artificial Neural Network
Homework 2 - Evolutionary Algorithms
Homework 3 - Reinforcement Learning

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)

## Project Description
  Homework 1 - Training Artificial Neural Network

  In this homework, you will perform experiments on artificial neural network (ANN) training and draw
  conclusions from the experimental results. You will partially implement and train multi layer perceptron
  (MLP) and convolutional neural network (CNN) classifiers on CIFAR-10 dataset [1]. The implementations will be in Python language and you will be using PyTorch [2] and NumPy [3]. You can visit the
  link provided in the references [2, 3] to understand the usage of the libraries.

  Homework 2 - Evolutionary Algorithms
  
  In this homework, you will perform experiments on evolutionary algorithm and draw conclusions from
  the experimental results. The task is to create an image made of filled circles, visually similar to a given
  RGB source image (painting.png).
  The implementations will be in Python language and you will be using OpenCV package to draw the
  images. 

  Homework 3 - Reinforcement Learning
  
  In this homework, you will train a Reinforcement Learning agent in atari environment to win games.
  More specifically, you will create an RL model to play the first stage of the Super Mario Bros game. The
  implementations will be in Python language and you will be using PyTorch [1], OpenAI Gym [2], NES
  Emulator and Stable-Baselines3 [3]. You can visit the link provided in the references [1–3] to understand
  the usage of the libraries. You will also use TensorBoard to track the progress of your agent. You
  can download necessary libraries (except PyTorch, whose download method is introduced before) using
  pip install gym super mario bros==7.3.0 nes py stable-baselines3[extra]. Make sure PyTorch
  is already installed before installing Stable Baselines3, otherwise Stable Baselines3 may automatically
  download CPU version of PyTorch, regardless whether you have GPU or not.


## Installation
  ##Homework 1 - Training Artificial Neural Network
  ##Homework 2 - Evolutionary Algorithms
  pip install -r requirements.txt
  
  
  ##Homework 3 - Reinforcement Learning
  
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install wheel==0.38.4
  pip install setuptools==65.5.0 
  pip install gym_super_mario_bros==7.3.0 nes_py stable-baselines3[extra]

