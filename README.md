# My fork
The is a fork of jalused's [implementation](https://github.com/jalused/Deconvnet-keras) restructured and updated to be compatible with Keras 2.0 and a tensorflow backend.

# Deconvnet
This is a implementation of Deconvnet in keras, following Matthew D.Zeiler's paper [Visualizing and Understanding Convolutional Networks](http://arxiv.org/pdf/1311.2901v3.pdf)

## Feature
Given a pre-trained keras model, this repo can visualize features of specified layer including dense layer.  

## Dependencies
* [Keras](https://github.com/fchollet/keras) >= 2.1 (Tensorflow Backend)
* Python >= 3.4


## Shortage
* The code implements visualize function for only Convolution2D, MaxPooling2D, Flatten, Input, Activation layers, thus cannot handle other type of layers.
* The code support only plain networks, thus cannot visualize ResNet, Highway Networks or something.
