# About

This is a script to generate new images of human faces using the technique of generative adversarial networks (GAN), as described in the paper by [Ian J. Goodfellow](http://arxiv.org/abs/1406.2661). The code is based on a modified version of facebook's [eyescream project](https://github.com/facebook/eyescream).

# Example images

# Requirements

To run this script optimally you need
* Torch
* Torch image library (`luarocks install image`)
* [display](https://github.com/szym/display)
* [Labeled Faces in the Wild, cropped version](http://conradsanderson.id.au/lfwcrop/)

# Execution

To train a new model, follow these steps:
* Download the [lfw cropped dataset](http://conradsanderson.id.au/lfwcrop/).
* Clone the repository.
* Change in `train.lua` the line `DATASET.setDirs({"/path/to/lfw_cropped"})` to match your dataset's directory.
* Start the training with the command: `qlua train_cats.lua --gpu=0 --plot --N_epoch=2000`, which will train on the GPU, plot images during the training and use 2000 random images during each epoch.