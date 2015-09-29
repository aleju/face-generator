# About

This is a script to generate new images of human faces using the technique of generative adversarial networks (GAN), as described in the paper by [Ian J. Goodfellow](http://arxiv.org/abs/1406.2661).
GANs train two networks at the same time: A Generator (G) that draws/creates new images and a Discriminator (D) that distinguishes between real and fake images. G learns to trick D into thinking that his images are real (i.e. learns to produce good looking images). D learns to prevent getting tricked (i.e. learns what real images look like).
Ideally you end up with a G that produces beautiful images that look like real ones. On human faces that works reasonably well, probably because they contain a lot of structure (autoencoders work well on them too).

The code of this repository is a modified version of facebook's [eyescream project](https://github.com/facebook/eyescream). It does not use the upscaling feature (laplacian pyramid).

Note: This project is not fully finished yet, but mostly funtional.

# Example images

# Requirements

To run this script optimally you need
* [Torch](http://torch.ch/) with the following libraries (most of them are probably already installed by default):
  * `nn` (`luarocks install nn`)
  * `paths` (`luarocks install paths`)
  * `image` (`luarocks install image`)
  * `optim` (`luarocks install optim`)
  * `cutorch` (`luarocks install cutorch`) (only for GPU training)
  * `cunn` (`luarocks install cunn`) (only for GPU training)
* [display](https://github.com/szym/display)
* [Labeled Faces in the Wild, cropped version](http://conradsanderson.id.au/lfwcrop/) (colored)

# Execution

To train a new model, follow these steps:
* Download the [lfw cropped dataset](http://conradsanderson.id.au/lfwcrop/). You should chose the colored dataset as you can activate `--grayscale` mode via the command line parameters.
* Clone the repository.
* Change in `train.lua` the line `DATASET.setDirs({"/path/to/lfw_cropped"})` to match your dataset's directory.
* Start the training with the command `th train.lua --gpu=0 --plot --grayscale`, which will train on grayscale images on the GPU and plot via `display` during the training. You can see the plots by opening `http://localhost:8000`.

# Architecture

G is a very small network which starts (by default) with a 100 dimensional noise vector, followed by one hidden layer of size 2048 (using PReLU activation) and the output layer, which consists of all image pixels (Sigmoid). No dropout is used.

D contains three subnetworks:
* A fully connected network that works directly on the images (without convolutions). It contains two layers of size 1024.
* A small convolutional network with two layers of 64 kernels (3x3) feeding into a 1024 fully connected layer.
* A small convolutional network with two layers of 32 kernels (5x5) feeding into a 1024 fully connected layer.

The three networks are concatenated to a vector of size 3*1024, followed by two fully connected layers of size 2048 and a last layer with 1 neuron.
All activations are PReLUs, except for the last layer, which uses sigmoid. Dropout is used between all fully connected layers. Spatial Dropout (drops out full kernel results) is used at the end of the convolutional layers.
The architecture is intended to capture the rough structure (via the fully connected subnetwork), as well as fine details (3x3 conv net, e.g. for eyes) and rougher details (5x5 conv net, e.g. for skin).

# Command Line Parameters

The `train.lua` script has the following parameters:
* `--batchSize` (default 16): The size of each batch, which will be split in two parts for G and D, making each one of them half-size. So a setting of 4 will create a batch of size of 2 for D (one image fake, one real) and another batch of size 2 for G. Because of that, the minimum size is 4 (and batches must be even sized).
* `--save` (default "logs"): Directory to save the weights to.
* `--saveFreq` (default 30): Save weights every N epochs.
* `--network` (default ""): Name of a weights file in the save directory to load.
* `--plot`: Whether to plot during training.
* `--N_epoch` (default 1000): How many examples to use during each epoch (-1 means "use the whole dataset").
* `--G_SGD_lr` (default 0.02): Learning rate for G's SGD, if SGD is used as the optimizer. (Note: There is no decay. You better use Adam or Adagrad.)
* `--G_SGD_momentum` (default 0): Momentum for G's SGD.
* `--D_SGD_lr` (default 0.02): Learning rate for D's SGD, if SGD is used as the optimizer. (Note: There is no decay. You better use Adam or Adagrad.)
* `--D_SGD_momentum` (default 0): Momentum for D's SGD.
* `--G_adam_lr` (default -1): Adam learning rate for G (-1 is automatic).
* `--D_adam_lr` (default -1): Adam learning rate for D (-1 is automatic).
* `--G_L1` (default 0): L1 penalty on the weights of G.
* `--G_L2` (default 0): L2 penalty on the weights of G.
* `--D_L1` (default 0): L1 penalty on the weights of D.
* `--D_L2` (default 1e-4): L2 penalty on the weights of D.
* `--D_iterations` (default 1): How often to optimize D per batch (e.g. 2 for D and 1 for G means that D will be trained twice as much).
* `--G_iterations` (default 1): How often to optimize G per batch.
* `--D_maxAcc` (default 1.01): Stop training of D roughly around that accuracy level until G has catched up. (Sounds good in theory, doesn't produce good results in practice.)
* `--D_clamp` (default 1): To which value to clamp D's gradients (e.g. 5 means -5 to +5, 0 is off).
* `--G_clamp` (default 5): To which value to clamp G's gradients (e.g. 5 means -5 to +5, 0 is off).
* `--D_optmethod` (default "adam"): Optimizer to use for D, either "sgd" or "adam" or "adagrad".
* `--G_optmethod` (default "adam"): Optimizer to use for D, either "sgd" or "adam" or "adagrad".
* `--threads` (default 8): Number of threads.
* `--gpu` (default -1): Index of the GPU to train on (0-4 or -1 for cpu).
* `--noiseDim` (default 100): Dimensionality of noise vector that will be fed into G.
* `--window` (default 3): ID of the first plotting window (in display), will also use about 3 window-ids beyond that.
* `--scale` (default 32): Scale of the images to train on (height, width). Loaded images will be converted to that size.
* `--autoencoder` (default ""): Path to the autoencoder to load (optional). Can be trained via `train_autoencoder.lua`. If set, the autoencoder will produce images and G will try to learn how to refine them so that D thinks they are real images. Didn't produce good results.
* `--seed` (default 1): Seed to use for the RNG.
* `--weightsVisFreq` (default 0): How often to update the windows showing the activity of the network (only if >0; implies starting with `qlua` instead of `th` if set to >0).
* `--grayscale`: Whether to activate grayscale mode on the images, i.e. training will happen on grayscale images.
