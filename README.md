# About

This is a script to generate new images of human faces using the technique of generative adversarial networks (GAN), as described in the paper by [Ian J. Goodfellow](http://arxiv.org/abs/1406.2661).
GANs train two networks at the same time: A Generator (G) that draws/creates new images and a Discriminator (D) that distinguishes between real and fake images. G learns to trick D into thinking that his images are real (i.e. learns to produce good looking images). D learns to prevent getting tricked (i.e. learns what real images look like).
Ideally you end up with a G that produces beautiful images that look like real ones. On human faces that works reasonably well, probably because they contain a lot of structure (autoencoders work well on them too).

The code in this repository is a modified version of facebook's [eyescream project](https://github.com/facebook/eyescream).

# Example images

![Example faces](images/faces.png?raw=true "Example faces")

*32x32 grayscale images (upscaled due to screenshot) (generated from [Labeled Faces in the Wild, cropped version](http://conradsanderson.id.au/lfwcrop/))*

![Example faces grayscale 16x16](images/grayscale_best_0002_base.jpg?raw=true "Example faces grayscale 16x16")
![Example faces grayscale 32x32 c2f](images/grayscale_best_0002_c2f_32.jpg?raw=true "Example faces grayscale 32x32 c2f")

*Generated 16x16 grayscale images (left), scaled to 32x32 with coarse to fine / laplacian pyramid method (right). Scaling + coarse to fine often tends to cause distortions.*

![Example faces grayscale 16x16](images/color_best_0001_base.jpg?raw=true "Example faces color 16x16")
![Example faces grayscale 32x32 c2f](images/color_best_0001_c2f_32.jpg?raw=true "Example faces color 32x32 c2f")

*Generated 16x16 color images (left), scaled to 32x32 with coarse to fine / laplacian pyramid method (right). The distortions are even more visible here. Maybe a better architecture for the coarse to fine networks would help.*

<!---
![Example faces color](images/best_0008_c2f_32.jpg?raw=true "Example faces color")
*32x32 color images, generated first as 16x16 images and then scaled and refined.*
-->

# Requirements

To generate the dataset:
* [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) (original dataset without funneling)
* Python 2.7 (only tested with that version)
* Scipy + Numpy
* scikit-image

To run the GAN part:
* [Torch](http://torch.ch/) with the following libraries (most of them are probably already installed by default):
  * `nn` (`luarocks install nn`)
  * `paths` (`luarocks install paths`)
  * `image` (`luarocks install image`)
  * `optim` (`luarocks install optim`)
  * `cutorch` (`luarocks install cutorch`)
  * `cunn` (`luarocks install cunn`)
  * `dpnn` (`luarocks install dpnn`)
* [display](https://github.com/szym/display)
* Nvidia GPU with >= 4 GB memory
* cudnn3

# Usage

Building the dataset:
* Download [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) and extract it somewhere
* In `dataset/` run `python generate_dataset.py --path="/foo/bar/lfw"`, where `/foo/bar/lfw` is the path to your LFW dataset

To train a new model, follow these steps:
* Start `display` with `~/.display/run.js &`
* Open `http://localhost:8000` to see the training progress
* Train a 16x16 color generator with `th train.lua --scale=16` (add `--grayscale` for grayscale images)
* Train a 16 to 32 upscaler with `th train_c2f.lua --coarseSize=16 --fineSize=32` (add `--grayscale` for grayscale images)
* Sample images with `th sample.lua`

You might have to work with the command line parameters `--D_iterations` and `--G_iterations` to get decent performance.
Sometimes you also might have to change `--D_L2` (D's L2 norm) or `--G_L2` (G's L2 norm). (Similar parameters are available for L1.)
Learning speed can often be increased a little bit by increasing `--N_epoch` from 1000 to e.g. 5000 (random images per epoch).

# Architecture

G is a very small network which starts (by default) with a 100 dimensional noise vector, followed by one hidden layer of size 2048 (using PReLU activation) and the output layer, which consists of all image pixels (sigmoid activation). No dropout is used.

<!---
D for 16x16 images contains three subnetworks:
* A fully connected network that works directly on the images (without convolutions). It contains two layers of size 1024.
* A small convolutional network with two layers of 64 kernels (3x3) feeding into a 1024 fully connected layer. Contains one pooling layer.
* A small convolutional network with two layers of 32 and 64 kernels (5x5) feeding into a 1024 fully connected layer. Contains one pooling layers.

The three networks are concatenated to a vector of size 3*1024, followed by one fully connected layers of size 1024 and a final layer with 1 neuron.
All activations are PReLUs, except for the last layer, which uses sigmoid. Dropout is used between all fully connected layers. Spatial Dropout (drops out full kernel results) is used at the end of the convolutional layers.
The architecture is intended to capture the rough structure (via the fully connected subnetwork), as well as fine details (3x3 conv net, e.g. for eyes) and rougher details (5x5 conv net, e.g. for skin).
-->

D for 16x16 images is a convolutional neural net. All convolutional layers have a kernel size of 3x3. The rough architecture is:
* 128 kernels
* 128 kernels
* Average Pooling
* 512 kernels (2,2 stride)
* 1024 kernels (2,2 stride)
* Spatial Dropout
* Linear 1024
* Linear 1

All activations are PReLUs (exept for the last layer, which is sigmoid). The network also has a second branch, which only contains a tiny dense network and connects from the start (initial image) to the end (last linear layer), i.e. it skips most layers.
The second branch is necessary to kick off learning, otherwise the network learns nothing.

The coarse to fine network has the following structure:
* G:
  * SpatialConvolutionUpsample 64 kernels, 3x3, PReLU
  * SpatialConvolutionUpsample 64 kernels, 5x5, PReLU
  * SpatialConvolutionUpsample 128 kernels, 5x5, no activation
* D:
  * Convolution 64 kernels, 3x3, PReLU
  * Convolution 64 kernels, 3x3, PReLU
  * MaxPooling
  * Dropout
  * Linear 512, PReLU
  * Dropout
  * Linear 1, Sigmoid
  

Training is done with Adam (by default).

# Command Line Parameters

The `train.lua` script has the following parameters:
* `--batchSize` (default 16): The size of each batch, which will be split in two parts for G and D, making each one of them half-size. So a setting of 4 will create a batch of size of 2 for D (one image fake, one real) and another batch of size 2 for G. Because of that, the minimum size is 4 (and batches must be even sized).
* `--save` (default "logs"): Directory to save the weights to.
* `--saveFreq` (default 30): Save weights every N epochs.
* `--network` (default ""): Name of a weights file in the save directory to load.
* `--noplot`: Whether to NOT plot during training.
* `--N_epoch` (default 1000): How many examples to use during each epoch (-1 means "use the whole dataset").
* `--G_SGD_lr` (default 0.02): Learning rate for G's SGD, if SGD is used as the optimizer. (Note: There is no decay. You should use Adam or Adagrad.)
* `--G_SGD_momentum` (default 0): Momentum for G's SGD.
* `--D_SGD_lr` (default 0.02): Learning rate for D's SGD, if SGD is used as the optimizer. (Note: There is no decay. You should use Adam or Adagrad.)
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
* `--gpu` (default 0): Index of the GPU to train on (0-4 or -1 for cpu). Nothing is optimized for CPU.
* `--noiseDim` (default 100): Dimensionality of noise vector that will be fed into G.
* `--window` (default 3): ID of the first plotting window (in display), will also use about 3 window-ids beyond that.
* `--scale` (default 32): Scale of the images to train on (height, width). Loaded images will be converted to that size.
* `--autoencoder` (default ""): Path to the autoencoder to load (optional). Can be trained via `train_autoencoder.lua`. If set, the autoencoder will produce images and G will try to learn how to refine them so that D thinks they are real images. Didn't produce good results. Might also not work anymore.
* `--seed` (default 1): Seed to use for the RNG.
* `--weightsVisFreq` (default 0): How often to update the windows showing the activity of the network (only if >0; implies starting with `qlua` instead of `th` if set to >0).
* `--grayscale`: Whether to activate grayscale mode on the images, i.e. training will happen on grayscale images.
* `--denoise`: If added as parameter, the script will try to load a denoising autoencoder from `logs/denoiser_CxHxW.net`, where C is the number of image channels (1 or 3), H is the height of the images (see `--scale`) and W is the width. A denoiser can be trained using `train_denoiser.lua`.

The `train_c2f.lua` script has the following parameters:
* `--save` (default "logs"):       Same as above
* `--saveFreq` (default 30):           Same as above
* `--network` (default ""):           Same as above
* `--noplot` plot while training: Same as above
* `--D_sgd_lr` (default 0.02):         Same as above
* `--G_sgd_lr` (default 0.02):         Same as above
* `--D_sgd_momentum`   (default 0):            Same as above
* `--G_sgd_momentum`   (default 0):           Same as above
* `--batchSize`        (default 32):           Same as above
* `--N_epoch`          (default 1000):         Same as above
* `--G_L1`             (default 0):            Same as above
* `--G_L2`             (default 0e-6):         Same as above
* `--D_L1`             (default 1e-7):         Same as above
* `--D_L2`             (default 0e-6):        Same as above
* `--D_iterations`     (default 1):            Same as above
* `--G_iterations`     (default 1):            Same as above
* `--D_clamp`          (default 1):            Same as above
* `--G_clamp`          (default 5):            Same as above
* `--D_optmethod`      (default "adam"):       Same as above
* `--G_optmethod`      (default "adam"):       Same as above
* `--threads`          (default 4):            Same as above
* `--gpu`              (default 0):            Same as above
* `--noiseDim`         (default 100):          Same as above
* `--window`           (default 3):            Same as above
* `--coarseSize`       (default 16):           The size of images that will be upscaled to the fine size (e.g. 16 with fineSize=32 means that images will be scaled from 16 to 32 height/width)
* `--fineSize`         (default 32):           The target size. Images will be scaled to that size and then refined.
* `--grayscale`:                               Same as above
* `--seed`             (default 1):            Same as above
