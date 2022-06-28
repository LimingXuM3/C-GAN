# C-GAN

## Overview

A PyTorch(python 3.6) and OpenCV test implementation of the paper 'C-GAN: Medical Image Steganography Based on Convergent GANs with Localization (under review)'.

## Usage

#### Load images
Put the training samples into ./data/images/train;
Put the val samples into ./data/images/val;

#### Train a steganographic model

```
python train.py
```

#### Encode the message into an image

```
python test_encode.py
```

#### Decode the message from steganographic image

```
python test_decode.py
```

#### Additional details

- The implementation of proposed C-GAN model is partially based on SteganoGAN (https://github.com/DAI-Lab/SteganoGAN). We improve the SteganoGAN by adopting Symmetrical Divergence and localization to ensure the fast convergence.
- This is developed on a Linux machine running Ubuntu 16.04.
- Use GPU for the high speed computation.
- Due to partial samples in LGK dataset related to private information, so please e-mail me (xulimmail@gmail.com) if you need the dataset and I will share a private link with you.
