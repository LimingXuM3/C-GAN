import zlib
from math import exp

import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d

rs = RSCodec(250)

import gc
import inspect
import json
import os
from collections import Counter
from torch.autograd import Variable
import numpy as np
import imageio
import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm
import torch.autograd as autograd

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def text_to_bits(text):
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits):
    return bytearray_to_text(bits_to_bytearray(bits))

def compute_gradient_penalty_gp(D, payload, generated):    
    alpha = Tensor(np.random.random((payload.size(0), 3, 1, 1)))
    interpolates = (alpha * payload + (1 - alpha) * generated).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs = None,
                create_graph = True, 
                retain_graph = True,
                only_inputs = True,
                )[0]
        
    gradients = gradients.view(payload.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim = 1) - 1)**2).mean()
        
    return gradient_penalty

def compute_gradient_penalty_div(D, real_imgs,fake_imgs, k=2, p=6):
    real_output = D(real_imgs.requires_grad_(True))
    fake_output = D(fake_imgs.requires_grad_(True))
    
    real_grad = autograd.grad(
        real_output, 
        real_imgs, 
        grad_outputs = None,
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True,
    )[0]
    
    fake_grad = autograd.grad(
        fake_output, 
        fake_imgs, 
        grad_outputs = None,
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True,
    )[0]
    
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
 
    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    
    return div_gp

def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result


def bits_to_bytearray(bits):
    ints = []
    bits=np.array(bits)
    bits=bits+0
    bits=bits.tolist()
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))

    return bytearray(ints)


def text_to_bytearray(text):
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x

def bytearray_to_text(x):
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        return False


def first_element(storage, loc):
    return storage


def gaussian(window_size, sigma):
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):

    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
