# -*- coding: utf-8 -*-
import argparse
import json
import os
from time import time

import torch

from steganoganphy import CGAN
from models import BasicCritic
from models import DenseDecoder
from models import BasicEncoder, DenseEncoder, ResidualEncoder

import numpy as np
import torchvision
from torchvision import transforms

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True, num_workers=8, batch_size=1, *args, **kwargs):

        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )

def main():
    torch.manual_seed(42)
    timestamp = str(time())
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--encoder', default="basic", type=str)
    parser.add_argument('--data_depth', default=3, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="images", type=str)
    parser.add_argument('--nc', type=int, default=3, help='input channel')
    parser.add_argument('--output', default=False, type=str)
    args = parser.parse_args()

    train = DataLoader(os.path.join("data", args.dataset, "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False)

    encoder = {
        "basic": BasicEncoder,
    }[args.encoder]
    steganogan = CGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        hidden_size=args.hidden_size,
        cuda=True,
        verbose=True,
        log_dir=os.path.join('trained_models', timestamp)
    )
    with open(os.path.join("trained_models", timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    steganogan.fit(train, validation, epochs=args.epochs)
    steganogan.save(os.path.join("trained_models", timestamp, "weights.steg"))
    if args.output:
        steganogan.save(args.output)

if __name__ == '__main__':
    main()
