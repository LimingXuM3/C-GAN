# -*- coding: utf-8 -*-
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
import torch.nn as nn
import argparse

from utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits, compute_gradient_penalty_gp, compute_gradient_penalty_div

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--jcbSize', type=int, default=8, help='size of sub-dimension for computing jacobian')
parser.add_argument('--imagesize', type=int, default=360, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=10, help='dimension of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='input channel')
parser.add_argument('--real_label', type=int, default=1)
parser.add_argument('--fake_label', type=int, default=0)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--alpha', type=float, default=20, help='weight for reconstruction')
parser.add_argument('--beta', type=float, default=0.01, help='weight for orthogonal loss')
parser.add_argument('--theta', type=float, default=0.1, help='weight for adversarial loss of recontructed images')
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--delta', type=float, default=0.0001, help='step size for computing jacobian')
parser.add_argument('--var', type=float, default=3, help='variance of gaussian noise')
opt = parser.parse_args()

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]

criterion = nn.BCELoss()
criterion_L1 = nn.L1Loss()
label = torch.Tensor(opt.batch_size)
input = torch.Tensor(opt.batch_size, opt.nc, opt.imagesize, opt.imagesize)
input_tile = torch.Tensor(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize)
noise = torch.Tensor(opt.batch_size, opt.nz, 1, 1)
regress_img = torch.Tensor(opt.batch_size, opt.nc, opt.imagesize, opt.imagesize)
pos_noise = torch.Tensor(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize)
zero_noise = torch.Tensor(opt.batch_size, opt.nz, 1, 1)
fixed_noise = torch.Tensor(opt.batch_size, opt.nz, 1, 1).normal_(0, opt.var)
label = torch.Tensor(opt.batch_size)
eye_label = torch.Tensor(opt.batch_size, opt.jcbSize, opt.jcbSize)
eye_nz = torch.Tensor(opt.batch_size, opt.jcbSize, opt.nz)
dimen = int((opt.batch_size * opt.nc * opt.imagesize * opt.imagesize) / opt.nz)

if opt.cuda:
    criterion.cuda()
    criterion_L1.cuda()
    input, label = input.cuda(), label.cuda()
    input_tile = input_tile.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    zero_noise = zero_noise.cuda()
    pos_noise = pos_noise.cuda()
    regress_img = regress_img.cuda()
    eye_label = eye_label.cuda()
    eye_nz = eye_nz.cuda()

class CGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, critic, cuda=False, verbose=False, log_dir=None, **kwargs):
        self.verbose = verbose
        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        kwargs['data_depth'] = data_depth
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)

    def _random_data(self, cover):
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _generator(self, cover, payload, quantize=False):
        generated = self.encoder(cover, payload)
        
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        return generated
    
    def _decoder(self, generated):
        decoded = self.decoder(generated)
        
        return decoded

    def _critic(self, image):
        return torch.mean(self.critic(image))

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, decoder_optimizer
    
    def _fit_critic(self, train, metrics):
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            #gradient_penalty = compute_gradient_penalty_gp(self._critic, payload.data, generated.data)
            gradient_penalty = compute_gradient_penalty_div(self._critic, payload.data, generated.data)
            
            self.critic_optimizer.zero_grad()
            (cover_score - generated_score + 10 * gradient_penalty * gradient_penalty).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    def _fit_coders(self, train, metrics):
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()

            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self._generator(cover, payload)
            decoded = self._decoder(generated)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            input.resize_as_(cover).copy_(cover)
            label.resize_(opt.batch_size).fill_(opt.real_label)
            regress_img.resize_as_(cover).copy_(cover)
            inputv = Variable(input)
            labelv = Variable(label)
            regress_imgv = Variable(regress_img)

            output = self._critic(inputv)
            noise.resize_(opt.batch_size, 3, 360, 360).normal_(0, opt.var)
            noisev = Variable(noise)
            G_x_z = self._generator(inputv, noisev)

            zero_noise.resize_(opt.batch_size, 3, 360, 360).fill_(0)
            zero_noisev = Variable(zero_noise)
            G_x_0 = self._generator(inputv, zero_noisev)
                    
            
            labelv = Variable(label.fill_(opt.real_label))             
            output = self._critic(G_x_z)
            outputs = torch.Tensor(opt.batch_size)
            outputs.resize_(opt.batch_size).fill_(output)
            outputs = outputs.to(self.device)
            
            output_0 = self._critic(G_x_0)
            outputs_0 = torch.Tensor(opt.batch_size)
            outputs_0.resize_(opt.batch_size).fill_(output_0).cuda()
            outputs_0 = outputs_0.to(self.device)
            
            errG = criterion(outputs, labelv) + opt.theta * criterion(outputs_0, labelv)
            errL1 = criterion_L1(G_x_0, regress_imgv)

            # localization
            input_tile.resize_(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize)
            real_cpu_tile = cover.repeat(opt.jcbSize, 1, 1, 1, 1)
            real_cpu_tile = real_cpu_tile.transpose(0, 1).contiguous()
            real_cpu_tile = real_cpu_tile.view(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize)
            input_tile.copy_(real_cpu_tile)
            input_tilev = Variable(input_tile)

            eye_label.resize_(opt.batch_size, opt.jcbSize, opt.jcbSize).copy_(torch.eye(opt.jcbSize))
            eye_labelv = Variable(eye_label)            
            eye_nz.resize_(dimen, opt.batch_size, opt.jcbSize, opt.nz).copy_(torch.eye(opt.nz)[torch.randperm(opt.nz)[:opt.jcbSize]])
            pos_noise_flatten = (opt.delta * eye_nz).view(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize)
            pos_noise.resize_(opt.batch_size * opt.jcbSize, opt.nc, opt.imagesize, opt.imagesize).copy_(pos_noise_flatten)
            pos_noisev = Variable(pos_noise)

            Jx = (self._generator(input_tilev, pos_noisev) - self._generator(input_tilev, -pos_noisev)) / (2 * opt.delta)
            Jx = Jx.view(opt.batch_size, opt.jcbSize, -1)
            Jx_T = Jx.transpose(1, 2)
            
            errOrth = criterion_L1(torch.matmul(Jx, Jx_T), opt.gamma * eye_labelv)

            self.decoder_optimizer.zero_grad()
            (errG + errL1 * opt.alpha + errOrth * opt.beta).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _validate(self, validate, metrics):
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self._generator(cover, payload)
            decoded = self._decoder(generated)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def _generate_samples(self, samples_path, cover, epoch):
        cover = cover.to(self.device)
        payload = self._random_data(cover)
        generated = self._generator(cover, payload)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover.png'.format(sample))
            sample_name = '{}.generated-{:2d}.png'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            image = (cover[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0

            image = sampled / 2.0
            imageio.imwrite(sample_path, (255.0 * image).astype('uint8'))

    def fit(self, train, validate, epochs=5):
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}
            print('metrics:', metrics)
            print('metrics.items:', metrics.items())

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)
            

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.bpp-{:03f}.p'.format(
                    self.epochs, self.fit_metrics['val.bpp'])

                self.save(os.path.join(self.log_dir, save_name))
                self._generate_samples(self.samples_path, sample_cover, epoch)

            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()

    def _make_payload(self, width, height, depth, text):
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)

    def encode(self, cover, output, text):
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

    def decode(self, image):

        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0
        print(image)

        # split and decode messages
        candidates = Counter()
        bits = image.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1      
        

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]
        return candidate

    def save(self, path):
        torch.save(self, path)

    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        if architecture and not path:
            model_name = '{}.steg'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError('Please provide either an architecture or a path to pretrained model.')

        steganogan = torch.load(path, map_location='cpu')
        steganogan.verbose = verbose

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()

        steganogan.set_device(cuda)
        return steganogan
