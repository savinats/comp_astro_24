# daneel/dream/gan.py
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Attempt to import the Logger you already have. If missing, continue without it.
try:
    from .logger import Logger
except Exception:
    Logger = None

from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler

class LightCurveProcessor:
    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True, gaussian_sigma=2):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize
        self.gaussian_sigma = gaussian_sigma

    def fourier_transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.abs(fft.fft(X, n=X.size))

    def process_xy(self, df_x):
        if self.fourier:
            df_x = df_x.apply(self.fourier_transform, axis=1)
            arr = np.vstack([row for row in df_x])
            arr = arr[:, : arr.shape[1] // 2]
        else:
            arr = df_x.values

        if self.normalize:
            arr = normalize(arr)

        if self.gaussian:
            arr = ndimage.gaussian_filter(arr, sigma=self.gaussian_sigma)

        if self.standardize:
            scaler = StandardScaler()
            arr = scaler.fit_transform(arr)

        return arr

    def make_square(self, arr, target_size=1600):
        N, M = arr.shape
        if M >= target_size:
            out = arr[:, :target_size]
        else:
            pad = target_size - M
            out = np.hstack([arr, np.zeros((N, pad), dtype=arr.dtype)])
        return out

class LightCurveDataset(Dataset):
    def __init__(self, csv_path, processor: LightCurveProcessor, split='train'):
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        self.X_df = df.drop('LABEL', axis=1)
        self.processor = processor

        arr = self.processor.process_xy(self.X_df)
        arr = self.processor.make_square(arr, target_size=1600)  # 40*40
        arr = arr.reshape((-1, 40, 40))  # (N, 40, 40)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)  # [0,1]
        arr = 2.0 * arr - 1.0  # [-1,1]
        self.data = arr.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 5 * 5),
            nn.BatchNorm1d(ngf * 8 * 5 * 5),
            nn.ReLU(True),

            View((-1, ngf * 8, 5, 5)),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),  # 10x10
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),  # 5x5
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            View((-1, ndf * 4 * 5 * 5)),
            nn.Linear(ndf * 4 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape[1:] if isinstance(shape, tuple) else shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class GAN:
    def __init__(self, params):
        """
        params: dict with keys:
          - dataset_path (path to folder containing exoTrain.csv/exoTest.csv)
          - out_dir (optional) base output dir
          - device (cpu or cuda)
          - nz (latent dim)
          - epochs, batch_size, lr
        """
        self.dataset_path = params.get('dataset_path', '.')
        self.out_dir = params.get('out_dir', './data')
        self.device = torch.device("cpu")
        self.nz = params.get('nz', 100)
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 64)
        self.lr = params.get('lr', 2e-4)
        self.beta1 = params.get('beta1', 0.5)

        self.logger = None
        if Logger is not None:
            self.logger = Logger('GAN', Path(self.dataset_path).stem)

        processor = LightCurveProcessor(fourier=True, normalize=True, gaussian=True, standardize=True, gaussian_sigma=2)
        train_csv = os.path.join(self.dataset_path, 'exoTrain.csv')
        self.dataset = LightCurveDataset(train_csv, processor)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.netG = Generator(nz=self.nz).to(self.device)
        self.netD = Discriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.fixed_noise = torch.randn(64, self.nz, device=self.device)
        self.real_label = 1.
        self.fake_label = 0.

        os.makedirs(os.path.join(self.out_dir, 'images', Path(self.dataset_path).stem), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'models', Path(self.dataset_path).stem), exist_ok=True)

    def train(self):
        iters = 0
        for epoch in range(1, self.epochs + 1):
            for i, data in enumerate(self.dataloader, 0):
                real_cpu = data.to(self.device)  # (B, 1, 40, 40)
                b_size = real_cpu.size(0)

                self.netD.zero_grad()
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()

                noise = torch.randn(b_size, self.nz, device=self.device)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                self.optD.step()

                self.netG.zero_grad()
                label.fill_(self.real_label)  # want fake classified as real
                output = self.netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                self.optG.step()

                if self.logger is not None:
                    self.logger.log(errD.detach(), errG.detach(), epoch, i, len(self.dataloader))
                    if iters % 100 == 0:
                        with torch.no_grad():
                            fake_fixed = self.netG(self.fixed_noise).detach().cpu()
                        self.logger.log_images(fake_fixed, num_images=64, epoch=epoch, n_batch=i, num_batches=len(self.dataloader))

                iters += 1

            torch.save(self.netG.state_dict(), os.path.join(self.out_dir, 'models', Path(self.dataset_path).stem, f'G_epoch_{epoch}.pth'))
            torch.save(self.netD.state_dict(), os.path.join(self.out_dir, 'models', Path(self.dataset_path).stem, f'D_epoch_{epoch}.pth'))
            print(f'Finished epoch {epoch}/{self.epochs}')

    def dream(self, n_images=16, checkpoint=None, out_dir=None):
        """
        Generate images from a trained generator.
        If checkpoint is given, load the weights.
        """
        if checkpoint:
            self.netG.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.netG.eval()
        z = torch.randn(n_images, self.nz, device=self.device)
        with torch.no_grad():
            imgs = self.netG(z).cpu()  # shape (N,1,40,40)

        imgs = (imgs + 1.0) / 2.0
        out_dir = out_dir or os.path.join(self.out_dir, 'images', Path(self.dataset_path).stem)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(imgs.size(0)):
            img = imgs[i].squeeze().numpy()
            np.save(os.path.join(out_dir, f'dream_{i}.npy'), img)
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(out_dir, f'dream_{i}.png'), img, cmap='gray')

        print(f'Saved {n_images} dreamed images to {out_dir}')
