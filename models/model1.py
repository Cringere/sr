import torch
import torch.nn as nn
import torch.optim as optim

from custom_losses.vgg_loss import VggSingleLoss

from utils import negative_labels, positive_labels

CHANNELS = 3
PATCH_SIZE = 96 // 4
SCALE_FACTOR = 4

GEN_LR = 5e-4
DISC_LR = 1e-4

GEN_FILE = 'model1_gen.tch'
DISC_FILE = 'model1_disc.tch'

LAMBDA_DISC = 1
LAMBDA_PIXEL = 0.1
LAMBDA_VGG = 1

LABEL_NOISE = 0.1

VGG_LOSS_MODEL = 'vgg16'
VGG_LOSS_LAYER = 12

class Trainer:
	def __init__(self, gen, disc, device):
		# nets
		self.gen = gen
		self.disc = disc

		# loss and optimizer
		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()
		self.vgg_loss = VggSingleLoss(layer=16).to(device)
		self.gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
		self.disc_optimizer = optim.Adam(disc.parameters(), lr=DISC_LR)

		# history
		self.gen_losses = []
		self.disc_losses = []

	def step(self, real_high_res, low_res):
		# create a fake high res image
		fake_high_res = self.gen(low_res)

		# discriminator loss
		disc_real_high_res = self.disc(real_high_res)
		disc_fake_high_res = self.disc(fake_high_res.detach())
		disc_loss = (
			self.bce_loss(disc_fake_high_res, negative_labels(disc_fake_high_res, LABEL_NOISE)) +
			self.bce_loss(disc_real_high_res, positive_labels(disc_real_high_res, LABEL_NOISE))
		) / 2

		# discriminator step
		self.disc_optimizer.zero_grad()
		disc_loss.backward()
		self.disc_optimizer.step()

		# generator - discriminator loss
		disc_fake_high_res = self.disc(fake_high_res)
		gen_disc_loss = self.bce_loss(disc_fake_high_res, positive_labels(disc_fake_high_res, LABEL_NOISE))

		# generator - pixel loss
		gen_pixel_loss = self.mse_loss(fake_high_res, real_high_res)

		# generator - VGG loss - TODO
		gen_vgg_loss = self.vgg_loss(fake_high_res, real_high_res)

		# generator - total loss
		gen_loss = (
			LAMBDA_DISC * gen_disc_loss +
			LAMBDA_PIXEL * gen_pixel_loss +
			LAMBDA_VGG * gen_vgg_loss
		) / (LAMBDA_DISC + LAMBDA_PIXEL + LAMBDA_VGG)

		# generator step
		self.gen_optimizer.zero_grad()
		gen_loss.backward()
		self.gen_optimizer.step()

		# history
		self.gen_losses.append(gen_loss.item())
		self.disc_losses.append(disc_loss.item())


class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.c = CHANNELS

		def conv_t(out_c, kernel_size, stride, padding, last=False):
			out =  nn.Sequential(
				nn.ConvTranspose2d(self.c, out_c, kernel_size, stride, padding),
				nn.ReLU() if not last else nn.Identity(),
				nn.BatchNorm2d(out_c) if not last else nn.Identity(),
			)
			self.c = out_c
			return out
		
		def conv(out_c, kernel_size, stride, padding, last=False):
			out =  nn.Sequential(
				nn.Conv2d(self.c, out_c, kernel_size, stride, padding),
				nn.ReLU() if not last else nn.Identity(),
				nn.BatchNorm2d(out_c) if not last else nn.Identity(),
			)
			self.c = out_c
			return out
		
		def conv_same(out_c, kernel_size):
			if kernel_size == 3:
				return conv(out_c, 3, 1, 1)
			if kernel_size == 5:
				return conv(out_c, 5, 1, 2)
			raise Exception(f'invalid kernel size: {kernel_size}')
		
		self.seq = nn.Sequential(
			conv(16, 1, 1, 0),
			conv_t(32, 2, 2, 0),
			conv_same(32, 5),
			conv_t(64, 1, 1, 0),
			# conv_t(128, 2, 2, 0),
			nn.ConvTranspose2d(
				in_channels=64,
				out_channels=64,
				kernel_size=4,
				stride=2,
				padding=4,
				dilation=3,
				output_padding=0,
			),
			nn.ReLU(),
			conv_same(128, 5),
			conv_t(CHANNELS, 1, 1, 0, last=True),
			nn.Tanh(),
		)

	def forward(self, x):
		return self.seq(x)


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.c = CHANNELS

		def conv(out_c, kernel_size, stride, padding, last=False):
			out =  nn.Sequential(
				nn.Conv2d(self.c, out_c, kernel_size, stride, padding),
				nn.ReLU() if not last else nn.Identity(),
				nn.BatchNorm2d(out_c) if not last else nn.Identity(),
			)
			self.c = out_c
			return out

		self.seq = nn.Sequential(
			conv(4, 3, 1, 0),
			conv(4, 3, 1, 0),
			nn.MaxPool2d(2),
			conv(8, 3, 1, 0),
			conv(8, 3, 1, 0),
			nn.MaxPool2d(2),
			conv(8, 3, 1, 0),
			conv(8, 3, 1, 0),
			conv(1, 2, 2, 0, last=True),
			nn.Sigmoid(),
		)
	
	def forward(self, x):
		return self.seq(x)
