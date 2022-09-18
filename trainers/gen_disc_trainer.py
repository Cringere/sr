import torch
import torch.nn as nn
import torch.optim as optim

from custom_losses.vgg_loss import VggSingleLoss

from utils import negative_labels, positive_labels

GEN_LR = 1e-4
DISC_LR = 1e-4

LAMBDA_DISC = 0.1
LAMBDA_PIXEL = 1.0
LAMBDA_VGG = 0.1

LABEL_NOISE = 0.1

VGG_LOSS_MODEL = 'vgg16'
VGG_LOSS_LAYER = 20

GEN_DISC_RATIO_THRESHOLD = 1.5

class Trainer:
	def __init__(self, gen, disc, device):
		# nets
		self.gen = gen
		self.disc = disc

		# loss and optimizer
		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()
		self.vgg_loss = VggSingleLoss(VGG_LOSS_MODEL, VGG_LOSS_LAYER).to(device)
		self.gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)
		self.disc_optimizer = optim.Adam(disc.parameters(), lr=DISC_LR)

		# history
		self.gen_losses = []
		self.disc_losses = []
		self.all_gen_losses = {
			'total': [],
			'gen-disc': [],
			'pixel': [],
			'vgg': []
		}

	def step(self, real_high_res, low_res):
		train_disc = True
		train_gen = True
		if len(self.gen_losses) > 0:
			disc_loss = self.disc_losses[-1]
			gen_loss = self.gen_losses[-1]
			if GEN_DISC_RATIO_THRESHOLD is not None:
				if disc_loss / gen_loss > GEN_DISC_RATIO_THRESHOLD:
					train_gen = False
				if gen_loss / disc_loss > GEN_DISC_RATIO_THRESHOLD:
					train_disc = False

		# create a fake high res image
		fake_high_res = self.gen(low_res)

		if train_disc:
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

			# extract loss
			disc_loss = disc_loss.item()

		if train_gen:
			# generator - discriminator loss
			disc_fake_high_res = self.disc(fake_high_res)
			gen_disc_loss = self.bce_loss(disc_fake_high_res, positive_labels(disc_fake_high_res, LABEL_NOISE))

			# generator - pixel loss
			gen_pixel_loss = self.mse_loss(fake_high_res, real_high_res)

			# generator - VGG loss
			gen_vgg_loss = self.vgg_loss(fake_high_res, real_high_res)

			# generator - total loss
			gen_loss = (
				LAMBDA_DISC * gen_disc_loss +
				LAMBDA_PIXEL * gen_pixel_loss +
				LAMBDA_VGG * gen_vgg_loss
			)

			# history
			self.all_gen_losses['total'].append(gen_loss.item())
			self.all_gen_losses['gen-disc'].append(gen_disc_loss.item())
			self.all_gen_losses['pixel'].append(gen_pixel_loss.item())
			self.all_gen_losses['vgg'].append(gen_vgg_loss.item())

			# generator step
			self.gen_optimizer.zero_grad()
			gen_loss.backward()
			self.gen_optimizer.step()

			# extract loss
			gen_loss = gen_loss.item()

		# history
		self.gen_losses.append(gen_loss)
		self.disc_losses.append(disc_loss)

	def get_total_losses(self):
		return {
			'generator': self.gen_losses,
			'discriminator': self.disc_losses,
		}
	
	def get_all_gen_losses(self):
		return self.all_gen_losses
