import torch
import torch.nn as nn
import torch.optim as optim

from custom_losses.vgg_loss import VggSingleLoss

from utils import negative_labels, positive_labels

GEN_LR = 1e-4

LAMBDA_PIXEL = 1
LAMBDA_VGG = 0.06

LABEL_NOISE = 0.1

VGG_LOSS_MODEL = 'vgg16'
VGG_LOSS_LAYER = 20

GEN_DISC_RATIO_THRESHOLD = 1.5

class Trainer:
	def __init__(self, gen, device):
		# nets
		self.gen = gen

		# loss and optimizer
		self.mse_loss = nn.MSELoss()
		self.vgg_loss = VggSingleLoss(VGG_LOSS_MODEL, VGG_LOSS_LAYER).to(device)
		self.gen_optimizer = optim.Adam(gen.parameters(), lr=GEN_LR)

		# history
		self.pixel_losses = []
		self.vgg_losses = []

	def step(self, real_high_res, low_res):
		# create a fake high res image
		fake_high_res = self.gen(low_res)

		# generator - pixel loss
		gen_pixel_loss = self.mse_loss(fake_high_res, real_high_res)

		# generator - VGG loss
		gen_vgg_loss = self.vgg_loss(fake_high_res, real_high_res)

		# generator - total loss
		gen_loss = (
			LAMBDA_PIXEL * gen_pixel_loss +
			LAMBDA_VGG * gen_vgg_loss
		)

		# history
		self.pixel_losses.append(gen_pixel_loss.item())
		self.vgg_losses.append(gen_vgg_loss.item())

		# generator step
		self.gen_optimizer.zero_grad()
		gen_loss.backward()
		self.gen_optimizer.step()

	def get_all_gen_losses(self):
		return {
			'pixel': self.pixel_losses,
			'vgg': self.vgg_losses,
		}
