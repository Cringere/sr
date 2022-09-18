'''
Taken from the paper: https://arxiv.org/abs/1609.04802
(with some changes)
'''

import torch.nn as nn

CHANNELS = 3
HIDDEN_CHANNELS = 64

FILE = 'sr_gan_disc.tch'

def block(in_channels, out_channels, stride):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, stride),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU()
	)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.pre = nn.Sequential(
			nn.Conv2d(CHANNELS, HIDDEN_CHANNELS, 3, 1, 'same'),
			nn.LeakyReLU(),
		)

		self.blocks = nn.Sequential(
			block(HIDDEN_CHANNELS * 1, HIDDEN_CHANNELS * 1, stride=2),

			block(HIDDEN_CHANNELS * 1, HIDDEN_CHANNELS * 2, stride=1),
			block(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS * 2, stride=2),
			
			block(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS * 4, stride=1),
			block(HIDDEN_CHANNELS * 4, HIDDEN_CHANNELS * 4, stride=2),

			block(HIDDEN_CHANNELS * 4, HIDDEN_CHANNELS * 8, stride=1),
			block(HIDDEN_CHANNELS * 8, HIDDEN_CHANNELS * 8, stride=2),
		)

		self.post = nn.Sequential(
			nn.AdaptiveAvgPool2d((6, 6)),
			nn.Flatten(),
			nn.Linear(512 * 6 * 6, 1024),
			nn.LeakyReLU(),
			nn.Linear(1024, 1),
			nn.Sigmoid(),
		)
	
	def forward(self, x):
		x = self.pre(x)
		x = self.blocks(x)
		x = self.post(x)
		return x
