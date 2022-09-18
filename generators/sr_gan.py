'''
Taken from the paper: https://arxiv.org/abs/1609.04802
'''

import torch.nn as nn

PATCH_SIZE = 24

CHANNELS = 3
HIDDEN_CHANNELS = 64

RES_BLOCKS = 16

UPSAMPLE_BLOCK_FACTOR = 2
UPSAMPLE_BLOCKS = 2

SCALE_FACTOR = UPSAMPLE_BLOCK_FACTOR * UPSAMPLE_BLOCKS

FILE = 'sr_gan_gen.tch'

class ResBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(channels, channels, 3, 1, padding='same'),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
			nn.Conv2d(channels, channels, 3, 1, padding='same'),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
		)
	
	def forward(self, x):
		return x + self.seq(x)

def upsample_block(channels):
	return nn.Sequential(
		nn.Conv2d(channels, channels * (UPSAMPLE_BLOCK_FACTOR ** 2), 3, 1, padding='same'),
		nn.PixelShuffle(UPSAMPLE_BLOCK_FACTOR),
		nn.PReLU()
	)

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.pre = nn.Sequential(
			nn.Conv2d(CHANNELS, HIDDEN_CHANNELS, 9, 1, padding='same'),
			nn.PReLU()
		)

		res_blocks = [ResBlock(HIDDEN_CHANNELS) for _ in range(RES_BLOCKS)]
		self.res_blocks = nn.Sequential(*res_blocks)

		self.post_res_blocks = nn.Sequential(
			nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, 1, padding='same'),
			nn.BatchNorm2d(HIDDEN_CHANNELS),
		)

		up_sample_blocks = [upsample_block(HIDDEN_CHANNELS) for _ in range(UPSAMPLE_BLOCKS)]
		self.up_sample = nn.Sequential(*up_sample_blocks)

		self.squeeze = nn.Conv2d(HIDDEN_CHANNELS, CHANNELS, 9, 1, padding='same')

	def forward(self, x):
		x = self.pre(x)
		
		x = self.post_res_blocks(self.res_blocks(x)) + x
	
		x = self.up_sample(x)

		x = self.squeeze(x)

		return x
