import torch
import torch.nn as nn

FILE = 'conv_generator.tch'

CHANNELS = 3
PATCH_SIZE = 96 // 4
SCALE_FACTOR = 4

class ConvResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, initial_bn=True):
		super().__init__()
		assert kernel_size % 2 == 1
		padding = (kernel_size - 1) // 2

		self.initial = nn.BatchNorm2d(in_channels) if initial_bn else nn.Identity()

		self.seq = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
			nn.LeakyReLU(),
			nn.BatchNorm2d(out_channels),
			nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
			nn.LeakyReLU(),
		)

		if in_channels != out_channels:
			self.c = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
		else:
			self.c = nn.Identity()
	
	def forward(self, x):
		x = self.initial(x)
		return self.seq(x) + self.c(x)

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
		
		def conv_res(out_c, kernel_size):
			out = ConvResBlock(self.c, out_c, kernel_size)
			self.c = out_c
			return out
		
		self.seq = nn.Sequential(
			conv_res(8, 3),
			conv_res(8, 3),
			conv_res(16, 3),
			conv_t(32, 2, 2, 0),
			conv_res(32, 3),
			conv_res(32, 3),
			conv_t(64, 2, 2, 0),
			conv_res(64, 3),
			conv_res(64, 3),
			conv(CHANNELS, 1, 1, 0, last=True),
			nn.Tanh(),
		)

	def forward(self, x):
		return self.seq(x)
