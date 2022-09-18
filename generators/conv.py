import torch
import torch.nn as nn

FILE = 'conv_generator.tch'

CHANNELS = 3
PATCH_SIZE = 96 // 4
SCALE_FACTOR = 4

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
			assert kernel_size % 2 == 1
			return conv(out_c, kernel_size, 1, (kernel_size - 1) // 2)
		
		self.seq = nn.Sequential(
			conv_same(32, 3),
			conv_t(32, 2, 2, 0),
			conv_same(64, 3),
			conv_t(64, 2, 2, 0),
			conv_same(128, 3),
			conv(CHANNELS, 1, 1, 0, last=True),
			nn.Tanh(),
		)

	def forward(self, x):
		return self.seq(x)
