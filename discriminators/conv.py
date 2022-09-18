import torch.nn as nn

FILE = 'conv_discriminator.tch'

CHANNELS = 3

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.c = CHANNELS

		def conv(out_c, kernel_size, stride, padding, last=False):
			out =  nn.Sequential(
				nn.Conv2d(self.c, out_c, kernel_size, stride, padding),
				nn.Dropout(p=0.1) if not last else nn.Identity(),
				nn.LeakyReLU() if not last else nn.Identity(),
				nn.BatchNorm2d(out_c) if not last else nn.Identity(),
			)
			self.c = out_c
			return out
		
		def conv_same(out_c, kernel_size):
			assert kernel_size % 2 == 1
			return conv(out_c, kernel_size, 1, (kernel_size - 1) // 2)


		self.seq = nn.Sequential(
			conv_same(8, 5),
			conv_same(16, 5),
			nn.MaxPool2d(2),
			conv_same(16, 3),
			conv_same(16, 3),
			conv_same(32, 3),
			nn.MaxPool2d(2),
			conv_same(32, 3),
			conv(1, 2, 2, 0, last=True),
			nn.Sigmoid(),
		)
	
	def forward(self, x):
		return self.seq(x)
