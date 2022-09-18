import torch

from generators.sr_gan import Generator, CHANNELS, PATCH_SIZE
from discriminators.sr_gan import Discriminator

def count_parameters(model, name):
	learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	n_params = "{:,}".format(learnable_parameters)
	print(f'{name} has {n_params} learnable parameters')

def check_generator(name, gen):
	print(name)
	x = torch.zeros((2, CHANNELS, PATCH_SIZE, PATCH_SIZE))
	print(f'x: {x.shape}')
	y = gen(x)
	print(f'y: {y.shape}')

if __name__ == '__main__':
	gen = Generator()
	disc = Discriminator()

	count_parameters(gen, 'generator')
	count_parameters(disc, 'discriminator')

	print()

	x = torch.zeros((2, CHANNELS, PATCH_SIZE, PATCH_SIZE))
	y = gen(x)
	d = disc(y)
	print(x.shape)
	print(y.shape)
	print(d.shape)
	print()

	x = torch.zeros((2, CHANNELS, PATCH_SIZE * 8, PATCH_SIZE * 8))
	y = gen(x)
	d = disc(y)
	print(x.shape)
	print(y.shape)
	print(d.shape)
	print()
