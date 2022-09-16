import torch
import torch.nn as nn

from models import model1

def count_parameters(model, name):
	learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	n_params = "{:,}".format(learnable_parameters)
	print(f'{name} has {n_params} learnable parameters')

if __name__ == '__main__':
	gen = model1.Generator()
	disc = model1.Discriminator()

	count_parameters(gen, 'generator')
	count_parameters(disc, 'discriminator')

	print()

	x = torch.zeros((2, model1.CHANNELS, model1.PATCH_SIZE, model1.PATCH_SIZE))
	y = gen(x)
	d = disc(y)
	print(x.shape)
	print(y.shape)
	print(d.shape)
	print()

	x = torch.zeros((2, model1.CHANNELS, model1.PATCH_SIZE * 8, model1.PATCH_SIZE * 8))
	y = gen(x)
	d = disc(y)
	print(x.shape)
	print(y.shape)
	print(d.shape)
	print()
