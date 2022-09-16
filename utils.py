from cProfile import label
import torch

def negative_labels(like, noise=0):
	if noise > 0:
		return torch.rand_like(like) * noise
	else:
		return torch.zeros_like(like)


def positive_labels(like, noise=0):
	return 1 - negative_labels(like, noise)
