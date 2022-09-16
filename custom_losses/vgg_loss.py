import torch.nn as nn

from torchvision import models

class VggSingleLoss(nn.Module):
	def __init__(self, model='vgg16', layer=8):
		super().__init__()

		# load model
		self.vgg = None
		if model == 'vgg11':
			self.vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

		if model == 'vgg16':
			self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

		if model == 'vgg19':
			self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

		assert self.vgg is not None

		# disable training
		self.vgg.eval()
		self.vgg.requires_grad_(False)

		# extract layers
		self.vgg = self.vgg.features[:layer]
		
		# underlying loss
		self.mse = nn.MSELoss()

	def forward(self, inputs, targets):
		return self.mse(
			self.vgg(inputs),
			self.vgg(targets),
		)
