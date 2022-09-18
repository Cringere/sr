import torch.nn as nn

from torchvision import models

class _VggBaseLoss(nn.Module):
	def __init__(self, model_name, disable_in_place=False):
		super().__init__()
		
		# load model and disable training
		vgg = None
		if model_name == 'vgg11':
			vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

		if model_name == 'vgg16':
			vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

		if model_name == 'vgg19':
			vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
		
		if vgg is None:
			raise ValueError(f'invalid model name {model_name}')

		#disable training and extract features
		vgg.eval()
		vgg.requires_grad_(False)
		self.features = vgg.features

		# replace in-place ReLUs with non in-place ReLUs
		if disable_in_place:
			for i in range(len(self.features)):
				if isinstance(self.features[i], nn.ReLU):
					self.features[i] = nn.ReLU()

		# underlying loss
		self.mse = nn.MSELoss()

class VggSingleLoss(_VggBaseLoss):
	def __init__(self, model_name='vgg16', layer=None):
		super().__init__(model_name)
		self.extracted = self.features if layer is None else self.features[: layer + 1]

	def forward(self, inputs, targets):
		return self.mse(
			self.extracted(inputs),
			self.extracted(targets),
		)

class VggGroupLoss(_VggBaseLoss):
	def __init__(self, model_name='vgg16', from_layer=8, to_layer=12):
		super().__init__(model_name, disable_in_place=True)
		
		from_layer = 0 if from_layer is None else from_layer
		to_layer = len(self.features) - 1 if to_layer is None else to_layer
		
		self.pre = self.features[: from_layer] if from_layer > 0 else None

		self.extracted = self.features[from_layer: to_layer + 1]

	def forward(self, inputs, targets):
		if self.pre is not None:
			inputs = self.pre(inputs)
			targets = self.pre(targets)
	
		total_error = 0
		for layer in self.extracted:
			inputs = layer(inputs)
			targets = layer(targets)
			total_error += self.mse(inputs, targets)
		return total_error / len(self.extracted)
