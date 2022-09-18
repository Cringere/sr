import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image

import os

import ray
ray.init()

@ray.remote
def load_and_process(path):
	'''
	Loads an image, converts it to a tensor and returns
	'''
	img = Image.open(path)
	img = F.to_tensor(img)
	return img

class Div2kSrDataset(Dataset):
	def __init__(
			self,
			root,
			high_size,
			low_size,
			normalize=True,
			preload_images=False):
		super().__init__()
		self.root = root

		# list files
		self.files = os.listdir(root)

		# prepare transforms
		self.to_tensor = T.ToTensor()
		self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None
		self.crop = T.RandomCrop(high_size)
		self.random_flip = T.Compose([
			T.RandomVerticalFlip(),
			T.RandomHorizontalFlip(),
		])
		self.to_lower = T.Resize(low_size)

		# preload images
		self.images = None
		if preload_images:			
			paths = [os.path.join(self.root, file_path) for file_path in self.files]
			images = [load_and_process.remote(p) for p in paths]
			self.images = ray.get(images)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		'''
		returns a tuple of (high res image, low res image)
		'''
		# load image
		if self.images is None:
			image_path = os.path.join(self.root, self.files[idx])
			img = Image.open(image_path)
			img = self.to_tensor(img)
		else:
			img = self.images[idx]

		# transform the image
		img = self.crop(img)
		img = self.random_flip(img)
		if self.normalize is not None:
			img = self.normalize(img)

		# extract the high res and low res images
		high_res = img
		low_res = self.to_lower(high_res)

		return high_res, low_res

	@classmethod
	def unnormalize(cls, img):
		return img * 0.5 + 0.5