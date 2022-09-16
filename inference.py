import torch
from torch.utils.data import DataLoader

from custom_datasets import Div2kSrDataset

from models import model1

from global_config import *
		
from torchvision.utils import save_image

import os
from dotenv import load_dotenv

if __name__ == '__main__':
	# environment variables
	load_dotenv()
	data_root = os.getenv('DIV2K_ROOT_TRAIN')

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# data
	dataset = Div2kSrDataset(
		root=data_root,
		high_size=512,
		low_size=512 // model1.SCALE_FACTOR,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=True,
	)

	# nets
	gen = torch.load(os.path.join(MODEL_FILES, model1.GEN_FILE))
	disc = torch.load(os.path.join(MODEL_FILES, model1.DISC_FILE))
	gen = gen.to(device)
	disc = disc.to(device)

	# inference
	gen.eval()
	with torch.no_grad():
		for high_res, low_res in dataloader:
			# load and pass images
			high_res = high_res.cpu()
			low_res = low_res.cpu()
			fake_high_res = gen(low_res.to(device)).cpu()

			# un-normalize
			high_res = Div2kSrDataset.unnormalize(high_res)
			low_res = Div2kSrDataset.unnormalize(low_res)
			fake_high_res = Div2kSrDataset.unnormalize(fake_high_res)

			# save image
			save_image(high_res, os.path.join(OUT, 'high_res.png'))
			save_image(low_res, os.path.join(OUT, 'low_res.png'))
			save_image(fake_high_res, os.path.join(OUT, 'fake_high_res.png'))
			
			break
