import torch
from torch.utils.data import DataLoader

from custom_datasets import Div2kSrDataset

import matplotlib.pyplot as plt

from models import model1

from global_config import *

from tqdm import tqdm

import os
from dotenv import load_dotenv

if __name__ == '__main__':
	# environment variables
	load_dotenv()
	data_root = os.getenv('DIV2K_ROOT_TRAIN')
	preload_images = os.getenv('PRELOAD_IMAGES') == 'True'
	preload_pool = int(os.getenv('PRELOAD_POOL'))
	batch_size = int(os.getenv('BATCH_SIZE'))
	epochs = int(os.getenv('EPOCHS'))
	load = os.getenv('LOAD') == 'True'
	save = os.getenv('SAVE') == 'True'

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# data
	dataset = Div2kSrDataset(
		root=data_root,
		high_size=model1.PATCH_SIZE * model1.SCALE_FACTOR * 4,
		low_size=model1.PATCH_SIZE * 4,
		preload_images=preload_images
	)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
	)

	# nets
	if load:
		gen = torch.load(os.path.join(MODEL_FILES, model1.GEN_FILE))
		disc = torch.load(os.path.join(MODEL_FILES, model1.DISC_FILE))
	else:
		gen = model1.Generator()
		disc = model1.Discriminator()

	gen = gen.to(device)
	disc = disc.to(device)

	# trainer - manages losses, optimizers and history
	trainer = model1.Trainer(gen, disc, device)

	# training loop
	gen.train()
	disc.train()
	for epoch in range(epochs):
		for high_res, low_res in tqdm(dataloader, desc=f'epoch {epoch}'):
			high_res = high_res.to(device)
			low_res = low_res.to(device)

			trainer.step(high_res, low_res)

	# save
	if save:
		torch.save(gen, os.path.join(MODEL_FILES, model1.GEN_FILE))
		torch.save(disc, os.path.join(MODEL_FILES, model1.DISC_FILE))

	# plot loss history
	plt.plot(trainer.gen_losses, label='generator')
	plt.plot(trainer.disc_losses, label='discriminator')
	plt.legend()
	plt.savefig(os.path.join(OUT, 'gen_disc_losses.png'))
