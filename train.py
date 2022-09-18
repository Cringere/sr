import torch
from torch.utils.data import DataLoader

from custom_datasets import Div2kSrDataset

import matplotlib.pyplot as plt

from generators import sr_gan as sr_gan_gen
from discriminators import sr_gan as sr_gan_disc
from trainers import gen_disc_trainer

from global_config import *

from tqdm import tqdm

import os
from dotenv import load_dotenv

gen_mod = sr_gan_gen
disc_mod = sr_gan_disc

if __name__ == '__main__':
	# environment variables
	load_dotenv()
	data_root = os.getenv('DIV2K_ROOT_TRAIN')
	preload_images = os.getenv('PRELOAD_IMAGES') == 'True'
	batch_size = int(os.getenv('BATCH_SIZE'))
	num_workers = int(os.getenv('NUM_WORKERS'))
	epochs = int(os.getenv('EPOCHS'))
	load = os.getenv('LOAD') == 'True'
	save = os.getenv('SAVE') == 'True'

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# data
	dataset = Div2kSrDataset(
		root=data_root,
		high_size=gen_mod.PATCH_SIZE * gen_mod.SCALE_FACTOR * 4,
		low_size=gen_mod.PATCH_SIZE * 4,
		preload_images=preload_images,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
	)

	# nets
	if load:
		gen = torch.load(os.path.join(MODEL_FILES, gen_mod.FILE))
		disc = torch.load(os.path.join(MODEL_FILES, disc_mod.FILE))
	else:
		gen = gen_mod.Generator()
		disc = disc_mod.Discriminator()

	gen = gen.to(device)
	disc = disc.to(device)

	# trainer - manages losses, optimizers and history
	trainer = gen_disc_trainer.Trainer(gen, disc, device)

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
		torch.save(gen, os.path.join(MODEL_FILES, gen_mod.FILE))
		torch.save(disc, os.path.join(MODEL_FILES, disc_mod.FILE))

	# plot loss history
	plt.clf()
	for label, losses in trainer.get_total_losses().items():
		plt.plot(losses, label=label)
	plt.legend()
	plt.savefig(os.path.join(OUT, 'total_losses.png'))

	# plot generator loss history
	plt.clf()
	for key, value in trainer.get_all_gen_losses().items():
		plt.plot(value, label=key)
	plt.legend()
	plt.savefig(os.path.join(OUT, 'gen_all_losses.png'))
