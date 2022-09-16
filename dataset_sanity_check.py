from custom_datasets import Div2kSrDataset

import os
from dotenv import load_dotenv

from torch.utils.data import DataLoader

if __name__ == '__main__':
	load_dotenv()

	dataset = Div2kSrDataset(
		root=os.getenv('DIV2K_ROOT_VAL'),
		high_size=256,
		low_size=256//4
	)

	loader = DataLoader(
		dataset,
		batch_size=16,
		shuffle=True
	)

	for high_res, low_res in loader:
		print(high_res.shape)
		print(low_res.shape)
		break
