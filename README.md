## SR
This project explores various super-resolution networks.

## Environment
Some files require a `.env` file to be present, here is an example of its format:
```
# pre-trained models
TORCH_HOME=~/torch-home

# data location
DIV2K_ROOT_TRAIN=~/datasets/div2k/DIV2K_train_HR
DIV2K_ROOT_VAL=~/datasets/div2k/DIV2K_valid_HR

# data loading - loading the entire dataset to the RAM
PRELOAD_IMAGES=True
PRELOAD_POOL=100

# training
BATCH_SIZE=32
EPOCHS=20

# loading and saving models
LOAD=False
SAVE=True
```
