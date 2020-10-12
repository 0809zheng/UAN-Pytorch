from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolderTest

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir,patch_size, upscale_factor, data_augmentation,
                             transform=transform())

def get_eval_set(data_dir, hr, upscale_factor):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolderEval(hr_dir, upscale_factor,
                             transform=transform())

def get_test_set(data_dir, lr, upscale_factor):
    lr_dir = join(data_dir, lr)
    return DatasetFromFolderTest(lr_dir, upscale_factor,
                             transform=transform())
