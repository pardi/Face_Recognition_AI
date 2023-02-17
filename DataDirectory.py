from glob import glob
from typing import Tuple, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


def get_transforms(size: Tuple[int, int]) -> Callable:
    """ Transforms to apply to the image."""
    transforms = [Resize(size=size),
                  ToTensor(),
                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return Compose(transforms)


class DatasetDirectory(Dataset):
    """
    A custom dataset class that loads images from folder.
    args:
    - directory: location of the images
    - transform: transform function to apply to the images
    - extension: file format
    """

    def __init__(self,
                 directory: str,
                 transforms: Callable = None,
                 extension: str = '.jpg'):

        self.transforms = transforms

        self.imgs_dataset = glob(directory + "*" + extension)

    def __len__(self) -> int:
        """ returns the number of items in the dataset """
        return len(self.imgs_dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """ load an image and apply transformation """

        return self.transforms(Image.open(self.imgs_dataset[index]))
