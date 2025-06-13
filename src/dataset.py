import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

mnist_transform = transforms.Compose([
  transforms.ToPILImage(),
  #transforms.Grayscale(),
  transforms.RGB(),
  transforms.Resize((64,64)),
  transforms.ToImage(),
  transforms.ToDtype(torch.float32, scale=True),
])

class SokobanDataset(Dataset):
  def __init__(self, data_folder: str, grayscale: bool = False):
    self._data_folder = data_folder
    self._transform = self._get_transform(grayscale=grayscale)
    self._image_paths = self._load_image_paths()

  def _get_transform(self, grayscale: bool = False):
    transforms_list: list = [transforms.ToPILImage()]
    if grayscale:
      transforms_list.append(transforms.Grayscale())
    transforms_list.extend([
      transforms.Resize((64, 64)),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
    ])
    return transforms.Compose(transforms_list)    

  def _load_image_paths(self):
    import os
    return [os.path.join(self._data_folder, f) for f in os.listdir(self._data_folder) if f.endswith('.png')]

  def __len__(self):
    return len(self._image_paths)

  def __getitem__(self, idx):
    from PIL import Image
    image = Image.open(self._image_paths[idx]).convert('RGB')
    image = self._transform(image)
    return image
