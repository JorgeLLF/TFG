
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype


class ImageTransformationsApplier(Dataset):

    def __init__(self, dataset, transformations=Compose([Resize((227, 227)), ToImage(),
                                                           ToDtype(torch.float32, scale=True)])):
        self.dataset = dataset
        self.transformations = transformations

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transformations(image), label

    def __len__(self):
        return len(self.dataset)