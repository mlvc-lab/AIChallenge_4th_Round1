import os

import PIL

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=test_transform):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        sample['image_name'] = img_name
        return sample
