import time
import pathlib
from os.path import isfile

import PIL
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models
import config
from utils import *

import nsml

# for ignore ImageNet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def bind_model(model):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = torch.load(filename)
        model.load_state_dict(state, strict=True)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        torch.save(model.state_dict(), filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
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


def inference(model, test_path: str) -> pd.DataFrame:
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data',
                                       transform=test_transform)

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    filename_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            x = x.to(device)
            _, pred = model(x)

            filename_list += data['image_name']
            y_pred += pred.detach().cpu().tolist()

    ret = pd.DataFrame({'file_name': filename_list, 'y_pred': y_pred})
    return ret


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(weight_file)
        print('load weight from {}.'.format(weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')


def main():
    # set arguments
    args = config.config()

    model = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                       width_mult=args.width_mult)
    
    # set multi-gpu
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])

    # load a checkpoint
    ckpt_file = pathlib.Path('checkpoint') / arch_name / args.dataset / args.load
    load_weight(model, ckpt_file)

    # bind the loaded model
    bind_model(model)
    nsml.save('test')


if __name__ == '__main__':
    main()
