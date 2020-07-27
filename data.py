import os
from pathlib import Path
import random
import shutil
from zipfile import ZipFile

from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, ImageNet
from tqdm import tqdm

valid_datasets = [
    'cifar10', 'cifar100', 'imagenet', 'things'
]


def _verify_dataset(dataset):
    r"""verify your dataset.  
    If your dataset name is unknown dataset, raise error message..
    """
    if dataset not in valid_datasets:
        msg = "Unknown dataset \'{}\'. ".format(dataset)
        msg += "Valid datasets are {}.".format(", ".join(valid_datasets))
        raise ValueError(msg)
    return dataset


def cifar10_loader(batch_size, num_workers, datapath, cuda):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = CIFAR10(
        root=datapath, train=True, download=True,
        transform=transform_train)
    valset = CIFAR10(
        root=datapath, train=False, download=True,
        transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader


def cifar100_loader(batch_size, num_workers, datapath, cuda):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = CIFAR100(
        root=datapath, train=True, download=True,
        transform=transform_train)
    valset = CIFAR100(
        root=datapath, train=False, download=True,
        transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader


def imagenet_loader(batch_size, num_workers, datapath, cuda):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = ImageNet(
        root=datapath, split='train', download=False,
        transform=transform_train)
    valset = ImageNet(
        root=datapath, split='val', download=False,
        transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader


def things_loader(batch_size, num_workers, datapath, cuda):
    transform = transforms.Compose([
        transforms.RandomResizedCrop (256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize,
    ])

    trainset = ImageFolder(str(Path(datapath) / 'train'), transform=transform)
    valset = ImageFolder(str(Path(datapath) / 'val'), transform=transform_val)

    if cuda:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def things_unzip_and_convert(source, target):
    """
    Unzip Data.zip and convert to ImageFolder loadable datset 
    """
    source = Path(source)
    temp = Path('/tmp/things')
    target = Path(target)

    # if target.exists():
    #     rm_tree(target)
    if not target.exists():
        target.mkdir()

    for zipname in source.glob("*.zip"):
        print(f"zipfile: {zipname}")
        if zipname.name in ['things_v1.zip', '06_상품.zip']:
            continue
        with ZipFile(zipname) as z:
            i = 0
            for fname in tqdm(z.namelist()):
                # print(Path(fname), Path(fname).suffix)

                if not Path(fname).suffix in ['.JPG', '.jpg' ]:
                    continue

                if i % 100 == 0:
                    label = str(Path(fname).parent.name).split('_')[-2]
            
                    # check and make label dir
                    label_dir = target / label
                    if not label_dir.exists():
                        label_dir.mkdir()
                    label_dir = target / label
                    if not label_dir.exists():
                        label_dir.mkdir()

                    # extract file
                    z.extract(fname, temp)

                    # resize
                    img = Image.open(temp/fname)
                    img.resize((480, 360))
                    img.save(Path(target)/label/(Path(fname).stem + '.jpg'))

                    # delete file
                    # (temp / fname).unlink()
                i+=1
    rm_tree(temp)


def data_split(source, target):
    # constant
    train = 'train'
    val = 'val'

    # prepare
    source = Path(source)
    target = Path(target)
    if not source.exists():
        raise FileNotFoundError
    if not target.exists():
        target.mkdir()

    train_path = target / train
    val_path = target / val
    if not train_path.exists():
        train_path.mkdir()
    if not val_path.exists():
        val_path.mkdir()

    for classname in tqdm(source.iterdir()):
        if not (train_path / classname.name).exists():
            (train_path / classname.name).mkdir()
        if not (val_path / classname.name).exists():
            (val_path / classname.name).mkdir()

        # split
        for instance in tqdm(classname.iterdir()):
            if random.random() > 0.5:
                # print(str(instance), str(train_path/classname.name/instance.name))
                shutil.copy(str(instance), str(train_path/classname.name/instance.name))
            else:
                shutil.copy(str(instance), str(val_path/classname.name/instance.name))
                # print(str(instance), str(val_path/classname.name/instance.name))


def DataLoader(batch_size, num_workers, dataset='things', datapath='/dataset/things_v1', cuda=True):
    r"""Dataloader for training/validation
    """
    DataSet = _verify_dataset(dataset)
    if DataSet == 'cifar10':
        return cifar10_loader(batch_size, num_workers, datapath, cuda)
    elif DataSet == 'cifar100':
        return cifar100_loader(batch_size, num_workers, datapath, cuda)
    elif DataSet == 'imagenet':
        return imagenet_loader(batch_size, num_workers, datapath, cuda)
    elif DataSet == 'things':
        return things_loader(batch_size, num_workers, datapath, cuda)


if __name__ == "__main__":
    things_unzip_and_convert('F:/src', 'F:/dst')
    # data_split('/home/kairos/Downloads/things', '/home/kairos/Downloads/things_split')