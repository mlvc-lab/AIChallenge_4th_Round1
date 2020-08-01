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

from augmentation import RandAugment

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


def cifar10_loader(batch_size, num_workers, datapath, image_size=32, cuda=False):
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


def cifar100_loader(batch_size, num_workers, datapath, image_size=32, cuda=False):
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


def imagenet_loader(batch_size, num_workers, datapath, image_size=224, cuda=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
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


def things_loader(batch_size, num_workers, datapath, image_size=224, cuda=False):
    if datapath in ["/dataset/things_v3_1", "/dataset/things_v3"]:
        normalize = transforms.Normalize(mean=[0.5919, 0.5151, 0.4966],
                                        std=[0.2087, 0.1992, 0.1988])
    elif datapath in ["/dataset/things_v4", "/dataset/things_v5"]:
        normalize = transforms.Normalize(mean=[0.6125, 0.8662, 0.9026],
                                        std=[1.0819, 1.1660, 1.1882])
    elif datapath in ["/dataset/things_v5xxxxxx"]:
        normalize = transforms.Normalize(mean=[0.7473, 0.7185, 0.7083],
                                        std=[0.2307, 0.2395, 0.2436])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),         #crop을 한다음 Resize
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Add RandAugment with N, M(hyperparameter)
    transform_train.transforms.insert(0, transforms.RandomApply([RandAugment(2, 4)], p=0.1))

    trainset = ImageFolder(str(Path(datapath) / 'train'), transform=transform_train)
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

                if i % 300 == 0:
                    label = str(Path(fname).parent.name).split('_')[-3]
            
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
                    (temp / fname).unlink()
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


def get_params(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.

    #모든 픽셀에 대해서 (H x W) 평균을 낸 다음에 이에 대해서 다 평균을 낸다.
    for (data, _) in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(f"mean : {mean} , std : {std}")

    return mean, std


def DataLoader(batch_size, num_workers, dataset='cifar10', datapath='../data', image_size=224, cuda=True, **kwargs):
    r"""Dataloader for training/validation
    """
    DataSet = _verify_dataset(dataset)
    if DataSet == 'cifar10':
        return cifar10_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'cifar100':
        return cifar100_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'imagenet':
        return imagenet_loader(batch_size, num_workers, datapath, image_size, cuda)
    elif DataSet == 'things':
        return things_loader(batch_size, num_workers, datapath, image_size, cuda)


if __name__ == "__main__":
    things_unzip_and_convert('/media/kairos/KM_SDRnHDR/AI-Challenge_dataset', '/home/kairos/Downloads/target')
