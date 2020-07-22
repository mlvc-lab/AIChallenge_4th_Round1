import pathlib
import argparse
import requests
from tqdm import tqdm

import models as models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024 * 32

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc=destination, miniters=0, unit='MB', unit_scale=1/32, unit_divisor=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download pretrained checkpoints')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet)')
    parser.add_argument('--layers', default=56, type=int, metavar='N',
                        help='number of layers in VGG/ResNet/WideResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('-o', '--out', type=str, default='pretrained_ckpt.pth',
                        help='output filename of pretrained model from our google drive')
    args = parser.parse_args()

    
    if args.arch == 'resnet':
        pass
    elif args.arch == ''
    else:
        print('Not prepared yet..\nProgram exit...')
        exit()


    arch_name = args.arch
    if args.arch in ['resnet']:
        arch_name += str(args.layers)

    ckpt_dir = pathlib.Path('checkpoint')
    dir_path = ckpt_dir / arch_name / "imagenet"
    dir_path.mkdir(parents=True, exist_ok=True)

    destination = dir_path / args.out
    download_file_from_google_drive(file_id, destination.as_posix())
