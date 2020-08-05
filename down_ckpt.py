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
                        help='number of layers in ResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    args = parser.parse_args()

    resnet_name = ['resnet18.pth', 'resnet34.pth', 'resnet50.pth', 'resnet101.pth']
    efficient_name = ['efficientnet-b0.pth', 'efficientnet-b1.pth','efficientnet-b2.pth',
                        'efficientnet-b3.pth','efficientnet-b4.pth','efficientnet-b5.pth',
                        'efficientnet-b6.pth','efficientnet-b7.pth']
    rexnet_name = ['rexnet-1.0.pth', 'rexnet-1.3.pth', 'rexnet-1.5.pth', 'rexnet-2.0.pth']
    mobilenetv2_name = ['mobilenetv2.pth']    

    if args.arch == 'resnet':
        print("download resnet (18, 34, 50, 101)")
        file_ids = []
        file_ids.append("1z-CDKeQ-09yp_rdFOn-eGJADgnOnce_O")
        file_ids.append("1GdziQGEjWWPv009pWZOV9j153Or9sVko")
        file_ids.append("1Rka3os7W-JU8FqjqPp6z_XtQztX8O05q")
        file_ids.append("157XjwYnWLWPOW33PY2yaiFA7JwDQ4CqJ")

    elif args.arch == 'efficientnet':
        print("download efficientnet (b0, b1, b2, b3, b4, b5, b6, b7)")
        file_ids = []
        file_ids.append("1BX8dXwhbb6uOFOimApBmwjSEVlOpoeit")
        file_ids.append("15HEyaF7azgzz7REhROduXWT-N4T7qZAq")
        file_ids.append("1MMHgl7Jxa9Eld65FruPHlCuah_96E2qo")
        file_ids.append("12AKBsSBC91T7P7WTbSPN31VhlBHhK0r7")
        file_ids.append("1GMplu69P3IpaSCv2r23ysxRgfVglzLdR")
        file_ids.append("1ThI_FQ9Cdhd4VWarYc_j2iasJ7R3GIqw")
        file_ids.append("1Dc2JOMiraPTitWEO5w21ab9LzNknJpeY")
        file_ids.append("1fubwDcTL0D8Zv8e7Hcl3uYGmAyFbKFRr")

    elif args.arch == "mobilenetv2":
        file_ids = []
        file_ids.append("1c145idQufGnzi1pdG1qY6ba6ZYvR_qo1")

    elif args.arch == "rexnet":
        file_ids = []
        file_ids.append("1PItzouHH8E2nEmSt2plHHhH_Nh49o3gf")
        file_ids.append("1AdATP9at1I2mS2tXOq4vHMuQHZMClytL")
        file_ids.append("1s0joqNOhEHDJ8WydKieP9boWs-Knez0D")
        file_ids.append("1ZAIVRxPZyH7YtCE73WlC_8J6NQD5pZsh")
    else:
        print('Not prepared yet..\nProgram exit...')
        exit()


    arch_name = args.arch

    ckpt_dir = pathlib.Path('checkpoint')
    dir_path = ckpt_dir / arch_name / "imagenet"
    dir_path.mkdir(parents=True, exist_ok=True)

    if arch_name == "resnet":
        model_name = resnet_name
    elif arch_name == "rexnet":
        model_name = rexnet_name
    elif arch_name == "efficientnet":
        model_name = efficient_name
    elif arch_name == "mobilenetv2":
        model_name = mobilenetv2_name

    for file_id, name in zip(file_ids, model_name):
        destination = dir_path / name
        download_file_from_google_drive(file_id, destination.as_posix())
