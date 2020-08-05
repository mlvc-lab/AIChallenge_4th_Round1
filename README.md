# AI-Challenge Base Code
  
You can train or test ResNet/MobileNet/MobileNetV2 on CIFAR10/CIFAR100/ImageNet.  
Specially, you can train or test on any device (CPU/sinlge GPU/multi GPU) and different device environment available.

----------

## Requirements

- `python 3.5+`
- `pytorch 1.4+`
- `torchvision 0.4+`
- `tqdm` (for downloading pretrained chekpoints)
- `sacred` (for logging on omniboard)
- `pymongo` (for logging on omniboard)

----------

## Files

- `config.py`: set configuration
- `data.py`: data loading
- `down_ckpt.py`: download pretrained checkpoints
- `main.py`: main python file for training or testing
- `models`
  - `__init__.py`
  - `mobilenet.py`
  - `mobilenetv2.py`
  - `resnet.py`
- `utils.py`

----------

## How to download a pretrained Model

The pretrained models of ResNet56 trained on CIFAR100 is only available now...

```text
usage: down_ckpt.py [-h] [-a ARCH] [--layers N] [--width-mult WM] [-o OUT]
                    DATA

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet (default:
                        cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 | resnet
                        (default: mobilenet)
  --layers N            number of layers in ResNet (default: 56)
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  -o OUT, --out OUT     output filename of pretrained model from our google
                        drive
```

### Usage

```shell
python down_ckpt.py cifar100 -a resnet --layers 56 -o pretrained_ckpt.pth
```

### ResNet56/CIFAR100 pretrained checkpoint experiment detail

- #epochs: 200
- batch size: 256
- initial learning rate: 0.1
- SGD
- momentum: 0.9
- weight decay: 5e-4
- no nesterov
- multi-step scheduler
  - gamma: 0.1
  - milestones: [100, 150]

----------

## How to train / test networks

```text
usage: main.py [-h] [-a ARCH] [--layers N] [--width-mult WM] [--datapath PATH]
               [-j N] [--epochs N] [-b N] [--lr LR] [--momentum M]
               [--wd WEIGHT_DECAY] [--nest] [--sched TYPE] [--step-size STEP]
               [--milestones EPOCH [EPOCH ...]] [--gamma GAMMA] [-p N]
               [--ckpt PATH] [-E] [-C] [-g GPU [GPU ...]] [-F]
               DATA

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet (default:
                        cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 | resnet
                        (default: resnet)
  --layers N            number of layers in ResNet (default: 56)
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  --datapath PATH       where you want to load/save your dataset? (default:
                        ../data)
  -j N, --workers N     number of data loading workers (default: 8)
  --epochs N            number of total epochs to run (default: 200)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay (default: 5e-4)
  --nest, --nesterov    use nesterov momentum?
  --sched TYPE, --scheduler TYPE
                        scheduler: step | multistep | exp | cosine (default:
                        step)
  --step-size STEP      period of learning rate decay / maximum number of
                        iterations for cosine annealing scheduler (default:
                        30)
  --milestones EPOCH [EPOCH ...]
                        list of epoch indices for multi step scheduler (must
                        be increasing) (default: 30 80)
  --gamma GAMMA         multiplicative factor of learning rate decay (default:
                        0.1)
  -p N, --print-freq N  print frequency (default: 100)
  --ckpt PATH           path of checkpoint for testing model (default: none)
  -E, --evaluate        test model?
  -C, --cuda            use cuda?
  -g GPU [GPU ...], --gpuids GPU [GPU ...]
                        GPU IDs for using (default: 0)
  -F, -finetune         finetuning?
```

### Training

#### Train a network using default scheduler (stepLR) with multi-GPU

```shell
python main.py cifar10 -a resnet --layers 56 -C -g 0 1 2 3
```

or

``` shell
python main.py cifar10 -a mobilenet -C -g 0 1 2 3
```

#### Train a network using multi-step scheduler with multi-GPU

```shell
python main.py cifar10 -a resnet --layers 56 -C -g 0 1 2 3 --scheduler multistep --milestones 100 150 --gamma 0.1
```

### Test

```shell
python main.py cifar10 -a resnet --layers 56 -C -g 0 1 2 3 -E --ckpt ckpt_best.pth
```

----------

## References

- [torchvision models github codes](https://github.com/pytorch/vision/tree/master/torchvision/models)
- [MobileNet Cifar GitHub (unofficial)](https://github.com/kuangliu/pytorch-cifar)
- [MobileNetV2 Cifar GitHub (unofficial)](https://github.com/tinyalpha/mobileNet-v2_cifar10)
